import os
import re
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse
from typing import Optional
import numpy as np
import torch
import torch.nn as nn

# HF Transformers (for SELFIES-TED decoder as in G2.py)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput

# Optional imports for web fetching
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# Optional imports for visuals
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
except Exception:
    Chem = None
    Draw = None

try:
    from matplotlib import cm
except Exception:
    cm = None

# Optional: joblib + sentencepiece for 5M generator artifacts
try:
    import joblib
except Exception:
    joblib = None

try:
    import sentencepiece as spm
except Exception:
    spm = None

# Optional: selfies (for SELFIES→SMILES/PSMILES conversion, as in G2)
try:
    import selfies as sf
except Exception:
    sf = None

RDKit_AVAILABLE = Chem is not None
SELFIES_AVAILABLE = sf is not None


# =============================================================================
# DOI NORMALIZATION / RESOLUTION HELPERS
# =============================================================================
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

def normalize_doi(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    # remove common prefixes
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:\s*", "", s, flags=re.IGNORECASE)
    # trim trailing punctuation often attached in text
    s = s.rstrip(").,;]}")
    return s if _DOI_RE.match(s) else None

def doi_to_url(doi: str) -> str:
    # doi is assumed normalized
    return f"https://doi.org/{doi}"

def doi_resolves(doi_url: str, timeout: float = 6.0) -> bool:
    """
    Best-effort resolver check. Keeps pipeline robust against dead/unregistered DOIs.
    If requests is unavailable, do not block.
    """
    if requests is None:
        return True
    try:
        r = requests.head(doi_url, allow_redirects=True, timeout=timeout)
        if r.status_code == 405:
            # Some resolvers disallow HEAD; fall back to a lightweight GET.
            r = requests.get(doi_url, allow_redirects=True, timeout=timeout, stream=True)
        return 200 <= r.status_code < 400
    except Exception:
        return False

# =============================================================================
# CITATION / DOMAIN TAGGING HELPERS (domain-style citations like "(nature.com)")
# =============================================================================
def _url_to_domain(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url.strip():
        return None
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return None
    try:
        netloc = urlparse(u).netloc.strip().lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Reduce to ROOT domain (nature.com, springer.com, etc.)
        parts = [p for p in netloc.split(".") if p]
        if len(parts) <= 2:
            return netloc or None

        second_level = {
            "co.uk", "ac.uk", "gov.uk", "org.uk",
            "co.jp", "ne.jp", "or.jp",
            "com.au", "net.au", "org.au", "edu.au",
            "co.in", "com.br", "com.cn",
        }

        last2 = ".".join(parts[-2:])
        last3 = ".".join(parts[-3:])

        if last2 in second_level and len(parts) >= 3:
            return last3
        if last3 in second_level and len(parts) >= 4:
            return ".".join(parts[-4:])

        return last2
    except Exception:
        return None

def _attach_source_domains(obj: Any) -> Any:
    """
    Recursively add a short source_domain field where URLs are present.
    This enables domain-style citations like "(nature.com)".
    """
    if isinstance(obj, list):
        return [_attach_source_domains(x) for x in obj]

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[k] = _attach_source_domains(v)

        for url_key in ("url", "landing_page", "landingPage", "doi_url", "pdf_url", "link", "href"):
            v = out.get(url_key)
            dom = _url_to_domain(v) if isinstance(v, str) else None
            if dom:
                out.setdefault("source_domain", dom)
                break
        return out

    return obj


def _index_citable_sources(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add 'cite_tag' fields for citable web/RAG items using DOI-first URL tags.
    Requirement:
      - Paper citations must use the COMPLETE DOI URL (https://doi.org/...) as the bracket text.
      - If DOI is not available, fall back to the best http(s) URL.
    Never uses numbered citations like [1], [2].
    """
    if not isinstance(report, dict):
        return report

    citation_index: Dict[str, Any] = {"sources": []}

    def is_citable_item(d: Dict[str, Any]) -> bool:
        if not isinstance(d, dict):
            return False
        for k in ("url", "landing_page", "landingPage", "doi_url", "pdf_url", "link", "href"):
            if isinstance(d.get(k), str) and (d[k].startswith("http://") or d[k].startswith("https://")):
                return True
        if isinstance(d.get("doi"), str) and d["doi"].strip():
            return True
        return False

    def get_best_url(d: Dict[str, Any]) -> Optional[str]:
        # DOI-first: if DOI exists, ALWAYS prefer doi.org for citation link text and href.
        doi = normalize_doi(d.get("doi", ""))
        if doi:
            return doi_to_url(doi)
        for k in ("url", "landing_page", "landingPage", "doi_url", "pdf_url", "link", "href"):
            v = d.get(k)
            if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")):
                return v
        return None

    def walk_and_tag(node: Any) -> Any:
        if isinstance(node, list):
            return [walk_and_tag(x) for x in node]

        if isinstance(node, dict):
            out = {k: walk_and_tag(v) for k, v in node.items()}

            if is_citable_item(out):
                # Citation tag MUST be DOI URL (preferred) or best URL (fallback).
                url = get_best_url(out)
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    # If an existing cite_tag is non-URL (e.g., a domain tag), replace it.
                    cur = out.get("cite_tag")
                    if not (isinstance(cur, str) and cur.strip().startswith(("http://", "https://"))):
                        out["cite_tag"] = url.strip()
                else:
                    # If we cannot form a URL, leave as-is (should be rare due to is_citable_item).
                    pass

                # Maintain a compact index (optional, harmless for UIs)
                url = get_best_url(out)
                dom = out.get("source_domain") or (_url_to_domain(url) if url else None) or "source"
                citation_index["sources"].append(
                    {
                        # tag is the bracket text requirement (DOI URL or URL)
                        "tag": out.get("cite_tag") if isinstance(out.get("cite_tag"), str) else url,
                        "domain": dom,
                        "title": out.get("title") or out.get("name") or "Untitled",
                        "url": url,
                        "doi": out.get("doi"),
                    }
                )
            return out

        return node

    tagged = walk_and_tag(report)
    if isinstance(tagged, dict):
        tagged.setdefault("citation_index", citation_index)
        return tagged

    report["citation_index"] = citation_index
    return report



 # =============================================================================
 # ENFORCE INLINE CLICKABLE LITERATURE CITATIONS (distributed, not clustered)
 # =============================================================================
_CITE_COUNT_PATTERNS = [
    r"(?:at\s+least\s+)?(\d{1,3})\s*(?:citations|citation|papers|paper|sources|source|references|reference)\b",
    r"\bcite\s+(\d{1,3})\s*(?:papers|paper|sources|source|references|reference|citations|citation)\b",
    r"\b(\d{1,3})\s*(?:papers|paper|sources|source|references|reference|citations|citation)\s*(?:minimum|min)\b",
]


def _infer_required_citation_count(text: str, default_n: int = 10) -> int:
    q = (text or "").lower()
    for pat in _CITE_COUNT_PATTERNS:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(n, 200))
            except Exception:
                pass
    return max(1, int(default_n))


def _collect_citation_links_from_report(report: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return unique (domain, url) pairs from report['citation_index']['sources'].
    Link text is strictly the root domain. URL must be http(s).
    """
    out: List[Tuple[str, str]] = []
    seen: set = set()
    if not isinstance(report, dict):
        return out
    ci = report.get("citation_index", {})
    sources = ci.get("sources") if isinstance(ci, dict) else None
    if not isinstance(sources, list):
        return out

    for s in sources:
        if not isinstance(s, dict):
            continue
        url = s.get("url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            continue
        # cite_text is DOI URL (tag) if present; else fall back to the URL itself.
        cite_text = s.get("tag") if isinstance(s.get("tag"), str) and s.get("tag").strip() else url
        if not isinstance(cite_text, str) or not cite_text.strip():
            cite_text = url
        cite_text = cite_text.strip()
        key = url.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append((cite_text, url.strip()))
    return out


def _ensure_distributed_inline_citations(md: str, report: Dict[str, Any], min_needed: int) -> str:
    """
    If the model fails to include enough inline clickable paper citations, inject them
    in a distributed way (one per eligible paragraph, outside code blocks).
    Tool citations ([T]) are NOT modified.
    """
    if not isinstance(md, str) or not md.strip():
        return md
    if not isinstance(report, dict):
        return md
    if min_needed <= 0:
        return md

    citations = _collect_citation_links_from_report(report)
    if not citations:
        return md

    # Count existing literature links by URL (any markdown link).
    existing_urls = set(re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", md))
    need = max(0, int(min_needed) - len(existing_urls))
    if need <= 0:
        return md

    # Only use citations not already present.
    remaining: List[Tuple[str, str]] = [(d, u) for (d, u) in citations if u not in existing_urls]
    if not remaining:
        return md

    # Split by fenced code blocks; do not inject inside them.
    parts = re.split(r"(```[\s\S]*?```)", md)
    rem_i = 0

    for pi, part in enumerate(parts):
        if rem_i >= len(remaining) or need <= 0:
            break
        if part.startswith("```") and part.endswith("```"):
            continue

        # Split into paragraph blocks (preserve blank-line separators).
        segs = re.split(r"(\n\s*\n)", part)
        for si in range(0, len(segs), 2):
            if rem_i >= len(remaining) or need <= 0:
                break
            para = segs[si]
            if not isinstance(para, str) or not para.strip():
                continue
            # Skip headings.
            if para.lstrip().startswith("#"):
                continue
            # Skip paragraphs that already contain at least one markdown link.
            if re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", para):
                continue

            # Prefer injecting into evidence-bearing paragraphs first to avoid "clutter".
            # If paragraph doesn't look like a literature-backed claim, skip it in this pass.
            if not re.search(r"\b(reported|shown|demonstrated|study|studies|literature|evidence|review|according)\b", para, flags=re.IGNORECASE):
                continue

            cite_text, url = remaining[rem_i]
            # Requirement: bracket text is the COMPLETE DOI URL (or URL fallback).
            segs[si] = para.rstrip() + f" [{cite_text}]({url})"
            rem_i += 1
            need -= 1

        parts[pi] = "".join(segs)

    # Second pass (if still need citations): allow any non-heading paragraph without links.
    if need > 0 and rem_i < len(remaining):
        md2 = "".join(parts)
        parts2 = re.split(r"(```[\s\S]*?```)", md2)
        for pi, part in enumerate(parts2):
            if rem_i >= len(remaining) or need <= 0:
                break
            if part.startswith("```") and part.endswith("```"):
                continue
            segs = re.split(r"(\n\s*\n)", part)
            for si in range(0, len(segs), 2):
                if rem_i >= len(remaining) or need <= 0:
                    break
                para = segs[si]
                if not isinstance(para, str) or not para.strip():
                    continue
                if para.lstrip().startswith("#"):
                    continue
                if re.search(r"\[[^\]]+\]\((https?://[^)]+)\)", para):
                    continue
                cite_text, url = remaining[rem_i]
                segs[si] = para.rstrip() + f" [{cite_text}]({url})"
                rem_i += 1
                need -= 1
            parts2[pi] = "".join(segs)
        return "".join(parts2)

    return "".join(parts)


def _normalize_and_dedupe_literature_links(md: str, report: Dict[str, Any]) -> str:
    """
    Enforce the single citation requirement:
      - Link text must be the COMPLETE DOI URL (preferred) or URL fallback.
      - Each DOI/URL must appear at most once in the entire answer.
    Only operates outside fenced code blocks.
    """
    if not isinstance(md, str) or not md.strip():
        return md
    if not isinstance(report, dict):
        return md

    # Build url -> preferred_text mapping (DOI URL / URL)
    url_to_text: Dict[str, str] = {}
    ci = report.get("citation_index", {})
    sources = ci.get("sources") if isinstance(ci, dict) else None
    if isinstance(sources, list):
        for s in sources:
            if not isinstance(s, dict):
                continue
            url = s.get("url")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            tag = s.get("tag")
            pref = tag.strip() if isinstance(tag, str) and tag.strip() else url.strip()
            url_to_text[url.strip()] = pref

    parts = re.split(r"(```[\s\S]*?```)", md)
    seen_urls: set = set()

    def _rewrite_and_dedupe(text: str) -> str:
        def repl(m: re.Match) -> str:
            txt = m.group(1)
            url = m.group(2).strip()
            if url in seen_urls:
                # remove duplicate citation entirely (and any leading space before it if present)
                return ""
            seen_urls.add(url)
            pref = url_to_text.get(url, url)
            return f"[{pref}]({url})"
        # Rewrite link text to preferred, then dedupe by URL
        return re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", repl, text)

    for i, part in enumerate(parts):
        if part.startswith("```") and part.endswith("```"):
            continue
        parts[i] = _rewrite_and_dedupe(part)
        # Cleanup: collapse double spaces created by removals
        parts[i] = re.sub(r"[ \t]{2,}", " ", parts[i])
        parts[i] = re.sub(r"\n{3,}", "\n\n", parts[i])

    return "".join(parts)


def autolink_doi_urls(md: str) -> str:
    """
    Wrap bare DOI URLs in Markdown links outside code blocks.
    Prevents plain DOI URLs from rendering as non-clickable text.
    """
    if not md:
        return md
    parts = re.split(r"(```[\s\S]*?```)", md)
    for i, part in enumerate(parts):
        if part.startswith("```") and part.endswith("```"):
            continue
        parts[i] = re.sub(
            r"(?<!\]\()(?P<u>https?://doi\.org/10\.\d{4,9}/[^\s\)\],;]+)",
            lambda m: f"[{m.group('u')}]({m.group('u')})",
           part,
            flags=re.IGNORECASE,
        )
    return "".join(parts)

def _assign_tool_tags_to_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each tool output has a [T#] tag for tool-citation style.
    This does NOT modify tool outputs beyond adding a 'cite_tag' key when missing.
    """
    if not isinstance(report, dict):
        return report

    tool_outputs = report.get("tool_outputs", {})
    if not isinstance(tool_outputs, dict):
        return report

    # Stable order (common core tools first)
    ordered_tools = [
        "data_extraction",
        "cl_encoding",
        "property_prediction",
        "polymer_generation",
        "rag_retrieval",
        "web_search",
        "report_generation",
    ]

    # Tag assignment: keep existing cite_tags if present
    tool_tag_map: Dict[str, str] = {}
    tag = "[T]"

    # First pass: assign in preferred order
    for tool in ordered_tools:
        node = tool_outputs.get(tool)
        if node is None:
            continue
        tool_tag_map[tool] = tag
        if isinstance(node, dict) and not node.get("cite_tag"):
            node["cite_tag"] = tag

    # Second pass: any remaining tools in tool_outputs
    for tool, node in tool_outputs.items():
        if tool in tool_tag_map or node is None:
            continue
        tool_tag_map[tool] = tag
        if isinstance(node, dict) and not node.get("cite_tag"):
            node["cite_tag"] = tag

    # Also tag summary nodes (best-effort, no structural assumptions)
    try:
        summary = report.get("summary", {}) or {}
        if isinstance(summary, dict):
            # common mapping
            key_to_tool = {
                "data_extraction": "data_extraction",
                "cl_encoding": "cl_encoding",
                "property_prediction": "property_prediction",
                "generation": "polymer_generation",
                "polymer_generation": "polymer_generation",
                "rag_retrieval": "rag_retrieval",
                "web_search": "web_search",
                "report_generation": "report_generation",
            }
            for k, tool in key_to_tool.items():
                node = summary.get(k)
                if isinstance(node, dict) and tool in tool_tag_map and not node.get("cite_tag"):
                    node["cite_tag"] = tool_tag_map[tool]
    except Exception:
        pass

    report.setdefault("tool_tag_index", tool_tag_map)
    return report


def _render_tool_outputs_verbatim_md(report: Dict[str, Any]) -> str:
    """
    Render tool outputs as verbatim JSON blocks (no tweaking of content values).
    """
    if not isinstance(report, dict):
        return ""

    tool_outputs = report.get("tool_outputs", {}) or {}
    if not isinstance(tool_outputs, dict):
        return ""

    # Prefer a stable display order; include any extra keys afterward
    preferred = [
        "data_extraction",
        "cl_encoding",
        "property_prediction",
        "polymer_generation",
        "rag_retrieval",
        "web_search",
        "report_generation",
    ]
    keys = [k for k in preferred if k in tool_outputs] + [k for k in tool_outputs.keys() if k not in preferred]

    chunks: List[str] = []
    for k in keys:
        out = tool_outputs.get(k)
        if out is None:
            continue
        tag = out.get("cite_tag") if isinstance(out, dict) else None
        header = f"### {tag} {k}" if isinstance(tag, str) and tag else f"### {k}"
        chunks.append(header)
        try:
            chunks.append("```json\n" + json.dumps(out, indent=2, ensure_ascii=False) + "\n```")
        except Exception:
            chunks.append("```text\n" + str(out) + "\n```")
    return "\n\n".join(chunks)


# =============================================================================
# PICKLE / JOBLIB COMPATIBILITY SHIMS (Fix generator loading error)
# =============================================================================
class LatentPropertyModel:
    """
    Compatibility shim for joblib/pickle artifacts saved with references like:
      __main__.LatentPropertyModel

    The original training code likely defined this in a script, so pickle recorded it under __main__.
    When loading from Gradio, __main__ is different, so unpickling fails.

    This shim is intentionally minimal:
      - pickle will restore attributes into this object
      - predict(...) attempts to delegate to a plausible underlying model attribute if present
    """
    def predict(self, X):
        # Common patterns: wrapper stores underlying estimator under one of these attributes.
        for attr in ("model", "gpr", "gpr_model", "estimator", "predictor", "_model", "_gpr"):
            if hasattr(self, attr):
                obj = getattr(self, attr)
                if hasattr(obj, "predict"):
                    return obj.predict(X)
        # If the wrapper itself has been restored with a custom predict, this will never be hit.
        raise AttributeError(
            "LatentPropertyModel shim could not find an underlying predictor. "
            "Artifact expects a wrapped model attribute with a .predict method."
        )


def _install_unpickle_shims() -> None:
    """
    Ensure that any classes pickled under __main__ are available at load time.
    This is critical for joblib artifacts created from scripts (training/fit scripts).
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "LatentPropertyModel"):
        setattr(main_mod, "LatentPropertyModel", LatentPropertyModel)


def _safe_joblib_load(path: str):
    """
    joblib.load wrapper that patches __main__ symbols on common pickle failures and retries once.
    """
    if joblib is None:
        raise RuntimeError("joblib not installed but required to load *.joblib artifacts (pip install joblib).")

    try:
        return joblib.load(path)
    except Exception as e:
        msg = str(e)
        # Targeted fix for your exact failure mode
        if "Can't get attribute 'LatentPropertyModel' on <module '__main__'" in msg:
            _install_unpickle_shims()
            return joblib.load(path)
        raise


# =============================================================================
# PATHS (per your 5M pipeline artifacts)
# =============================================================================
DOWNSTREAM_BESTWEIGHTS_5M_DIR = "/home/kaur-m43/multimodal_downstream_bestweights_5M"
INVERSE_DESIGN_5M_DIR = "/home/kaur-m43/multimodal_inverse_design_output_5M_polybart_style/best_models"


# =============================================================================
# Property name canonicalization
# =============================================================================
def canonical_property_name(name: str) -> str:
    """
    Map user/tool inputs to the canonical keys used in PROPERTY_HEAD_PATHS/GENERATOR_DIRS.
    """
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)

    aliases = {
        "tg": "glass transition",
        "glass transition temperature": "glass transition",
        "glass transition temp": "glass transition",
        "glass transition (tg)": "glass transition",
        "t g": "glass transition",

        "td": "thermal decomposition",
        "thermal decomp": "thermal decomposition",
        "thermal decomposition temperature": "thermal decomposition",

        "sv": "specific volume",
    }
    return aliases.get(s, s)


# =============================================================================
# NEW: best-effort inference of property + target_value from questions text
# (used only when callers omit property/target_value but provide questions)
# =============================================================================
_NUM_RE = r"[-+]?\d+(?:\.\d+)?"

def infer_property_from_text(text: str) -> Optional[str]:
    s = (text or "").lower()
    # explicit "property: ..."
    m = re.search(r"\bproperty\b\s*[:=]\s*([a-zA-Z _-]+)", s)
    if m:
        cand = m.group(1).strip().lower()
        if "glass" in cand or re.search(r"\btg\b", cand):
            return "glass transition"
        if "density" in cand or re.search(r"\brho\b", cand):
            return "density"
        if "melting" in cand or re.search(r"\btm\b", cand):
            return "melting"
        if "specific" in cand or re.search(r"\bsv\b", cand):
            return "specific volume"
        if "decomp" in cand or "decomposition" in cand or re.search(r"\btd\b", cand):
            return "thermal decomposition"

    if "thermal decomposition" in s or "decomposition temperature" in s or "decomposition" in s or re.search(r"\btd\b", s):
        return "thermal decomposition"
    if "specific volume" in s or re.search(r"\bsv\b", s):
        return "specific volume"
    if "glass transition" in s or "glass-transition" in s or re.search(r"\btg\b", s):
        return "glass transition"
    if "melting" in s or "melt temperature" in s or re.search(r"\btm\b", s):
        return "melting"
    if "density" in s or re.search(r"\brho\b", s):
        return "density"
    return None

def infer_target_value_from_text(text: str, prop: Optional[str]) -> Optional[float]:
    sl = (text or "").lower()

    m = re.search(rf"\b(target_value|target|tgt)\b\s*[:=]?\s*({_NUM_RE})", sl)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            pass

    prop = canonical_property_name(prop or "") if prop else ""
    patterns = []
    if prop == "glass transition":
        patterns = [rf"\b(tg|glass\s*transition)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "density":
        patterns = [rf"\b(density|rho)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "melting":
        patterns = [rf"\b(tm|melting)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "specific volume":
        patterns = [rf"\b(specific\s*volume|sv)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "thermal decomposition":
        patterns = [rf"\b(td|thermal\s*decomposition|decomposition)\b\s*[:=]?\s*({_NUM_RE})"]

    for pat in patterns:
        m = re.search(pat, sl)
        if m:
            try:
                return float(m.group(m.lastindex))
            except Exception:
                pass

    # token-near-number fallback (within 80 chars)
    tokens = []
    if prop == "glass transition":
        tokens = ["tg", "glass transition"]
    elif prop == "density":
        tokens = ["density", "rho"]
    elif prop == "melting":
        tokens = ["tm", "melting"]
    elif prop == "specific volume":
        tokens = ["specific volume", "sv"]
    elif prop == "thermal decomposition":
        tokens = ["td", "thermal decomposition", "decomposition"]

    for tok in tokens:
        for mt in re.finditer(re.escape(tok), sl):
            window = sl[mt.end():mt.end() + 80]
            mn = re.search(rf"({_NUM_RE})", window)
            if mn:
                try:
                    return float(mn.group(1))
                except Exception:
                    pass

    return None

PROPERTY_HEAD_PATHS = {
    "density": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "density", "best_run_checkpoint.pt"),
    "glass transition": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "glass_transition", "best_run_checkpoint.pt"),
    "melting": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "melting", "best_run_checkpoint.pt"),
    "specific volume": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "specific_volume", "best_run_checkpoint.pt"),
    "thermal decomposition": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "thermal_decomposition", "best_run_checkpoint.pt"),
}

PROPERTY_HEAD_META = {
    "density": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "density", "best_run_metadata.json"),
    "glass transition": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "glass_transition", "best_run_metadata.json"),
    "melting": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "melting", "best_run_metadata.json"),
    "specific volume": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "specific_volume", "best_run_metadata.json"),
    "thermal decomposition": os.path.join(DOWNSTREAM_BESTWEIGHTS_5M_DIR, "thermal_decomposition", "best_run_metadata.json"),
}

GENERATOR_DIRS = {
    "density": os.path.join(INVERSE_DESIGN_5M_DIR, "density"),
    "glass transition": os.path.join(INVERSE_DESIGN_5M_DIR, "glass_transition"),
    "melting": os.path.join(INVERSE_DESIGN_5M_DIR, "melting"),
    "specific volume": os.path.join(INVERSE_DESIGN_5M_DIR, "specific_volume"),
    "thermal decomposition": os.path.join(INVERSE_DESIGN_5M_DIR, "thermal_decomposition"),
}


# =============================================================================
# Tokenizers (SentencePiece etc. — kept for backward compatibility)
# =============================================================================
class SimpleCharTokenizer:
    def __init__(self, vocab_chars: List[str], special_tokens=("<pad>", "<s>", "</s>", "<unk>")):
        self.special_tokens = list(special_tokens)
        chars = [c for c in vocab_chars if c not in self.special_tokens]
        self.vocab = list(self.special_tokens) + chars
        self.piece_to_id = {p: i for i, p in enumerate(self.vocab)}
        self.id_to_piece = {i: p for i, p in enumerate(self.vocab)}

    def encode(self, text: str, out_type=int):
        return [self.piece_to_id.get(ch, self.piece_to_id.get("<unk>")) for ch in text]

    def decode(self, ids: List[int]) -> str:
        pieces = [self.id_to_piece.get(int(i), "") for i in ids]
        return "".join([p for p in pieces if p not in self.special_tokens])

    def PieceToId(self, piece: str) -> Optional[int]:
        return self.piece_to_id.get(piece, None)

    def IdToPiece(self, idx: int) -> str:
        return self.id_to_piece.get(int(idx), "")

    def get_piece_size(self) -> int:
        return len(self.vocab)


class SentencePieceTokenizerWrapper:
    """
    Minimal wrapper with:
      - encode(text) -> List[int]
      - decode(ids) -> str
      - PieceToId(piece) / IdToPiece(id)
      - get_piece_size()
      - special_tokens and optional _blocked_ids
    """
    def __init__(self, model_path: str):
        if spm is None:
            raise RuntimeError("sentencepiece not installed but required for spm_5M.model (pip install sentencepiece).")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.Load(model_path)
        if not ok:
            raise RuntimeError(f"Failed to load SentencePiece model at: {model_path}")

        self.special_tokens = []
        for t in ("<pad>", "<s>", "</s>", "<unk>"):
            if self.sp.PieceToId(t) >= 0:
                self.special_tokens.append(t)

        blocked = []
        for t in ("<pad>", "<unk>"):
            tid = self.PieceToId(t)
            if tid is not None:
                blocked.append(tid)
        setattr(self, "_blocked_ids", blocked)

        # Safety: require '*' token
        if self.PieceToId("*") is None:
            raise RuntimeError("SentencePiece tokenizer loaded but '*' token not found – aborting for safe PSMILES generation.")

    def encode(self, text: str, out_type=int):
        return list(self.sp.EncodeAsIds(text))

    def decode(self, ids: List[int]) -> str:
        return self.sp.DecodeIds(list(map(int, ids)))

    def PieceToId(self, piece: str) -> Optional[int]:
        pid = self.sp.PieceToId(piece)
        return None if pid < 0 else int(pid)

    def IdToPiece(self, idx: int) -> str:
        return self.sp.IdToPiece(int(idx))

    def get_piece_size(self) -> int:
        return int(self.sp.GetPieceSize())


def normalize_generated_psmiles_out(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r"\[\*\]", "*", s)


def psmiles_to_rdkit_smiles(psmiles: str) -> str:
    """
    RDKit typically expects wildcard as [*]. Convert '*' -> '[*]' (but keep already-bracketed wildcards).
    """
    if not isinstance(psmiles, str):
        return ""
    s = psmiles
    if "*" in s and "[*]" not in s:
        s = re.sub(r"\*", "[*]", s)
    return s

# --- UI-safe endpoint normalization (ONLY [At]/[AT] -> [*]) ---
_AT_BRACKET_UI_RE = re.compile(r"\[(at)\]", flags=re.IGNORECASE)

def replace_at_with_star(psmiles: str) -> str:
    if not isinstance(psmiles, str) or not psmiles:
        return psmiles
    return _AT_BRACKET_UI_RE.sub("[*]", psmiles)

# =============================================================================
# SELFIES utilities (minimal subset mirroring G2.py behaviour)
# =============================================================================
_SELFIES_TOKEN_RE = re.compile(r"\[[^\[\]]+\]")


def _selfies_compact(selfies_str: str) -> str:
    return str(selfies_str).replace(" ", "").strip()


def _ensure_two_at_endpoints(selfies_str: str) -> str:
    """
    Simple endpoint regularization. For polymer-style SELFIES this would
    normally enforce two special endpoints; here we just compact.
    """
    return _selfies_compact(selfies_str)


def selfies_to_smiles(selfies_str: str) -> str:
    """
    Decode SELFIES to a canonical SMILES using RDKit, if available.
    """
    if not SELFIES_AVAILABLE:
        return _selfies_compact(selfies_str)

    try:
        s = _selfies_compact(selfies_str)
        smi = sf.decoder(s)
        if not isinstance(smi, str) or not smi:
            return s
        if not RDKit_AVAILABLE:
            return smi
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except Exception:
            return smi
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return _selfies_compact(selfies_str)


def pselfies_to_psmiles(selfies_str: str) -> str:
    """
    For this orchestrator we treat pSELFIES→PSMILES as SELFIES→canonical SMILES.
    The G2 training script used a more elaborate At/[*] polymer mapping; if you
    want that exact behaviour, you can replace this with the full pselfies_to_psmiles
    utilities from G2.py.
    """
    return selfies_to_smiles(selfies_str)


# =============================================================================
# SELFIES-TED decoder (as in G2.py, but simplified to core functionality)
# =============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN", None)
SELFIES_TED_MODEL_NAME = os.environ.get("SELFIES_TED_MODEL_NAME", "ibm-research/materials.selfies-ted")

# Generation hyperparameters (mirroring G2 defaults)
GEN_MAX_LEN = 256
GEN_MIN_LEN = 10
GEN_TOP_P = 0.92
GEN_TEMPERATURE = 1.0
GEN_REPETITION_PENALTY = 1.05
LATENT_NOISE_STD_GEN = 0.15   # default exploration std for generation


def _hf_load_with_retries(load_fn, max_tries: int = 5, base_sleep: float = 2.0):
    """
    Small helper to make HF loading more robust, copied from G2 spirit.
    """
    import time
    last_err = None
    for t in range(max_tries):
        try:
            return load_fn()
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (1.6 ** t) + np.random.rand()
            print(f"[WARN] HF load attempt {t+1}/{max_tries} failed: {e}. Sleeping {sleep_s:.1f}s then retry.")
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to load model from HF. Last error: {last_err}")


def load_selfies_ted_and_tokenizer(model_name: str = SELFIES_TED_MODEL_NAME):
    """
    Load tokenizer + seq2seq model for SELFIES-TED, exactly as in G2.py (but without side effects).
    """
    def _load_tok():
        return AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, use_fast=True)

    def _load_model():
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)

    tok = _hf_load_with_retries(_load_tok, max_tries=5)
    model = _hf_load_with_retries(_load_model, max_tries=5)
    return tok, model


class CLConditionedSelfiesTEDGenerator(nn.Module):
    """
    Same structure as in G2.py: take a CL embedding (latent) and project it
    into a fixed-length memory that conditions a SELFIES-TED seq2seq model.
    """
    def __init__(self, tok, seq2seq_model, cl_emb_dim: int = 600, mem_len: int = 4):
        super().__init__()
        self.tok = tok
        self.model = seq2seq_model
        self.mem_len = int(mem_len)
        self.cl_emb_dim = int(cl_emb_dim)

        d_model = int(getattr(self.model.config, "d_model", getattr(self.model.config, "hidden_size", 1024)))
        self.cl_to_d = nn.Sequential(
            nn.Linear(self.cl_emb_dim, d_model),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
        )
        self.mem_pos = nn.Embedding(self.mem_len, d_model)

    def build_encoder_outputs(self, z: torch.Tensor) -> Tuple[BaseModelOutput, torch.Tensor]:
        device = z.device
        B = z.size(0)
        d = self.cl_to_d(z)  # (B, d_model)
        d = d.unsqueeze(1).expand(B, self.mem_len, d.size(-1)).contiguous()
        pos = torch.arange(self.mem_len, device=device).unsqueeze(0).expand(B, -1)
        d = d + self.mem_pos(pos)
        attn = torch.ones((B, self.mem_len), dtype=torch.long, device=device)
        return BaseModelOutput(last_hidden_state=d), attn

    def forward_train(self, z: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out, attn = self.build_encoder_outputs(z)
        out = self.model(
            encoder_outputs=enc_out,
            attention_mask=attn,
            labels=labels,
        )
        loss = out.loss
        return {"loss": loss, "ce": loss.detach()}

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        num_return_sequences: int = 1,
        max_len: int = GEN_MAX_LEN,
        top_p: float = GEN_TOP_P,
        temperature: float = GEN_TEMPERATURE,
        repetition_penalty: float = GEN_REPETITION_PENALTY,
    ) -> List[str]:
        """
        Latent→pSELFIES generation, as in G2.
        """
        self.eval()
        z = z.to(next(self.parameters()).device)
        enc_out, attn = self.build_encoder_outputs(z)
        gen = self.model.generate(
            encoder_outputs=enc_out,
            attention_mask=attn,
            do_sample=True,
            top_p=float(top_p),
            temperature=float(temperature),
            repetition_penalty=float(repetition_penalty),
            num_return_sequences=int(num_return_sequences),
            max_length=int(max_len),
            min_length=int(GEN_MIN_LEN),
            pad_token_id=int(self.tok.pad_token_id) if self.tok.pad_token_id is not None else None,
            eos_token_id=int(self.tok.eos_token_id) if self.tok.eos_token_id is not None else None,
        )
        outs = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        outs = [_ensure_two_at_endpoints(_selfies_compact(o)) for o in outs]
        return outs


# =============================================================================
# Latent→property helper (uses G2-style LatentPropertyModel joblib artifacts)
# =============================================================================
def _predict_latent_property(latent_model: Any, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mirror G2's predict_latent_property(model, z):
      - PCA transform if present
      - GPR predict (scaled y)
      - inverse-transform via y_scaler if present
    """
    z_use = np.asarray(z, dtype=np.float32)
    if z_use.ndim == 1:
        z_use = z_use.reshape(1, -1)

    # Optional PCA
    pca = getattr(latent_model, "pca", None)
    if pca is not None:
        z_use = pca.transform(z_use.astype(np.float32))

    # GPR or wrapped predictor
    gpr = getattr(latent_model, "gpr", None)
    if gpr is not None and hasattr(gpr, "predict"):
        y_s = gpr.predict(z_use)
    elif hasattr(latent_model, "predict"):
        y_s = latent_model.predict(z_use)
    else:
        raise RuntimeError("Latent property model has no usable predictor (expected .gpr or .predict).")

    y_s = np.array(y_s, dtype=np.float32).reshape(-1)

    # Optional scaler to get back to original units
    y_scaler = getattr(latent_model, "y_scaler", None)
    if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
        y_u = y_scaler.inverse_transform(y_s.reshape(-1, 1)).reshape(-1)
    else:
        y_u = y_s.copy()

    return y_s.astype(np.float32), y_u.astype(np.float32)


# =============================================================================
# Legacy models (kept for backward compatibility; not used in new generation path)
# =============================================================================
class TransformerDecoderOnly(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 8,
        nhead: int = 10,
        ff_mult: int = 4,
        dropout: float = 0.1,
        tie_embeddings: Optional[nn.Embedding] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.token_emb = tie_embeddings if tie_embeddings is not None else nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(4096, hidden_size)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_embeddings is not None:
            self.lm_head.weight = tie_embeddings.weight

    def _make_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ):
        B, Ld = decoder_input_ids.size()
        device = decoder_input_ids.device
        pos_ids = torch.arange(Ld, device=device).unsqueeze(0).expand(B, Ld)
        x = self.token_emb(decoder_input_ids) + self.pos_emb(pos_ids)

        tgt_mask = self._make_causal_mask(Ld, device)
        tgt_key_padding_mask = (decoder_attention_mask == 0) if decoder_attention_mask is not None else None

        y = self.decoder(
            tgt=x,
            memory=encoder_hidden_states,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None
        )
        y = self.ln_f(y)
        return self.lm_head(y)


class InverseDesignDecoder(nn.Module):
    """
    Legacy decoder-only inverse design model (kept for backward compatibility).
    The new generation path uses CLConditionedSelfiesTEDGenerator instead.
    """
    def __init__(self, vocab_size: int, hidden_size: int = 600, latent_dim: int = 600,
                 num_memory_tokens: int = 8, decoder_layers: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_memory_tokens = num_memory_tokens

        self.memory_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * num_memory_tokens)
        )

        self.decoder = TransformerDecoderOnly(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            nhead=10,
            ff_mult=4,
            dropout=0.1,
            tie_embeddings=None
        )

    def encode_memory_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        memory_flat = self.memory_proj(latent)
        return memory_flat.view(latent.size(0), self.num_memory_tokens, self.hidden_size)


# =============================================================================
# Config
# =============================================================================
class OrchestratorConfig:
    def __init__(self):
        self.base_dir = "."
        self.cl_weights_path = "/home/kaur-m43/multimodal_output_5M/best/pytorch_model.bin"
        self.chroma_db_path = "chroma_polymer_db_big"
        self.rag_embedding_model = "text-embedding-3-small"

        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.spm_model_path = "/home/kaur-m43/spm_5M.model"
        self.spm_vocab_path = "/home/kaur-m43/spm_5M.vocab"

        self.springer_api_key = os.getenv("SPRINGER_NATURE_API_KEY", "")
        self.semantic_scholar_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

        self.available_tools = {
            "data_extraction": True,
            "rag_retrieval": True,
            "cl_encoding": True,
            "property_prediction": True,
            "polymer_generation": True,
            "web_search": True,
            "report_generation": True,   # <-- FIX: required by the Gradio interface
            "mol_render": True,
            "gen_grid": True,
            "prop_attribution": True,
        }


TOOL_DESCRIPTIONS = {
    "data_extraction": {
        "name": "Extract Polymer Multimodal Data",
        "description": "Extracts graphs, 3D geometry, fingerprints, and PSMILES",
        "input": "PSMILES string or CSV file path",
        "output": "JSON with graph, geometry, fingerprints, and canonical PSMILES",
    },
    "rag_retrieval": {
        "name": "RAG Knowledge Base Query",
        "description": "Retrieves relevant literature from local polymer KB (Chroma)",
    },
    "cl_encoding": {
        "name": "Contrastive Learning Encoder",
        "description": "Encodes polymers using pretrained 4-encoder CL system",
    },
    "property_prediction": {
        "name": "Property Prediction (5M best_run_checkpoint + normalization)",
        "description": (
            "Predicts polymer properties using CL embeddings + best_run_checkpoint.pt "
            "and applies saved normalization to return values in original units. "
            "Prefers embeddings from cl_encoding if present."
        ),
    },
    "polymer_generation": {
        "name": "Inverse Design Generation (5M PolyBART-style)",
        "description": (
            "Generates polymer PSMILES from a target property using StandardScaler + GPR "
            "+ decoder_best_fold*.pt + SELFIES-TED backbone (as in G2.py). "
            "Requires target_value; optionally uses CL embeddings from cl_encoding or "
            "seed_psmiles to bias the latent sampling."
        ),
    },
    "web_search": {
        "name": "On-the-fly Literature Search (real & virtual libraries)",
        "description": (
            "CrossRef, OpenAlex, EuropePMC, arXiv, Semantic Scholar, Springer Nature (API key), Internet Archive"
        ),
    },
    "report_generation": {  # <-- FIX: required by the Gradio interface
        "name": "Report Generation",
        "description": (
            "Synthesizes available tool outputs into a single structured report object "
            "(summary + tool outputs) that can be rendered by the UI."
        ),
    },
    "mol_render": {
        "name": "Molecule Rendering",
        "description": "2D render of PSMILES with optional highlights (PNG)",
    },
    "gen_grid": {
        "name": "Generation Grid",
        "description": "Grid of generated polymers with optional score badges (PNG)",
    },
    "prop_attribution": {
        "name": "Property Attribution",
        "description": (
            "Per-atom attribution heatmap for predictions using leave-one-atom-out occlusion "
            "and top-K highlighting (PNG)."
        ),
    },
}


# =============================================================================
# Orchestrator
# =============================================================================
class PolymerOrchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._openai_client = None
        self._openai_unavailable_reason = None
        self._data_extractor = None
        self._rag_retriever = None
        self._cl_encoder = None
        self._psmiles_tokenizer = None

        # cached: (head_module, y_scaler, meta, ckpt_path)
        self._property_heads: Dict[str, Tuple[torch.nn.Module, Any, Dict[str, Any], str]] = {}
        # cached: (decoder_model, latent_prop_model, scaler_y, selfies_tok, meta, paths)
        self._property_generators: Dict[str, tuple] = {}
        # cached SELFIES-TED backbones keyed by model name
        self._selfies_ted_cache: Dict[str, Tuple[Any, Any]] = {}

        self.system_prompt = self._build_system_prompt()

    @property
    def openai_client(self):
        if self._openai_client is None:
            try:
                from openai import OpenAI
                if not self.config.openai_api_key:
                    self._openai_unavailable_reason = "OPENAI_API_KEY missing or empty"
                    self._openai_client = None
                else:
                    self._openai_client = OpenAI(api_key=self.config.openai_api_key)
            except Exception as e:
                self._openai_unavailable_reason = f"OpenAI client init failed: {e}"
                self._openai_client = None
        return self._openai_client

    def _build_system_prompt(self) -> str:
        tools_info = json.dumps(TOOL_DESCRIPTIONS, indent=2)
        available = [k for k, v in self.config.available_tools.items() if v]
        return (
            "You are the tool-planning module for **PolyAgent**, a polymer-science agent.\n"
            "Your job is to inspect the user's questions and decide which tools\n"
            "to run in which order. \n\n"
            "Critical tool dependencies:\n"
            "- property_prediction should run AFTER cl_encoding when possible and should reuse cl_encoding.embedding.\n"
            "- polymer_generation is inverse-design and REQUIRES target_value (property -> PSMILES).\n\n"
            f"Available tools (JSON spec):\n{tools_info}\n\n"
            f"Enabled: {', '.join(available)}"
        )

    # =============================================================================
    # Planner: LLM tool-calling (no rule-based planner)
    # =============================================================================
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        schema_keys = ["analysis", "tools_required", "execution_plan"]

        if self.openai_client is None:
            return {
                "analysis": user_query,
                "tools_required": [],
                "execution_plan": [],
                "note": f"OpenAI unavailable ({self._openai_unavailable_reason or 'unknown'})."
            }

        sys_prompt = (
            self.system_prompt
            + "\nYou must create a tool execution plan. Do not answer the science.\n"
            + "Return a plan with keys exactly: "
            + json.dumps(schema_keys)
        )

        plan_tool = {
            "type": "function",
            "function": {
                "name": "make_plan",
                "description": "Create a tool execution plan for PolyAgent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis": {"type": "string"},
                        "tools_required": {"type": "array", "items": {"type": "string"}},
                        "execution_plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "step": {"type": "integer"},
                                    "tool": {"type": "string"},
                                    "action": {"type": "string"},
                                    "input": {"type": "string"},
                                },
                                "required": ["step", "tool", "action"]
                            }
                        }
                    },
                    "required": ["analysis", "tools_required", "execution_plan"]
                }
            }
        }

        # Preferred: function/tool-calling
        try:
            response = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_query},
                ],
                tools=[plan_tool],
                tool_choice={"type": "function", "function": {"name": "make_plan"}},
                temperature=0.2,
                max_tokens=700,
            )

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                args = tool_calls[0].function.arguments
                plan = json.loads(args)
                for k in schema_keys:
                    if k not in plan:
                        raise ValueError(f"Missing key '{k}' in tool plan")
                return plan

            raise RuntimeError("Tool-calling plan not returned; falling back to JSON mode.")
        except Exception:
            # Safe fallback: JSON response_format (still LLM-generated, not rule-based)
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": sys_prompt + "\nReturn ONLY a JSON object and nothing else."},
                        {"role": "user", "content": user_query},
                    ],
                    temperature=0.2,
                    max_tokens=700,
                    response_format={"type": "json_object"},
                )
                plan = json.loads(response.choices[0].message.content)
                for k in schema_keys:
                    if k not in plan:
                        raise ValueError(f"Missing key '{k}' in model response")
                return plan
            except Exception as e:
                return {
                    "analysis": user_query,
                    "tools_required": [],
                    "execution_plan": [],
                    "note": f"OpenAI planning failed: {str(e)}"
                }

    def execute_plan(self, plan: Dict[str, Any], user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        results = {"plan": plan, "steps": [], "final_output": None, "errors": []}
        intermediate_data = user_inputs or {}

        for step in plan.get("execution_plan", []):
            step_num = step.get("step", 0)
            tool_name = step.get("tool", "")
            action = step.get("action", "")
            try:
                if tool_name == "data_extraction":
                    output = self._run_data_extraction(step, intermediate_data)
                elif tool_name == "rag_retrieval":
                    output = self._run_rag_retrieval(step, intermediate_data)
                elif tool_name == "cl_encoding":
                    output = self._run_cl_encoding(step, intermediate_data)
                elif tool_name == "property_prediction":
                    output = self._run_property_prediction(step, intermediate_data)
                elif tool_name == "polymer_generation":
                    output = self._run_polymer_generation(step, intermediate_data)
                elif tool_name == "web_search":
                    output = self._run_web_search(step, intermediate_data)
                elif tool_name == "report_generation":  # <-- FIX
                    output = self._run_report_generation(step, intermediate_data)
                elif tool_name == "mol_render":
                    output = self._run_mol_render(step, intermediate_data)
                elif tool_name == "gen_grid":
                    output = self._run_gen_grid(step, intermediate_data)
                elif tool_name == "prop_attribution":
                    output = self._run_prop_attribution(step, intermediate_data)
                else:
                    output = {"error": f"Unknown tool: {tool_name}"}

                results["steps"].append({"step": step_num, "tool": tool_name, "action": action, "output": output})
                intermediate_data[f"step_{step_num}_output"] = output
                intermediate_data[tool_name] = output
            except Exception as e:
                results["errors"].append(f"Error in step {step_num} ({tool_name}): {str(e)}")

        if results["steps"]:
            results["final_output"] = results["steps"][-1]["output"]
        return results

    # ----------------- Data extraction ----------------- #
    def _run_data_extraction(self, step: Dict, data: Dict) -> Dict:
        if self._data_extractor is None:
            try:
                from Data_Modalities import AdvancedPolymerMultimodalExtractor
            except Exception as e:
                return {"error": f"Data_Modalities import failed: {e}"}
            self._data_extractor = AdvancedPolymerMultimodalExtractor(csv_file="")

        psmiles = data.get("psmiles", data.get("smiles", "")) or data.get("seed_psmiles", "")
        if not psmiles:
            return {"error": "No PSMILES provided"}

        canonical = self._data_extractor.validate_and_standardize_smiles(psmiles)
        if not canonical:
            return {"error": f"Invalid PSMILES: {psmiles}"}

        return {
            "canonical_psmiles": canonical,
            "graph": self._data_extractor.generate_molecular_graph(canonical),
            "geometry": self._data_extractor.optimize_3d_geometry(canonical),
            "fingerprints": self._data_extractor.calculate_morgan_fingerprints(canonical),
        }

    # ----------------- RAG retrieval ----------------- #
    def _run_rag_retrieval(self, step: Dict, data: Dict) -> Dict:
        try:
            from rag_pipeline import (
                build_retriever_from_web,
                build_retriever,
                POLYMER_KEYWORDS,
                DEFAULT_TMP_DOWNLOAD_DIR,
                DEFAULT_MAILTO,
                PolymerStyleOpenAIEmbeddings,
            )
            from langchain_community.vectorstores import Chroma
        except Exception as e:
            return {"error": f"Could not import polymer rag_pipeline: {e}"}

        if self._rag_retriever is None:
            try:
                persist_dir = self.config.chroma_db_path
                if os.path.isdir(persist_dir) and os.listdir(persist_dir):
                    embeddings = PolymerStyleOpenAIEmbeddings(
                        model=self.config.rag_embedding_model,
                        api_key=self.config.openai_api_key
                    )
                    vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
                    self._rag_retriever = vector_store.as_retriever(search_kwargs={"k": 6})
                else:
                    papers_dir = DEFAULT_TMP_DOWNLOAD_DIR
                    pdfs_present = os.path.isdir(papers_dir) and any(f.lower().endswith(".pdf") for f in os.listdir(papers_dir))
                    if pdfs_present:
                        self._rag_retriever = build_retriever(
                            papers_path=papers_dir,
                            persist_dir=persist_dir,
                            k=6,
                            embedding_model=self.config.rag_embedding_model,
                            vector_backend="chroma",
                        )
                    else:
                        self._rag_retriever = build_retriever_from_web(
                            polymer_keywords=POLYMER_KEYWORDS,
                            persist_dir=persist_dir,
                            tmp_download_dir=papers_dir,
                            k=6,
                            embedding_model=self.config.rag_embedding_model,
                            vector_backend="chroma",
                            mailto=DEFAULT_MAILTO,
                        )
            except Exception as e:
                return {"error": f"Failed to initialize RAG retriever: {e}"}

        query = data.get("query", data.get("question", step.get("input", ""))) or ""
        if not query:
            return {"error": "No query provided"}

        try:
            docs = self._rag_retriever.get_relevant_documents(query)
        except Exception as e:
            return {"error": f"RAG retrieval failed: {e}"}

        results = []
        for i, doc in enumerate(docs or [], 1):
            meta = getattr(doc, "metadata", {}) or {}
            page_content = getattr(doc, "page_content", "") or ""
            results.append({
                "rank": i,
                "content": page_content[:800],
                "title": meta.get("title", "Unknown"),
                "year": meta.get("year", ""),
                "source": meta.get("source", meta.get("source_path", "")),
                "venue": meta.get("venue", meta.get("journal", "")),
                # NEW: preserve citable identifiers when present in metadata
                "url": meta.get("url") or meta.get("link") or meta.get("href") or "",
                "doi": meta.get("doi") or "",
            })
        return {"query": query, "results": results}

    # ----------------- CL encoding ----------------- #
    def _ensure_cl_encoder(self):
        if self._cl_encoder is None:
            try:
                from cl_models import (
                    GineEncoder, NodeSchNetWrapper, FingerprintEncoder,
                    PSMILESDebertaEncoder, MultimodalContrastiveModel, build_psmiles_tokenizer
                )
            except Exception:
                raise RuntimeError("cl_models not available in python path")

            if self._psmiles_tokenizer is None:
                self._psmiles_tokenizer = build_psmiles_tokenizer()
            vocab_sz = getattr(self._psmiles_tokenizer, "vocab_size", 270)

            gine = GineEncoder().to(self.config.device)
            schnet = NodeSchNetWrapper().to(self.config.device)
            fp = FingerprintEncoder().to(self.config.device)
            psm = PSMILESDebertaEncoder(model_dir_or_name=None, hidden_fallback=600, vocab_fallback=vocab_sz).to(self.config.device)
            model = MultimodalContrastiveModel(gine, schnet, fp, psm, emb_dim=600).to(self.config.device)

            try:
                state_dict = torch.load(self.config.cl_weights_path, map_location=self.config.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

            model.eval()
            self._cl_encoder = model

    def _prepare_batch_from_extraction(self, multimodal_data: Dict) -> Dict:
        batch: Dict[str, Dict[str, torch.Tensor]] = {}

        # graph
        if "graph" in multimodal_data:
            graph = multimodal_data["graph"]
            node_features = graph.get("node_features", [])
            if len(node_features) > 0:
                atomic_nums, chirality, formal_charge = [], [], []
                for nf in node_features:
                    atomic_nums.append(int(nf.get("atomic_num", nf.get("atomic_number", 6))))
                    chirality.append(float(nf.get("chirality", 0)))
                    formal_charge.append(float(nf.get("formal_charge", 0)))

                z_tensor = torch.tensor(atomic_nums, dtype=torch.long, device=self.config.device)
                chirality_tensor = torch.tensor(chirality, dtype=torch.float, device=self.config.device)
                formal_charge_tensor = torch.tensor(formal_charge, dtype=torch.float, device=self.config.device)

                edge_indices = graph.get("edge_indices", [])
                if edge_indices:
                    ei = torch.tensor(edge_indices, dtype=torch.long, device=self.config.device)
                    if ei.dim() == 2 and ei.size(1) == 2:
                        edge_index = ei.t().contiguous()
                    elif ei.dim() == 2 and ei.size(0) == 2:
                        edge_index = ei.contiguous()
                    else:
                        edge_index = torch.tensor([[], []], dtype=torch.long, device=self.config.device)
                else:
                    edge_index = torch.tensor([[], []], dtype=torch.long, device=self.config.device)

                edge_features = graph.get("edge_features", [])
                if edge_features:
                    edge_attr = torch.tensor(
                        [[ef.get("bond_type", 0), ef.get("stereo", 0), float(ef.get("is_conjugated", False))]
                         for ef in edge_features],
                        dtype=torch.float, device=self.config.device,
                    )
                else:
                    edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float, device=self.config.device)

                # reconcile sizes
                num_ei = edge_index.size(1)
                num_ea = edge_attr.size(0)
                if num_ei != num_ea:
                    if num_ei == 0:
                        edge_attr = torch.zeros((0, 3), dtype=torch.float, device=self.config.device)
                    elif num_ea > num_ei:
                        edge_attr = edge_attr[:num_ei].contiguous()
                    else:
                        pad = torch.zeros((num_ei - num_ea, 3), dtype=torch.float, device=self.config.device)
                        edge_attr = torch.cat([edge_attr, pad], dim=0)

                batch["gine"] = {
                    "z": z_tensor,
                    "chirality": chirality_tensor,
                    "formal_charge": formal_charge_tensor,
                    "edge_index": edge_index,
                    "edge_attr": edge_attr,
                    "batch": torch.zeros(z_tensor.size(0), dtype=torch.long, device=self.config.device),
                }

        # geometry
        if "geometry" in multimodal_data:
            geom = multimodal_data["geometry"]
            best_conf = geom.get("best_conformer", {})
            if best_conf:
                atomic_numbers = best_conf.get("atomic_numbers", [])
                coordinates = best_conf.get("coordinates", [])
                if atomic_numbers and coordinates:
                    batch["schnet"] = {
                        "z": torch.tensor(atomic_numbers, dtype=torch.long, device=self.config.device),
                        "pos": torch.tensor(coordinates, dtype=torch.float, device=self.config.device),
                        "batch": torch.zeros(len(atomic_numbers), dtype=torch.long, device=self.config.device),
                    }

        # fingerprints
        if "fingerprints" in multimodal_data:
            fp_dict = multimodal_data["fingerprints"]
            morgan_bits = fp_dict.get("morgan_r3_bits", [])
            if morgan_bits:
                fp_vec = [1 if b else 0 for b in morgan_bits[:2048]]
                if len(fp_vec) < 2048:
                    fp_vec += [0] * (2048 - len(fp_vec))
                batch["fp"] = {
                    "input_ids": torch.tensor(fp_vec, dtype=torch.long, device=self.config.device).unsqueeze(0),
                    "attention_mask": torch.ones(1, 2048, dtype=torch.bool, device=self.config.device),
                }

        # psmiles tokenization for psmiles encoder
        if self._psmiles_tokenizer is None:
            try:
                from cl_models import build_psmiles_tokenizer
                self._psmiles_tokenizer = build_psmiles_tokenizer()
            except Exception:
                self._psmiles_tokenizer = None

        psmiles_str = multimodal_data.get("canonical_psmiles", "")
        if psmiles_str and self._psmiles_tokenizer is not None:
            enc = self._psmiles_tokenizer(psmiles_str, truncation=True, padding="max_length", max_length=128)
            batch["psmiles"] = {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long, device=self.config.device).unsqueeze(0),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long, device=self.config.device).unsqueeze(0),
            }

        return batch

    def _run_cl_encoding(self, step: Dict, data: Dict) -> Dict:
        multimodal_data = data.get("data_extraction", {})
        if not multimodal_data or "canonical_psmiles" not in multimodal_data:
            return {"error": "No multimodal data found. Run data_extraction first."}

        self._ensure_cl_encoder()

        try:
            batch_mods = self._prepare_batch_from_extraction(multimodal_data)
            with torch.no_grad():
                embeddings_dict = self._cl_encoder.encode(batch_mods)

            # enforce that all four modalities are present (gine, schnet, fp, psmiles)
            required_modalities = ("gine", "schnet", "fp", "psmiles")
            missing = [m for m in required_modalities if m not in embeddings_dict]
            if missing:
                return {"error": f"Missing CL embeddings for modalities: {', '.join(missing)}"}

            all_embs = [embeddings_dict[k] for k in required_modalities]
            final_embedding = torch.stack(all_embs, dim=0).mean(dim=0).squeeze(0).contiguous()

            return {
                "embedding": final_embedding.detach().cpu().tolist(),
                "embedding_dim": int(final_embedding.shape[-1]),
                "modalities_used": list(required_modalities),
                "psmiles": multimodal_data["canonical_psmiles"],
            }
        except Exception as e:
            return {"error": f"Failed to encode: {e}"}

    # ----------------- Property heads (downstream) ----------------- #
    def _load_property_head(self, property_name: str):
        import torch.nn as nn

        property_name = canonical_property_name(property_name)
        prop_ckpt = PROPERTY_HEAD_PATHS.get(property_name)
        prop_meta = PROPERTY_HEAD_META.get(property_name)

        if prop_ckpt is None:
            raise ValueError(f"No property head registered for: {property_name}")
        if not os.path.exists(prop_ckpt):
            raise FileNotFoundError(f"Property head checkpoint not found: {prop_ckpt}")

        if property_name in self._property_heads:
            return self._property_heads[property_name]

        meta: Dict[str, Any] = {}
        if prop_meta and os.path.exists(prop_meta):
            try:
                with open(prop_meta, "r") as fh:
                    meta = json.load(fh)
            except Exception:
                meta = {}

        ckpt = torch.load(prop_ckpt, map_location=self.config.device, weights_only=False)

        # locate state dict
        state_dict = None
        for k in ("state_dict", "model_state_dict", "model_state", "head_state_dict", "regressor_state_dict"):
            if isinstance(ckpt, dict) and k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        if state_dict is None:
            raise RuntimeError(f"Could not find a usable state dict in {prop_ckpt}")

        class RegressionHeadOnly(nn.Module):
            def __init__(self, hidden_dim=600, dropout=0.1):
                super().__init__()
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                )

            def forward(self, x):
                return self.head(x).squeeze(-1)

        head = RegressionHeadOnly(hidden_dim=600, dropout=float(meta.get("dropout", 0.1))).to(self.config.device)

        # normalize key prefixes
        normalized = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            if nk.startswith("regressor."):
                nk = nk.replace("regressor.", "head.", 1)

            if nk.startswith("head."):
                normalized[nk] = v
            elif re.match(r"^\d+\.", nk):
                normalized["head." + nk] = v
            else:
                normalized["head." + nk] = v

        head.load_state_dict(normalized, strict=False)
        head.eval()

        # y scaler
        y_scaler = None
        if isinstance(ckpt, dict):
            for sk in ("y_scaler", "scaler_y", "target_scaler", "y_normalizer"):
                if sk in ckpt:
                    y_scaler = ckpt.get(sk)
                    break

        if y_scaler is None and isinstance(meta, dict) and joblib is not None:
            for path_key in ("y_scaler_path", "target_scaler_path", "scaler_path", "y_norm_path"):
                spath = meta.get(path_key)
                if spath and isinstance(spath, str) and os.path.exists(spath):
                    try:
                        y_scaler = joblib.load(spath)
                        break
                    except Exception:
                        y_scaler = None

        self._property_heads[property_name] = (head, y_scaler, meta, prop_ckpt)
        return self._property_heads[property_name]

    def _run_property_prediction(self, step: Dict, data: Dict) -> Dict:
        property_name = data.get("property", data.get("property_name", None))
        if property_name is None:
            return {"error": "Specify property name"}

        property_name = canonical_property_name(property_name)
        if property_name not in PROPERTY_HEAD_PATHS:
            return {"error": f"Unsupported property: {property_name}"}

        # Prefer embedding from cl_encoding output if available
        emb_from_cl = None
        cl = data.get("cl_encoding", None)
        if isinstance(cl, dict) and isinstance(cl.get("embedding"), list) and len(cl["embedding"]) == 600:
            emb_from_cl = torch.tensor([cl["embedding"]], dtype=torch.float32, device=self.config.device)

        # If no embedding provided, compute via extraction + CL
        multimodal = data.get("data_extraction", None)
        psmiles = data.get("psmiles", data.get("smiles", None))
        if emb_from_cl is None:
            if psmiles and not multimodal:
                multimodal = self._run_data_extraction({"step": -1}, {"psmiles": psmiles})
                if "error" in multimodal:
                    return multimodal
                data["data_extraction"] = multimodal
            if not multimodal or "canonical_psmiles" not in multimodal:
                return {"error": "No multimodal data; provide psmiles or data_extraction first."}

            self._ensure_cl_encoder()
            try:
                batch_mods = self._prepare_batch_from_extraction(multimodal)
                with torch.no_grad():
                    embs = self._cl_encoder.encode(batch_mods)

                # enforce all four modalities
                required_modalities = ("gine", "schnet", "fp", "psmiles")
                missing = [m for m in required_modalities if m not in embs]
                if missing:
                    return {"error": f"CL encoder did not return embeddings for modalities: {', '.join(missing)}"}

                all_embs = [embs[k] for k in required_modalities]
                emb_from_cl = torch.stack(all_embs, dim=0).mean(dim=0)  # (B,600)
            except Exception as e:
                return {"error": f"Failed to compute CL embedding: {e}"}

        # Predict
        try:
            head, y_scaler, meta, ckpt_path = self._load_property_head(property_name)
            with torch.no_grad():
                pred_norm = head(emb_from_cl).squeeze(0).item()

            pred_value = float(pred_norm)

            # 1) Preferred: inverse_transform using the actual scaler object if available
            if y_scaler is not None and hasattr(y_scaler, "inverse_transform"):
                try:
                    inv = y_scaler.inverse_transform(np.array([[pred_norm]], dtype=float))
                    pred_value = float(inv[0][0])
                except Exception:
                    pred_value = float(pred_norm)

            # 2) Fallback: use metadata params if scaler object is missing
            else:
                mean = (meta or {}).get("scaler_mean", None)
                scale = (meta or {}).get("scaler_scale", None)

                # StandardScaler inverse: x = x_scaled * scale + mean
                try:
                    if isinstance(mean, list) and isinstance(scale, list) and len(mean) == 1 and len(scale) == 1:
                        pred_value = float(pred_norm) * float(scale[0]) + float(mean[0])
                except Exception:
                    pred_value = float(pred_norm)

            # best-effort psmiles context
            out_psmiles = None
            if isinstance(multimodal, dict):
                out_psmiles = multimodal.get("canonical_psmiles")
            if out_psmiles is None and isinstance(cl, dict):
                out_psmiles = cl.get("psmiles")
            if out_psmiles is None:
                out_psmiles = psmiles

            return {
                "psmiles": out_psmiles,
                "property": property_name,
                "predictions": {property_name: pred_value},
                "prediction_normalized": float(pred_norm),
                "head_checkpoint_path": ckpt_path,
                "metadata_path": PROPERTY_HEAD_META.get(property_name, ""),
                "normalization_applied": bool(
                    (y_scaler is not None and hasattr(y_scaler, "inverse_transform")) or
                    ((meta or {}).get("scaler_mean") is not None and (meta or {}).get("scaler_scale") is not None)
                ),
                "used_cl_embedding": True,
            }
        except Exception as e:
            return {"error": f"Property prediction failed: {e}"}

    # ----------------- Inverse-design generator (NEW: CL + SELFIES-TED, as in G2) ----------------- #
    def _get_selfies_ted_backend(self, model_name: str) -> Tuple[Any, Any]:
        """
        Cache and return (tokenizer, model) for a given SELFIES-TED model name.
        """
        if not model_name:
            model_name = SELFIES_TED_MODEL_NAME
        if model_name in self._selfies_ted_cache:
            return self._selfies_ted_cache[model_name]
        tok, model = load_selfies_ted_and_tokenizer(model_name)
        model.to(self.config.device)
        self._selfies_ted_cache[model_name] = (tok, model)
        return tok, model

    def _load_property_generator(self, property_name: str):
        """
        Load PolyBART-style inverse-design artifacts produced by G2.py:
          - decoder_best_fold*.pt : state_dict of CLConditionedSelfiesTEDGenerator
          - standardscaler_*.joblib : StandardScaler on property values
          - gpr_psmiles_*.joblib : LatentPropertyModel (z->property)
          - meta.json : meta info (selfies_ted_model, cl_emb_dim, mem_len, tol_scaled, ...)
        """
        property_name = canonical_property_name(property_name)
        if property_name in self._property_generators:
            return self._property_generators[property_name]

        base_dir = GENERATOR_DIRS.get(property_name)
        if base_dir is None:
            raise ValueError(f"No generator registered for: {property_name}")
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Generator directory not found: {base_dir}")

        meta_path = os.path.join(base_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found in {base_dir}")

        if joblib is None:
            raise RuntimeError("joblib not installed but required to load *.joblib artifacts (pip install joblib).")

        with open(meta_path, "r") as fh:
            meta = json.load(fh) if fh else {}

        best_fold = None
        for k in ("best_fold", "selected_fold", "fold", "bestFold", "best_fold_idx"):
            if k in meta:
                try:
                    best_fold = int(meta[k])
                    break
                except Exception:
                    best_fold = None
        if best_fold is None:
            best_fold = 1

        decoder_path = os.path.join(base_dir, f"decoder_best_fold{best_fold}.pt")
        if not os.path.exists(decoder_path):
            decs = sorted([p for p in os.listdir(base_dir) if p.startswith("decoder_best_fold") and p.endswith(".pt")])
            if not decs:
                raise FileNotFoundError(f"No decoder_best_fold*.pt found in {base_dir}")
            decoder_path = os.path.join(base_dir, decs[0])

        scaler_path = None
        gpr_path = None
        for fn in os.listdir(base_dir):
            low = fn.lower()
            if low.startswith("standardscaler_") and low.endswith(".joblib"):
                scaler_path = os.path.join(base_dir, fn)
            if low.startswith("gpr_psmiles_") and low.endswith(".joblib"):
                gpr_path = os.path.join(base_dir, fn)

        if not scaler_path or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"StandardScaler *.joblib not found in {base_dir}")
        if not gpr_path or not os.path.exists(gpr_path):
            raise FileNotFoundError(f"GPR *.joblib not found in {base_dir}")

        # Latent property model and scaler (G2-style LatentPropertyModel)
        _install_unpickle_shims()
        scaler_y = _safe_joblib_load(scaler_path)   # StandardScaler on property
        latent_prop_model = _safe_joblib_load(gpr_path)  # should be LatentPropertyModel dataclass-like

        # SELFIES-TED backbone
        selfies_ted_name = meta.get("selfies_ted_model", SELFIES_TED_MODEL_NAME)
        tok, selfies_backbone = self._get_selfies_ted_backend(selfies_ted_name)

        cl_emb_dim = int(meta.get("cl_emb_dim", 600))
        mem_len = int(meta.get("mem_len", 4))

        decoder_model = CLConditionedSelfiesTEDGenerator(
            tok=tok,
            seq2seq_model=selfies_backbone,
            cl_emb_dim=cl_emb_dim,
            mem_len=mem_len,
        ).to(self.config.device)

        ckpt = torch.load(decoder_path, map_location=self.config.device, weights_only=False)
        # In G2, decoder_best_fold*.pt is a plain state_dict; keep robust fallback
        state_dict = None
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        elif isinstance(ckpt, dict):
            for k in ("state_dict", "model_state_dict", "decoder_state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    state_dict = ckpt[k]
                    break
        if state_dict is None:
            raise RuntimeError(f"Could not find a usable state dict in decoder checkpoint: {decoder_path}")

        decoder_model.load_state_dict(state_dict, strict=False)
        decoder_model.eval()

        paths = {
            "base_dir": base_dir,
            "meta_json": meta_path,
            "decoder_checkpoint": decoder_path,
            "scaler_joblib": scaler_path,
            "gpr_joblib": gpr_path,
            "selfies_ted_model": selfies_ted_name,
        }

        self._property_generators[property_name] = (decoder_model, latent_prop_model, scaler_y, tok, meta, paths)
        return self._property_generators[property_name]

    @torch.no_grad()
    def _sample_latents_for_target(
        self,
        latent_prop_model: Any,
        target_value: float,
        num_samples: int,
        latent_dim: int,
        tol_scaled: float,
        y_scaler: Optional[Any] = None,
        seed_latents: Optional[List[np.ndarray]] = None,
        latent_noise_std: float = LATENT_NOISE_STD_GEN,
        extra_factor: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Simple PolyBART-style latent sampler:
          - if seed_latents provided, sample Gaussian noise around them and L2-normalize
          - else, sample random latents on unit hypersphere
          - score via latent_prop_model (z->property), keep those near target.
        """
        def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
            n = np.linalg.norm(x, axis=-1, keepdims=True)
            return x / np.clip(n, eps, None)

        # target in scaled space
        if y_scaler is not None and hasattr(y_scaler, "transform"):
            target_s = float(y_scaler.transform(np.array([[target_value]], dtype=np.float32))[0, 0])
        else:
            target_s = float(target_value)

        n_candidates = max(num_samples * extra_factor, num_samples * 4, 64)
        latents: List[np.ndarray] = []

        if seed_latents:
            seeds = [np.asarray(z, dtype=np.float32).reshape(-1) for z in seed_latents]
            for z0 in seeds:
                z0 = _l2_normalize_np(z0.reshape(1, -1)).reshape(-1)
                latents.append(z0)
                per_seed = max(1, n_candidates // max(1, len(seeds)) - 1)
                for _ in range(per_seed):
                    noise = np.random.randn(latent_dim).astype(np.float32) * float(latent_noise_std)
                    z = z0 + noise
                    z = _l2_normalize_np(z.reshape(1, -1)).reshape(-1)
                    latents.append(z)
        else:
            for _ in range(n_candidates):
                z = np.random.randn(latent_dim).astype(np.float32)
                z = _l2_normalize_np(z.reshape(1, -1)).reshape(-1)
                latents.append(z)

        Z = np.stack(latents, axis=0).astype(np.float32)
        y_s, y_u = _predict_latent_property(latent_prop_model, Z)
        errors = np.abs(y_s - target_s)

        idx_sorted = np.argsort(errors)
        kept = [i for i in idx_sorted if errors[i] <= float(tol_scaled)]
        if len(kept) < num_samples:
            kept = list(idx_sorted[:num_samples])
        kept = kept[:num_samples]

        return Z[kept], y_s[kept], y_u[kept], target_s

    @torch.no_grad()
    def _run_polymer_generation(self, step: Dict, data: Dict) -> Dict:
        """
        Inverse-design generation (CL latent → pSELFIES via SELFIES-TED → PSMILES).

        Corrections implemented:
          1) ONLY return RDKit-valid generated outputs (filter invalid candidates).
          2) Replace bracketed [At]/[AT]/[aT]/... with [*] AFTER the RDKit validity check
             but BEFORE writing the response payload.
        """
        property_name = data.get("property", data.get("property_name", None))
        if property_name is None:
            return {"error": "Specify property name for generation"}

        property_name = canonical_property_name(property_name)
        if property_name not in GENERATOR_DIRS:
            return {"error": f"Unsupported property: {property_name}"}

        # STRICT: require target_value (support a few common aliases)
        if data.get("target_value", None) is not None:
            target_value = data["target_value"]
        elif data.get("target", None) is not None:
            target_value = data["target"]
        elif data.get("target_property_value", None) is not None:
            target_value = data["target_property_value"]
        else:
            return {"error": "Generation requires target_value (inverse design: property -> PSMILES)."}

        try:
            target_value = float(target_value)
        except Exception:
            return {"error": f"target_value must be numeric, got: {target_value!r}"}

        num_samples = int(data.get("num_samples", 4))
        if num_samples < 1:
            num_samples = 1

        top_p = float(data.get("top_p", GEN_TOP_P))
        temperature = float(data.get("temperature", GEN_TEMPERATURE))
        rep_pen = float(data.get("repetition_penalty", GEN_REPETITION_PENALTY))
        max_len = int(data.get("max_len", GEN_MAX_LEN))
        latent_noise_std = float(data.get("latent_noise_std", LATENT_NOISE_STD_GEN))
        extra_factor = int(data.get("extra_factor", 8))
        tol_scaled_override = data.get("tol_scaled", None)

        try:
            decoder_model, latent_prop_model, scaler_y, selfies_tok, meta, paths = self._load_property_generator(property_name)
        except Exception as e:
            return {"error": f"Failed to load inverse-design generator bundle: {e}"}

        latent_dim = int(getattr(decoder_model, "cl_emb_dim", 600))

        # choose target scaler: prefer latent_prop_model.y_scaler, fall back to scaler_y
        y_scaler = getattr(latent_prop_model, "y_scaler", None)
        if y_scaler is None:
            y_scaler = scaler_y if scaler_y is not None else None

        tol_scaled = float(tol_scaled_override) if tol_scaled_override is not None else float(meta.get("tol_scaled", 0.5))

        # Collect seed latents from available sources:
        seed_latents: List[np.ndarray] = []
        cl_enc = data.get("cl_encoding", None)
        if isinstance(cl_enc, dict) and isinstance(cl_enc.get("embedding"), list):
            emb = np.asarray(cl_enc["embedding"], dtype=np.float32)
            if emb.shape[0] == latent_dim:
                seed_latents.append(emb)

        # Optional seed pSMILES strings for biasing
        seeds_str: List[str] = []
        if isinstance(data.get("seed_psmiles_list"), list):
            seeds_str.extend([str(x) for x in data["seed_psmiles_list"] if isinstance(x, str)])
        if data.get("seed_psmiles"):
            seeds_str.append(str(data["seed_psmiles"]))
        if data.get("psmiles") and not seeds_str:
            seeds_str.append(str(data["psmiles"]))

        seeds_str = list(dict.fromkeys(seeds_str))

        # If seed strings provided but no seed latents yet, compute CL embeddings for each seed
        if seeds_str and not seed_latents:
            self._ensure_cl_encoder()
            for s in seeds_str:
                ex = self._run_data_extraction({}, {"psmiles": s})
                if isinstance(ex, dict) and "error" in ex:
                    continue
                cl = self._run_cl_encoding({}, {"data_extraction": ex})
                if isinstance(cl, dict) and isinstance(cl.get("embedding"), list):
                    z = np.asarray(cl["embedding"], dtype=np.float32)
                    if z.shape[0] == latent_dim:
                        seed_latents.append(z)

        # Sample latents targeting the property
        try:
            Z_keep, y_s_keep, y_u_keep, target_s = self._sample_latents_for_target(
                latent_prop_model=latent_prop_model,
                target_value=target_value,
                num_samples=num_samples,
                latent_dim=latent_dim,
                tol_scaled=tol_scaled,
                y_scaler=y_scaler,
                seed_latents=seed_latents if seed_latents else None,
                latent_noise_std=latent_noise_std,
                extra_factor=extra_factor,
            )
        except Exception as e:
            return {"error": f"Failed to sample latents conditioned on property: {e}", "paths": paths}

        # --- helpers ---
        at_bracket_re = re.compile(r"\[(at)\]", flags=re.IGNORECASE)

        def _at_to_star_bracket(s: str) -> str:
            if not isinstance(s, str) or not s:
                return s
            return at_bracket_re.sub("[*]", s)

        def _is_rdkit_valid(psmiles: str) -> bool:
            # If RDKit is unavailable, we cannot validate; treat as "valid" but flag it below.
            if Chem is None:
                return True
            try:
                probe = psmiles_to_rdkit_smiles(psmiles)
                m = Chem.MolFromSmiles(probe)
                return m is not None
            except Exception:
                return False

        # Decode latents → pSELFIES → PSMILES; filter to RDKit-valid ONLY.
        # Shortening strategy (3rd approach): generate MORE valid candidates, then keep the shortest valid K.
        requested_k = int(num_samples)

        # candidates are tuples:
        #   (len(psmiles), abs(y_s - target_s), psmiles_out, selfies_str, y_s, y_u)
        candidates: List[Tuple[int, float, str, str, float, float]] = []

        # Reuse existing knob (extra_factor) to control "generate more" without adding new API surface.
        candidates_per_latent = max(1, int(extra_factor))
        max_gen_rounds = 4  # best-effort retries to satisfy requested_k under RDKit validity filtering

        Z_round, y_s_round, y_u_round = Z_keep, y_s_keep, y_u_keep
        for _round in range(max_gen_rounds):
            for i in range(Z_round.shape[0]):
                z_vec = torch.tensor(Z_round[i], dtype=torch.float32, device=self.config.device).unsqueeze(0)
                try:
                    outs = decoder_model.generate(
                        z=z_vec,
                        num_return_sequences=candidates_per_latent,
                        max_len=max_len,
                        top_p=top_p,
                        temperature=temperature,
                        repetition_penalty=rep_pen,
                    )
                    for selfies_str in (outs or []):
                        psm_raw = pselfies_to_psmiles(selfies_str)

                        # Correction #1: validate FIRST on the raw returned string
                        if _is_rdkit_valid(psm_raw):
                            # Correction #2: convert [At] -> [*] AFTER validation, BEFORE response writing
                            psm_out = _at_to_star_bracket(psm_raw)
                            candidates.append(
                                (
                                    len(psm_out) if isinstance(psm_out, str) else 0,
                                    abs(float(y_s_round[i]) - float(target_s)),
                                    psm_out,
                                    selfies_str,
                                    float(y_s_round[i]),
                                    float(y_u_round[i]),
                                )
                            )
                except Exception:
                    continue

            # Stop early once we have enough valid candidates to select the shortest K.
            if len(candidates) >= requested_k:
                break

            # If still short, resample latents and try again (best-effort; keeps validity constraints).
            try:
                Z_round, y_s_round, y_u_round, target_s = self._sample_latents_for_target(
                    latent_prop_model=latent_prop_model,
                    target_value=target_value,
                    num_samples=requested_k,
                    latent_dim=latent_dim,
                    tol_scaled=tol_scaled,
                    y_scaler=y_scaler,
                    seed_latents=seed_latents if seed_latents else None,
                    latent_noise_std=latent_noise_std,
                    extra_factor=extra_factor,
                )
            except Exception:
                break

        # Keep shortest valid K (tie-break by closeness to target in scaled space)
        candidates.sort(key=lambda t: (t[0], t[1]))
        selected = candidates[:requested_k]

        # Ensure we return as many as requested when possible (repeat shortest valid if needed).
        if selected and len(selected) < requested_k:
            while len(selected) < requested_k:
                selected.append(selected[0])

        generated_psmiles: List[str] = [t[2] for t in selected]
        selfies_raw: List[str] = [t[3] for t in selected]
        decoded_scaled: List[float] = [t[4] for t in selected]
        decoded_unscaled: List[float] = [t[5] for t in selected]

        return {
            "property": property_name,
            "target_value": float(target_value),
            "num_samples": int(len(generated_psmiles)),
            "generated_psmiles": generated_psmiles,   # RDKit-valid ONLY; [At]->[*] applied after validation
            "generated_selfies": selfies_raw,         # aligned with generated_psmiles
            "latent_property_predictions": {
                "scaled": decoded_scaled,
                "unscaled": decoded_unscaled,
                "target_scaled": float(target_s),
                "tol_scaled": float(tol_scaled),
            },
            "inverse_design_paths": paths,
            "selfies_ted_model": meta.get("selfies_ted_model", SELFIES_TED_MODEL_NAME),
            "latent_dim": int(latent_dim),
            "used_seed_latents": bool(seed_latents),
            "seed_psmiles_used": seeds_str,
            "rdkit_validation": {
                "enabled": bool(Chem is not None),
                "note": "Only RDKit-valid generated candidates are returned when RDKit is available."
                        if Chem is not None else
                        "RDKit not available; validity filtering could not be applied.",
            },
            "sampler": {
                "MAX_LENGTH": max_len,
                "TOP_P": top_p,
                "TEMPERATURE": temperature,
                "REPETITION_PENALTY": rep_pen,
                "LATENT_NOISE_STD": latent_noise_std,
                "EXTRA_FACTOR": extra_factor,
            },
        }

    # ----------------- Web tools ----------------- #
    def _crossref_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        url = "https://api.crossref.org/works"
        params = {
            "query.bibliographic": query,
            "rows": rows,
            "filter": "type:journal-article,from-pub-date:2015-01-01",
        }
        try:
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            out = []
            for it in items:
                cr_type = (it.get("type") or "").lower()
                if cr_type and cr_type != "journal-article":
                    continue
                title = " ".join(it.get("title", [])) if it.get("title") else ""
                doi = normalize_doi(it.get("DOI", "")) or ""

                publisher = (it.get("publisher") or "").lower()
                # Optional: exclude Brill explicitly
                if doi and doi.startswith("10.1163/"):
                    continue
                if "brill" in publisher:
                    continue
                pub_year = None
                if it.get("published-print") and isinstance(it["published-print"].get("date-parts"), list):
                    pub_year = it["published-print"]["date-parts"][0][0]
                elif it.get("created"):
                    pub_year = it["created"].get("date-parts", [[None]])[0][0]

                doi_url = doi_to_url(doi) if doi else ""
                if doi_url and not doi_resolves(doi_url):
                    doi = ""
                    doi_url = ""

                # Prefer DOI URL when valid; otherwise fall back to Crossref's URL field if present.
                landing = (it.get("URL") or "") if isinstance(it.get("URL"), str) else ""
                out.append({
                    "title": title,
                    "doi": doi,
                    "url": doi_url or landing or "",
                    "year": pub_year,
                    "source": "CrossRef",
                    "type": cr_type,
                    "publisher": it.get("publisher", ""),
                })
            return out
        except Exception as e:
            return [{"error": f"CrossRef query failed: {e}"}]

    def _openalex_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        try:
            url = "https://api.openalex.org/works"
            params = {"search": query, "per-page": rows}
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            items = r.json().get("results", [])
            out = []
            for it in items:
                oa_type = (it.get("type") or "").lower()
                if oa_type and oa_type not in {"journal-article", "proceedings-article", "posted-content"}:
                    continue

                doi = normalize_doi(it.get("doi", "")) or ""
                # Optional: exclude Brill explicitly
                if doi and doi.startswith("10.1163/"):
                    continue

                pl = (it.get("primary_location") or {})
                landing = (
                    pl.get("landing_page_url")
                    or ((pl.get("source") or {}).get("homepage_url"))
                    or ""
                )
                doi_url = doi_to_url(doi) if doi else ""
                if doi_url and not doi_resolves(doi_url):
                    doi = ""
                    doi_url = ""

                out.append({
                    "title": it.get("title", ""),
                    "doi": doi,                  # normalized, not a URL
                    "url": landing or "",         # prefer landing page URL
                    "year": it.get("publication_year") or (it.get("publication_date", "")[:4]),
                    "venue": (it.get("host_venue") or {}).get("display_name", ""),
                    "type": oa_type,                    "source": "OpenAlex",
                })
            return out
        except Exception as e:
            return [{"error": f"OpenAlex query failed: {e}"}]

    def _epmc_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        try:
            base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {"query": query, "format": "json", "pageSize": rows}
            r = requests.get(base, params=params, timeout=12)
            r.raise_for_status()
            hits = r.json().get("resultList", {}).get("result", [])
            out = []
            for it in hits:
                out.append({
                    "title": it.get("title", ""),
                    "pmcid": it.get("pmcid", ""),
                    "year": it.get("pubYear", ""),
                    "abstract": it.get("abstractText", ""),
                    "source": "EuropePMC",
                })
            return out
        except Exception as e:
            return [{"error": f"Europe PMC query failed: {e}"}]

    def _arxiv_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        if BeautifulSoup is None:
            return [{"error": "bs4 not installed for arXiv parse"}]
        try:
            url = "http://export.arxiv.org/api/query"
            params = {"search_query": f"all:{query}", "start": 0, "max_results": rows}
            r = requests.get(url, params=params, timeout=12, headers={"User-Agent": "PolyOrch/1.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            out = []
            for entry in soup.find_all("entry"):
                title = (entry.title.text or "").strip()
                year = (entry.published.text or "")[:4] if entry.published else ""
                link = ""
                link_tag = entry.find("link", {"type": "text/html"})
                if link_tag and link_tag.get("href"):
                    link = link_tag["href"]
                elif entry.id:
                    link = entry.id.text
                out.append({"title": title, "url": link, "year": year, "source": "arXiv"})
            return out
        except Exception as e:
            return [{"error": f"arXiv query failed: {e}"}]

    def _semantic_scholar_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {"query": query, "limit": rows, "fields": "title,year,externalIds,url,venue,abstract"}
            headers = {}
            if self.config.semantic_scholar_key:
                headers["x-api-key"] = self.config.semantic_scholar_key
            r = requests.get(url, params=params, timeout=12, headers=headers)
            r.raise_for_status()
            papers = r.json().get("data", [])
            out = []
            for p in papers:
                doi = normalize_doi((p.get("externalIds") or {}).get("DOI", "")) or ""
                if doi and doi.startswith("10.1163/"):
                    continue
                doi_url = doi_to_url(doi) if doi else ""
                if doi_url and not doi_resolves(doi_url):
                    doi = ""
                out.append({
                    "title": p.get("title", ""),
                    "doi": doi,
                    "url": p.get("url", "") or "",
                    "year": p.get("year", ""),
                    "venue": p.get("venue", ""),
                    "abstract": p.get("abstract", ""),
                    "source": "SemanticScholar",
                })
            return out
        except Exception as e:
            return [{"error": f"Semantic Scholar query failed: {e}"}]

    def _springer_nature_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        if not self.config.springer_api_key:
            return [{"warning": "SPRINGER_NATURE_API_KEY not set; skipping Springer Nature"}]
        try:
            url = "https://api.springernature.com/metadata/json"
            params = {"q": query, "api_key": self.config.springer_api_key, "p": rows}
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            recs = r.json().get("records", [])
            out = []
            for rec in recs:
                title = rec.get("title", "")
                year = (rec.get("publicationDate", "") or "")[:4]
                urlp = ""
                if rec.get("url"):
                    urlp = rec["url"][0].get("value", "")
                out.append({"title": title, "doi": rec.get("doi", ""), "url": urlp, "year": year, "source": "SpringerNature"})
            return out
        except Exception as e:
            return [{"error": f"Springer Nature query failed: {e}"}]

    def _internet_archive_search(self, query: str, rows: int = 6) -> List[Dict[str, Any]]:
        if requests is None:
            return [{"error": "requests not installed"}]
        try:
            url = "https://archive.org/advancedsearch.php"
            params = {"q": query, "fl[]": "identifier,title,year,creator", "rows": rows, "output": "json"}
            r = requests.get(url, params=params, timeout=12)
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])
            out = []
            for d in docs:
                ident = d.get("identifier", "")
                out.append({
                    "title": d.get("title", ""),
                    "url": f"https://archive.org/details/{ident}" if ident else "",
                    "year": d.get("year", ""),
                    "source": "InternetArchive",
                })
            return out
        except Exception as e:
            return [{"error": f"Internet Archive query failed: {e}"}]

    def _fetch_page(self, url: str, max_chars: int = 1200) -> Dict[str, Any]:
        if requests is None or BeautifulSoup is None:
            return {"error": "requests or bs4 not available"}
        try:
            r = requests.get(url, timeout=12, headers={"User-Agent": "PolyOrch/1.0"})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""
            paras = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
            excerpt = ""
            for p in paras:
                if len(p) > 50:
                    excerpt = p
                    break
            if not excerpt:
                excerpt = soup.get_text(separator=" ", strip=True)[:max_chars]
            return {"title": title, "excerpt": excerpt[:max_chars], "url": url}
        except Exception as e:
            return {"error": f"Fetch failed: {e}", "url": url}

    def _run_web_search(self, step: Dict, data: Dict) -> Dict:
        src = (data.get("source", data.get("src", "crossref")) or "").lower()
        query = data.get("query", data.get("q", "")) or ""
        rows = int(data.get("rows", 6))

        if src in ("crossref", "openalex", "epmc", "arxiv", "semanticscholar", "springer", "internetarchive", "all") and not query:
            return {"error": f"No query provided for {src} search"}

        if src == "crossref":
            return {"source": "crossref", "query": query, "results": self._crossref_search(query, rows)}
        if src == "openalex":
            return {"source": "openalex", "query": query, "results": self._openalex_search(query, rows)}
        if src == "epmc":
            return {"source": "epmc", "query": query, "results": self._epmc_search(query, rows)}
        if src == "arxiv":
            return {"source": "arxiv", "query": query, "results": self._arxiv_search(query, rows)}
        if src == "semanticscholar":
            return {"source": "semanticscholar", "query": query, "results": self._semantic_scholar_search(query, rows)}
        if src == "springer":
            return {"source": "springer", "query": query, "results": self._springer_nature_search(query, rows)}
        if src == "internetarchive":
            return {"source": "internetarchive", "query": query, "results": self._internet_archive_search(query, rows)}
        if src == "fetch":
            url = data.get("url", "")
            if not url:
                return {"error": "No URL provided for fetch"}
            return {"source": "fetch", "url": url, "page": self._fetch_page(url)}
        if src == "all":
            aggregated = {
                "crossref": self._crossref_search(query, rows),
                "openalex": self._openalex_search(query, rows),
                "epmc": self._epmc_search(query, rows),
                "arxiv": self._arxiv_search(query, rows),
                "semanticscholar": self._semantic_scholar_search(query, rows),
                "springer": self._springer_nature_search(query, rows),
                "internetarchive": self._internet_archive_search(query, rows),
            }
            return {"source": "all", "query": query, "results": aggregated}

        return {"error": f"Unsupported web_search source: {src}"}

    # =============================================================================
    # REPORT GENERATION (FIX for Gradio interface expectations)
    # =============================================================================
    def generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal, interface-safe report generator used by the Gradio UI fallback.

        - Runs data_extraction -> cl_encoding -> property_prediction when possible
        - Optionally runs polymer_generation if generate=True or if target_value present
        - Optionally runs web_search if a query/literature_query is present
        """
        payload = dict(data or {})
        summary: Dict[str, Any] = {}

        # Seed psmiles/property
        prop = payload.get("property") or payload.get("property_name")
        if prop:
            payload["property"] = prop

        # NEW: infer property from questions when missing
        if not payload.get("property"):
            qtxt = payload.get("questions") or payload.get("question") or ""
            inferred_prop = infer_property_from_text(qtxt)
            if inferred_prop:
                payload["property"] = inferred_prop

        psmiles = payload.get("psmiles") or payload.get("seed_psmiles")
        if psmiles:
            payload["psmiles"] = psmiles

        # NEW: infer target_value from questions when missing (only useful for generation)
        if payload.get("target_value", None) is None:
            qtxt = payload.get("questions") or payload.get("question") or ""
            inferred_tgt = infer_target_value_from_text(qtxt, payload.get("property"))
            if inferred_tgt is not None:
                payload["target_value"] = float(inferred_tgt)

        # 1) data_extraction
        if psmiles and "data_extraction" not in payload:
            ex = self._run_data_extraction({"step": -1}, payload)
            payload["data_extraction"] = ex
            summary["data_extraction"] = ex

        # 2) cl_encoding
        if "data_extraction" in payload and "cl_encoding" not in payload:
            cl = self._run_cl_encoding({"step": -1}, payload)
            payload["cl_encoding"] = cl
            summary["cl_encoding"] = cl

        # 3) property_prediction
        if payload.get("property") and "property_prediction" not in payload:
            pp = self._run_property_prediction({"step": -1}, payload)
            payload["property_prediction"] = pp
            summary["property_prediction"] = pp

        # 4) polymer_generation (optional)
        do_gen = bool(payload.get("generate", False)) or (payload.get("target_value", None) is not None)
        if do_gen and payload.get("property") and payload.get("target_value", None) is not None:
            gen = self._run_polymer_generation({"step": -1}, payload)
            payload["polymer_generation"] = gen
            summary["generation"] = gen

        # 5) web_search (optional)
        q = payload.get("query") or payload.get("literature_query")
        src = payload.get("source") or "all"
        if q:
            ws = self._run_web_search({"step": -1}, {"source": src, "query": q, "rows": int(payload.get("rows", 6))})
            payload["web_search"] = ws
            summary["web_search"] = ws

        report = {
            "summary": summary,
            "tool_outputs": {
                "data_extraction": payload.get("data_extraction"),
                "cl_encoding": payload.get("cl_encoding"),
                "property_prediction": payload.get("property_prediction"),
                "polymer_generation": payload.get("polymer_generation"),
                "web_search": payload.get("web_search"),
                "rag_retrieval": payload.get("rag_retrieval"),
            },
            "questions": payload.get("questions") or payload.get("question") or "",
        }

        # Add domain tags + (domain.com) cite tags, and tool tags [T#]
        report = _attach_source_domains(report)
        report = _index_citable_sources(report)
        report = _assign_tool_tags_to_report(report)

        return report

    def _run_report_generation(self, step: Dict, data: Dict) -> Dict[str, Any]:
        return self.generate_report(data)

    def compose_gpt_style_answer(
        self,
        report: Dict[str, Any],
        case_brief: str = "",
        questions: str = "",
    ) -> Tuple[str, List[str]]:
        """
        Interface-safe composer. Uses OpenAI if available; otherwise returns a deterministic markdown.
        Must return: (final_markdown, list_of_image_paths).

        Updated requirements:
          - No fixed answer template: structure must follow the user's actual questions.
          - Literature/web citations must be domain-style like "nature.com" (never [1], [2], ...). No parentheses.
          - Tool-derived facts must cite as [T] only.
          - Tool outputs should be available verbatim without tweaking (appended as JSON blocks).
        """
        imgs: List[str] = []

        # Ensure tags exist even if caller didn't run generate_report()
        if isinstance(report, dict):
            report = _attach_source_domains(report)
            report = _index_citable_sources(report)
            report = _assign_tool_tags_to_report(report)

        if self.openai_client is None:
            # Deterministic fallback (no API dependency)
            md_lines = []
            if case_brief:
                md_lines.append(case_brief.strip())
                md_lines.append("")
            if questions:
                md_lines.append(questions.strip())
                md_lines.append("")
            md_lines.append("```json")
            try:
                md_lines.append(json.dumps(report, indent=2, ensure_ascii=False))
            except Exception:
                md_lines.append(str(report))
            md_lines.append("```")

            # Verbatim tool outputs (no tweaking)
            verb = _render_tool_outputs_verbatim_md(report) if isinstance(report, dict) else ""
            if verb:
                md_lines.append("\n---\n\n## Tool outputs (verbatim)\n")
                md_lines.append(verb)

            return "\n".join(md_lines), imgs

        # OpenAI-based synthesis
        try:
            prompt = (
                "You are PolyAgent - consider yourself as an expert in polymer science. Answer the user's questions using ONLY the provided report.\n"
                "Do NOT follow a fixed template. Let the structure be driven by the user's questions.\n\n"
                "CITATION RULES (STRICT):\n"
                "- Tool facts: when you use any information from a tool output, cite it as [T] (exactly; no numbering).\n"
                "- Literature/web facts: cite using the COMPLETE DOI URL (https://doi.org/...) in brackets as a Markdown hyperlink.\n"
                "  The bracket text MUST be the full DOI URL (or the best URL if DOI is unavailable), and the href MUST be that same URL.\n"
                "- NEVER use numbered citations like [1], [2] for papers.\n"
                "- Every literature/web/RAG citation MUST be an inline Markdown hyperlink placed immediately after the claim.\n"
                "- You are FORBIDDEN from adding any 'References', 'Sources', 'Bibliography', or 'Works Cited' section.\n"
                "- Distribute citations across the answer (do not cluster them in one place).\n"
                "- NON-DUPLICATES: Do not repeat the same paper link. Each DOI/URL may appear at most once in the entire answer.\n"
                "- Each major section should include at least 1 inline literature citation when relevant.\n"
                "- Do NOT invent DOIs, URLs, titles, or sources.\n\n"
                "- CITATIONS AS SPECIFIED ONLY: very strictly place each citation immediately after the claim it supports; do not add a references list.\n"
                "OUTPUT RULES (STRICT):\n"
                "- If a numeric value is not present in the report, write 'not available'.\n"
                "- Preserve polymer endpoint tokens exactly as '[*]' in any pSMILES/SMILES shown.\n"
                "- To prevent markdown mangling, put any pSMILES/SMILES inside code formatting.\n"
                "- Do not rewrite or tweak any tool outputs; if you refer to them, reference them by tag (e.g., [T2]).\n\n"
                f"CASE BRIEF:\n{case_brief}\n\n"
                f"QUESTIONS:\n{questions}\n\n"
                f"REPORT (JSON):\n{json.dumps(report, ensure_ascii=False)}\n"
            )
            resp = self.openai_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "Return a single markdown answer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2200,
            )
            txt = resp.choices[0].message.content or ""

            # Enforce distributed inline clickable paper citations (do not touch tool citations).
            # This corrects cases where the model under-cites or clusters citations.
            try:
                min_cites = _infer_required_citation_count(questions or "", default_n=10)
                txt = _ensure_distributed_inline_citations(txt, report, min_needed=min_cites)
            except Exception:
                pass


            # Enforce: DOI-URL bracket text + dedupe (each DOI/URL appears at most once)
            try:
                txt = _normalize_and_dedupe_literature_links(txt, report)
            except Exception:
                pass

            try:
                txt = autolink_doi_urls(txt)
            except Exception:
                pass

            # Always append verbatim tool outputs (no tweaking)
            verb = _render_tool_outputs_verbatim_md(report) if isinstance(report, dict) else ""
            if verb:
                txt = txt.rstrip() + "\n\n---\n\n## Tool outputs (verbatim)\n\n" + verb

            return txt, imgs
        except Exception as e:
            # Last-resort fallback
            md = f"OpenAI compose failed: {e}\n\n```json\n{json.dumps(report, indent=2, ensure_ascii=False)}\n```"
            # Still append verbatim tool outputs
            verb = _render_tool_outputs_verbatim_md(report) if isinstance(report, dict) else ""
            if verb:
                md = md.rstrip() + "\n\n---\n\n## Tool outputs (verbatim)\n\n" + verb
            return md, imgs

    # =============================================================================
    # VISUAL TOOLS (PNG-only)
    # =============================================================================
    def _run_mol_render(self, step: Dict, data: Dict) -> Dict[str, Any]:
        out_dir = Path("viz")
        out_dir.mkdir(parents=True, exist_ok=True)

        if Chem is None or Draw is None:
            return {"error": "RDKit not installed"}

        p = data.get("psmiles") or data.get("seed_psmiles")
        if not p:
            return {"error": "no psmiles"}

        mol = Chem.MolFromSmiles(psmiles_to_rdkit_smiles(p))
        if mol is None:
            return {"error": "invalid psmiles"}

        img = Draw.MolToImage(mol, size=(600, 400))
        png = str(out_dir / "mol.png")
        img.save(png)
        return {"png_path": png, "legend": p}

    def _run_gen_grid(self, step: Dict, data: Dict) -> Dict[str, Any]:
        out_dir = Path("viz")
        out_dir.mkdir(parents=True, exist_ok=True)

        if Chem is None or Draw is None:
            return {"error": "RDKit not installed"}

        p_list = data.get("psmiles_list")
        if p_list is None:
            gen = data.get("polymer_generation") or {}
            p_list = gen.get("generated_psmiles", [])
        if not p_list:
            return {"error": "no psmiles_list provided and no generated_psmiles found"}

        mols = []
        legends = []
        for i, p in enumerate(p_list, 1):
            m = Chem.MolFromSmiles(psmiles_to_rdkit_smiles(p)) if p else None
            if m is None:
                continue
            mols.append(m)
            legends.append(f"{i}")

        if not mols:
            return {"error": "no valid molecules to render"}

        img = Draw.MolsToGridImage(mols, molsPerRow=min(4, len(mols)), subImgSize=(300, 220), legends=legends, useSVG=False)
        png = str(out_dir / "gen_grid.png")
        img.save(png)
        return {"png_path": png, "n": len(mols)}

    def _run_prop_attribution(self, step: Dict, data: Dict) -> Dict[str, Any]:
        """
        FIXED explainability:
          - Leave-one-atom-out occlusion attribution:
              score_i = baseline_pred - pred(mask atom i -> wildcard)
          - Highlight ONLY meaningful atoms:
              * Rank by |score|
              * Apply relative threshold vs max |score| (default 0.25)
              * Cap by top-K
              * Ensure at least 1 atom highlighted
        """
        out_dir = Path("viz")
        out_dir.mkdir(parents=True, exist_ok=True)

        if Chem is None or Draw is None:
            return {"error": "RDKit not installed"}

        p = data.get("psmiles") or data.get("seed_psmiles")
        prop = canonical_property_name(data.get("property") or data.get("property_name") or "glass transition")
        top_k = int(data.get("top_k_atoms", data.get("top_k", 12)))

        # importance threshold controls
        min_rel_importance = float(data.get("min_rel_importance", 0.25))
        min_abs_importance = float(data.get("min_abs_importance", 0.0))

        if prop not in PROPERTY_HEAD_PATHS:
            return {"error": f"Unsupported property for attribution: {prop}"}
        if not p:
            return {"error": "no psmiles"}

        mol = Chem.MolFromSmiles(psmiles_to_rdkit_smiles(p))
        if mol is None:
            return {"error": "invalid psmiles"}

        num_atoms = mol.GetNumAtoms()
        if num_atoms <= 0:
            return {"error": "molecule has no atoms"}

        base_res = self._run_property_prediction({}, {"psmiles": p, "property": prop})
        if "error" in base_res or "predictions" not in base_res:
            return {"error": f"Baseline prediction failed: {base_res.get('error', 'unknown error')}"}

        baseline = base_res["predictions"].get(prop)
        if not isinstance(baseline, (float, int)):
            return {"error": "Baseline prediction not numeric"}

        # Occlusion loop (O(N_atoms) property predictions)
        scores: Dict[int, float] = {}
        for idx in range(num_atoms):
            try:
                tmp = Chem.RWMol(mol)
                tmp.GetAtomWithIdx(idx).SetAtomicNum(0)  # wildcard
                mutated = tmp.GetMol()
                mut_smiles = Chem.MolToSmiles(mutated)
                mut_psmiles = normalize_generated_psmiles_out(mut_smiles)  # [*] -> *
            except Exception:
                scores[idx] = 0.0
                continue

            mut_res = self._run_property_prediction({}, {"psmiles": mut_psmiles, "property": prop})
            mut_val = (mut_res.get("predictions") or {}).get(prop) if isinstance(mut_res, dict) else None
            if not isinstance(mut_val, (float, int)):
                scores[idx] = 0.0
            else:
                scores[idx] = float(baseline) - float(mut_val)

        # Select atoms: top-K by |score| but also require significance
        max_abs = max((abs(v) for v in scores.values()), default=0.0)
        rel_thresh = (min_rel_importance * max_abs) if max_abs > 0 else 0.0
        thresh = max(float(min_abs_importance), float(rel_thresh))

        ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)

        k_cap = max(1, min(top_k, num_atoms))
        selected = [i for i, v in ranked if abs(v) >= thresh]
        selected = selected[:k_cap]

        # Ensure at least one highlighted atom
        if not selected and ranked:
            selected = [ranked[0][0]]

        # Map colors (coolwarm) over selected only
        atom_colors: Dict[int, tuple] = {}
        sel_scores = np.array([scores[i] for i in selected], dtype=float)
        if cm is not None and sel_scores.size > 0:
            denom = (np.max(sel_scores) - np.min(sel_scores))
            if denom == 0:
                norm = np.full_like(sel_scores, 0.5)
            else:
                norm = (sel_scores - np.min(sel_scores)) / denom
            cmap = cm.get_cmap("coolwarm")
            for i, n in zip(selected, norm):
                r, g, b, _ = cmap(float(n))
                atom_colors[i] = (float(r), float(g), float(b))
        else:
            max_mag = max(abs(v) for v in sel_scores) if sel_scores.size else 1.0
            for i in selected:
                v = scores[i] / (max_mag or 1.0)
                if v >= 0:
                    atom_colors[i] = (1.0, 1.0 - 0.7 * v, 1.0 - 0.7 * v)
                else:
                    vv = abs(v)
                    atom_colors[i] = (1.0 - 0.7 * vv, 1.0 - 0.7 * vv, 1.0)

        try:
            img = Draw.MolToImage(
                mol,
                size=(700, 450),
                highlightAtoms=selected,
                highlightAtomColors=atom_colors,
            )
            png = str(out_dir / "prop_attribution.png")
            img.save(png)
            return {
                "png_path": png,
                "per_atom_scores": {int(i): float(v) for i, v in scores.items()},
                "highlighted_atoms": selected,
                "baseline_prediction": float(baseline),
                "property": prop,
                "method": "leave_one_atom_out_occlusion_thresholded_topk",
                "top_k_cap": int(k_cap),
                "selected_k": int(len(selected)),
                "min_rel_importance": float(min_rel_importance),
                "min_abs_importance": float(min_abs_importance),
                "used_threshold": float(thresh),
            }
        except Exception as e:
            return {"error": f"prop_attribution rendering failed: {e}"}

    # convenience
    def process_query(self, user_query: str, user_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        plan = self.analyze_query(user_query)
        results = self.execute_plan(plan, user_inputs)
        return results


if __name__ == "__main__":
    cfg = OrchestratorConfig()
    orch = PolymerOrchestrator(cfg)
    print("PolymerOrchestrator ready (5M heads + 5M inverse-design + LLM tool-calling planner + occlusion explainability).")
