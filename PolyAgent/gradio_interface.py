from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

# Load .env if present so OPENAI_API_KEY/OPENAI_MODEL are available
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import gradio as gr

try:
    from orchestrator import PolymerOrchestrator, OrchestratorConfig
except Exception as e:
    raise ImportError(
        "Could not import PolymerOrchestrator from orchestrator.py. "
        "Ensure the updated orchestrator file is present. "
        f"Original error: {e}"
    )


# =============================================================================
# DOI NORMALIZATION HELPERS
# =============================================================================
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

def normalize_doi(raw: str) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:\s*", "", s, flags=re.IGNORECASE)
    s = s.rstrip(").,;]}")
    return s if _DOI_RE.match(s) else None

def doi_to_url(doi: str) -> str:
    return f"https://doi.org/{doi}"

# -----------------------------------------------------------------------------
# Console defaults 
# -----------------------------------------------------------------------------
DEFAULT_CASE_BRIEF = (
    "We are developing a polymer film for high-barrier flexible packaging (food-contact). "
    "We need improved oxygen and water-vapor barrier while maintaining practical melt-processability "
    "(film extrusion/cast). Please use web_search to ground your recommendations in recent literature "
    "(last 5–10 years) on barrier improvement strategies (e.g., copolymerization, aromatic content, "
    "rigid side groups, crystallinity control, chain stiffness, and compatibilization). "
    "Constraints: avoid halogens; prioritize monomers with existing commercial suppliers; "
    "avoid overly brittle formulations."
)

DEFAULT_PROPERTY_NAME = "glass transition"
DEFAULT_SEED_PSMILES = "[*]CC(=O)OCCOCCOC(=O)C[*]"
DEFAULT_LITERATURE_QUERY = (
    "high barrier flexible packaging polyester copolymer Tg tuning oxygen permeability water vapor "
    "rigid aromatic units side groups 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
)
DEFAULT_TARGET_VALUE = 60.0
DEFAULT_NUM_GEN_SAMPLES = 6
DEFAULT_FETCH_TOP_N = 3

# Increased to help ensure >=10 citations in typical runs
DEFAULT_SEARCH_ROWS = 12

# Property-specific fallback targets (ONLY used when generation is requested but target not found in questions)
DEFAULT_TARGET_BY_PROPERTY = {
    "glass transition": 60.0,          # °C (example placeholder)
    "density": 1.20,                   # g/cm^3 (example placeholder)
    "melting": 150.0,                  # °C (example placeholder)
    "thermal decomposition": 350.0,    # °C (example placeholder)
}

# -----------------------------------------------------------------------------
# Run instructions bubble
# -----------------------------------------------------------------------------
RUN_INSTRUCTIONS_MD = (
    "### How to use PolyAgent\n"
    "\n"
    "PolyAgent is a web app with three **Tabs** at the top:\n"
    "- **PolyAgent Console** (main workflow)\n"
    "- **Tools** (run individual tools)\n"
    "- **Other LLMs** (baseline LLM-only answers)\n"
    "\n"
    "#### PolyAgent Console\n"
    "Use this Tab for the full, end-to-end run.\n"
    "1) In **Questions**, paste your request (one question or multiple).\n"
    "2) Click **Run PolyAgent**.\n"
    "3) Read the results in:\n"
    "   - **PolyAgent Answer**: the final structured response.\n"
    "   - **PNG Artifacts**: any available visuals (molecule render, generation grid, explainability heatmap).\n"
    "\n"
    "**Prompt tips (what PolyAgent detects automatically):**\n"
    "- **Inverse design / generation**: include words like `generate` or `inverse design` **and** include a numeric target\n"
    "  (examples: `target_value=60`, `target: 60`, `Tg 60`).\n"
    "- **Seed polymer**: provide a pSMILES either:\n"
    "  - inside a fenced code block, or\n"
    "  - with a keyed prefix like `seed_psmiles:`.\n"
    "- **Citations**: if you want a specific count, say it explicitly (example: `cite 10 papers`).\n"
    "\n"
    "#### Tools (debugging / run one step at a time)\n"
    "Use this Tab when you want to run a single tool and inspect its raw output.\n"
    "Each section is a collapsible **Accordion** with its own inputs and a run button:\n"
    "- **Data Extraction** (parse/canonicalize pSMILES; may also produce PNGs)\n"
    "- **Property Prediction**\n"
    "- **Polymer Generation (inverse design)**\n"
    "- **Web / RAG** (search + retrieval)\n"
    "- **Explainability**\n"
    "- **Diagnostics** (health checks, e.g., OpenAI probe)\n"
    "\n"
    "Outputs appear as JSON (for tool results) and/or PNGs (for visuals), depending on the tool.\n"
    "\n"
    "#### Other LLMs (no tools)\n"
    "Use this Tab to get a direct answer from a selected non-GPT model.\n"
    "It does **not** run PolyAgent tools (no property prediction, no generation tools, no retrieval).\n"
    "Pick a model, paste your prompt, and run it.\n"
)

def pretty_json(x: Any) -> str:
    try:
        return json.dumps(x, indent=2, ensure_ascii=False)
    except Exception:
        return str(x)


# -----------------------------------------------------------------------------
# Display normalization (MINIMAL): convert bracketed [At] endpoints to [*]
# -----------------------------------------------------------------------------
_AT_BRACKET_RE = re.compile(r"\[(at)\]", flags=re.IGNORECASE)


def _convert_at_to_star(psmiles: str) -> str:
    """
    Minimal, display-only conversion:
      - "[At]" / "[AT]" / ... -> "[*]"
    """
    if not isinstance(psmiles, str) or not psmiles:
        return psmiles
    return _AT_BRACKET_RE.sub("[*]", psmiles)


def _normalize_seed_inputs_for_display(obj: Any) -> Any:
    """
    Recursively normalize ONLY seed/input pSMILES fields for display.
    We do NOT touch generation outputs here to preserve exact tool-returned strings.
    """
    if isinstance(obj, str):
        if "[" in obj and "]" in obj and ("At" in obj or "AT" in obj or "at" in obj):
            return _convert_at_to_star(obj)
        return obj

    if isinstance(obj, list):
        return [_normalize_seed_inputs_for_display(x) for x in obj]

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("psmiles", "seed_psmiles", "seed_psmiles_used", "canonical_psmiles"):
                out[k] = _normalize_seed_inputs_for_display(v)
            else:
                out[k] = _normalize_seed_inputs_for_display(v)
        return out

    return obj

_ENDPOINT_TOKEN_RE = re.compile(r"\[\*\]")

def _escape_endpoint_tokens_for_markdown(text: str) -> str:
    """
    Escape '[*]' ONLY outside code blocks and inline code.
    This avoids turning '[*]' into '[\\*]' inside ```...``` where the backslash would show.
    """
    if not isinstance(text, str) or not text:
        return text

    # Split by fenced code blocks, keep delimiters
    parts = re.split(r"(```[\s\S]*?```)", text)
    out_parts = []

    for part in parts:
        # If this is a fenced code block, leave untouched
        if part.startswith("```") and part.endswith("```"):
            out_parts.append(part)
            continue

        # Split by inline code, keep delimiters
        subparts = re.split(r"(`[^`]*`)", part)
        for i, sp in enumerate(subparts):
            if sp.startswith("`") and sp.endswith("`"):
                continue
            subparts[i] = _ENDPOINT_TOKEN_RE.sub(r"[\\*]", sp)

        out_parts.append("".join(subparts))

    return "".join(out_parts)

# -----------------------------------------------------------------------------
# Auto-detect property / target_value / seed from Questions 
# -----------------------------------------------------------------------------
_NUM_RE = r"[-+]?\d+(?:\.\d+)?"

def _infer_property_from_questions(q: str) -> Optional[str]:
    """
    Infer canonical property name from free-text questions.
    Canonical keys must match orchestrator's PROPERTY_HEAD_PATHS/GENERATOR_DIRS keys.
    """
    s = (q or "").lower()

    # Allow explicit "property:" forms
    m = re.search(r"\bproperty\b\s*[:=]\s*([a-zA-Z _-]+)", s)
    if m:
        cand = m.group(1).strip().lower()
        # map common variants
        if "glass" in cand or re.search(r"\btg\b", cand):
            return "glass transition"
        if "density" in cand or re.search(r"\brho\b", cand):
            return "density"
        if "melting" in cand or re.search(r"\btm\b", cand):
            return "melting"
        if "decomp" in cand or "decomposition" in cand or re.search(r"\btd\b", cand):
            return "thermal decomposition"

    # Token-based inference
    if "thermal decomposition" in s or "decomposition temperature" in s or "decomposition" in s or re.search(r"\btd\b", s):
        return "thermal decomposition"
    if "glass transition" in s or "glass-transition" in s or re.search(r"\btg\b", s):
        return "glass transition"
    if "melting" in s or "melt temperature" in s or re.search(r"\btm\b", s):
        return "melting"
    if "density" in s or re.search(r"\brho\b", s):
        return "density"

    return None

def _infer_target_value_from_questions(q: str, prop: Optional[str]) -> Optional[float]:
    """
    Infer numeric target_value from free-text questions.
    - supports explicit: target_value=..., target: ..., tgt ...
    - supports property-attached: Tg 60, density 1.25, Td=380, Tm 180
    """
    sl = (q or "").lower()

    # Explicit
    m = re.search(rf"\b(target_value|target|tgt)\b\s*[:=]?\s*({_NUM_RE})", sl)
    if m:
        try:
            return float(m.group(2))
        except Exception:
            pass

    prop = (prop or "").strip().lower()
    prop_patterns: List[str] = []

    if prop == "glass transition":
        prop_patterns = [rf"\b(tg|glass\s*transition)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "density":
        prop_patterns = [rf"\b(density|rho)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "melting":
        prop_patterns = [rf"\b(tm|melting)\b\s*[:=]?\s*({_NUM_RE})"]
    elif prop == "thermal decomposition":
        prop_patterns = [rf"\b(td|thermal\s*decomposition|decomposition)\b\s*[:=]?\s*({_NUM_RE})"]

    for pat in prop_patterns:
        m = re.search(pat, sl)
        if m:
            try:
                return float(m.group(m.lastindex))
            except Exception:
                pass

    # Token-near-number fallback: pick first number within 80 chars after property token
    tokens: List[str] = []
    if prop == "glass transition":
        tokens = ["tg", "glass transition"]
    elif prop == "density":
        tokens = ["density", "rho"]
    elif prop == "melting":
        tokens = ["tm", "melting"]
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


def _infer_generate_intent(q: str) -> bool:
    """
    Decide if the user is asking for inverse design / generation.
    Conservative: only true when generation-ish verbs appear.
    """
    s = (q or "").lower()
    triggers = [
        "generate",
        "inverse design",
        "inverse-design",
        "design candidates",
        "propose candidates",
        "suggest candidates",
        "design polymer",
        "design polymers",
        "synthesize candidates",
        "optimize",
    ]
    return any(t in s for t in triggers)


def _infer_seed_psmiles_from_questions(q: str) -> Optional[str]:
    """
    Best-effort extraction of seed pSMILES from the Questions text without GUI changes.
    Supports:
      - seed_psmiles: <token>
      - psmiles=...
      - smiles=...
      - code block containing a single pSMILES/SMILES line
    """
    text = (q or "").strip()
    if not text:
        return None

    # 1) Prefer code block content 
    code_blocks = re.findall(r"```(?:\w+)?\s*([\s\S]*?)```", text)
    for block in code_blocks:
        for line in (block or "").splitlines():
            line = line.strip()
            if not line:
                continue
            # Heuristic: polymer pSMILES often includes [*] or [At]
            if "[*]" in line or "[At]" in line or "[AT]" in line or "*" in line or "[" in line:
                return line

    # 2) Keyed patterns
    m = re.search(r"(seed_psmiles|seed|psmiles|smiles)\s*[:=]\s*([^\s]+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(2).strip()

    return None

_SECOND_LEVEL_TLDS = {
    "co.uk",
    "ac.uk",
    "gov.uk",
    "org.uk",
    "co.jp",
    "ne.jp",
    "or.jp",
    "com.au",
    "net.au",
    "org.au",
    "edu.au",
    "co.in",
    "com.br",
    "com.cn",
}


def _root_domain(netloc: str) -> str:
    netloc = (netloc or "").strip().lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    parts = [p for p in netloc.split(".") if p]
    if len(parts) <= 2:
        return netloc
    last2 = ".".join(parts[-2:])
    last3 = ".".join(parts[-3:])
    # handle second-level public suffixes
    if last2 in _SECOND_LEVEL_TLDS and len(parts) >= 3:
        return last3
    if ".".join(parts[-2:]) in _SECOND_LEVEL_TLDS and len(parts) >= 3:
        return last3
    # if suffix looks like co.uk style
    if last2 in _SECOND_LEVEL_TLDS:
        return last3
    if last2.endswith(".uk") and len(parts) >= 3:
        if ".".join(parts[-2:]) in _SECOND_LEVEL_TLDS:
            return last3
    return last2

def _url_to_domain(url: str) -> Optional[str]:
    if not isinstance(url, str) or not url.strip():
        return None
    try:
        u = url.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            return None
        netloc = urlparse(u).netloc.strip().lower()
        if not netloc:
            return None
        return _root_domain(netloc)
    except Exception:
        return None


def _attach_source_domains(obj: Any) -> Any:
    """
    Recursively add a short source/domain field for RAG + web_search items where URLs are present.
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
    Build a compact citation index for web_search + rag retrieval items.
    Requirement:
      - Tag format is STRICTLY: COMPLETE DOI URL (https://doi.org/...) when DOI exists,
        otherwise the best available http(s) URL.
      - No numbered citations.
    """
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
        # DOI-first
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
                url = get_best_url(out)
                dom = out.get("source_domain") or (_url_to_domain(url) if url else None) or "source"
                tag = url.strip() if isinstance(url, str) and url.strip() else "source"
                # cite_tag must be DOI URL or URL fallback
                cur = out.get("cite_tag")
                if not (isinstance(cur, str) and cur.strip().startswith(("http://", "https://"))):
                    out["cite_tag"] = tag

                citation_index["sources"].append(
                    {
                        "tag": out.get("cite_tag"),
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
        tagged["citation_index"] = citation_index
        return tagged

    report["citation_index"] = citation_index
    return report


def ensure_orch(state: Dict[str, Any]) -> Tuple[PolymerOrchestrator, Dict[str, Any]]:
    if state.get("orch") is None:
        cfg = OrchestratorConfig()
        state["orch"] = PolymerOrchestrator(cfg)
        state["ctx"] = {}
        reason = getattr(state["orch"], "_openai_unavailable_reason", None)
        if reason:
            print("[OpenAI diagnostic]", reason)
    if "ctx" not in state:
        state["ctx"] = {}
    return state["orch"], state["ctx"]


# -----------------------------------------------------------------------------
# Extract tool output so the PLAN drives the final report
# -----------------------------------------------------------------------------
def _extract_tool_output(exec_res: Dict[str, Any], tool_name: str) -> Optional[Any]:
    """
    Best-effort extraction of a tool output from execute_plan() results.
    Supports a variety of common shapes:
      exec_res["steps"] = [{"tool": "...", "output": {...}}, ...]
      exec_res["steps"] = [{"tool": "...", "result": {...}}, ...]
      exec_res["steps"] = [{"tool": "...", "data": {...}}, ...]
    """
    if not isinstance(exec_res, dict):
        return None
    steps = exec_res.get("steps")
    if not isinstance(steps, list):
        return None

    tool_name = (tool_name or "").strip()
    if not tool_name:
        return None

    for s in steps:
        if not isinstance(s, dict):
            continue
        t = str(s.get("tool") or s.get("name") or "").strip()
        if t != tool_name:
            continue
        for k in ("output", "result", "data", "payload"):
            if k in s:
                return s.get(k)
        # fallback: sometimes the step dict itself is the output
        return s

    return None


def _compose_planner_prompt(
    case_brief: str,
    questions: str,
    property_name: str,
    seed_psmiles: str,
    literature_query: str,
    target_value: Optional[float],
) -> str:
    """
    Planner prompt updated to enforce:
      - per-question coverage
      - explicit mapping Qi -> steps
      - report_generation included as a planned step
    """
    lines = []
    lines.append("### CASE / CONTEXT (POLYMER SYSTEM)")
    if case_brief.strip():
        lines.append(case_brief.strip())
    if seed_psmiles.strip():
        lines.append(f"Seed pSMILES: {seed_psmiles.strip()}")
    if property_name.strip():
        lines.append(f"Primary property of interest: {property_name.strip()}")
    if target_value is not None:
        lines.append(f"Inverse-design target_value (required for generation): {target_value}")
    if literature_query.strip():
        lines.append(f"Literature query hint (optional): {literature_query.strip()}")

    lines.append("\n### USER QUESTIONS (ANSWER THESE)")
    q = questions.strip()
    if q:
        lines.append(q)
    else:
        lines.append(
            "Q1. Interpret the current formulation and key properties.\n"
            "Q2. Analyze structure–property relationships and root causes.\n"
            "Q3. Propose and (if possible) generate candidate polymers.\n"
            "Q4. Summarize evidence, limitations, and next experiments."
        )

    lines.append("\n### TOOLING REQUIREMENTS")
    lines.append(
        "- Select from tools: data_extraction, cl_encoding, property_prediction, polymer_generation,\n"
        "  rag_retrieval, web_search, report_generation, and PNG-only visual tools.\n"
        "- Plan a small, ordered tool chain (2–10 steps) that answers the USER QUESTIONS.\n"
        "- Ensure property_prediction uses cl_encoding output when possible.\n"
        "- polymer_generation is inverse design and REQUIRES target_value.\n"
        "- Do NOT answer the scientific questions yourself; only plan which tools to run."
    )

    # Critical: make the plan sensitive to the questions, not a fixed recipe
    lines.append("\n### PLANNING RULES (STRICT)")
    lines.append(
        "- Create an explicit mapping: for each question Qi, list the step numbers that address it.\n"
        "- Every planned step must contribute to at least one Qi.\n"
        "- If a Qi needs literature evidence, include web_search and/or rag_retrieval steps.\n"
        "- Include a final report_generation step that synthesizes tool outputs into answers for each Qi.\n"
        "- If a Qi cannot be answered from tools, plan to state 'not available' for missing numeric values "
        "and provide clearly labeled qualitative expectations where appropriate."
    )

    return "\n".join(lines)


def _seed_inputs(
    property_name: str,
    seed_psmiles: str,
    literature_query: str,
    target_value: Optional[float],
    questions: str,
) -> Dict[str, Any]:
    """
    Provide user_inputs to execute_plan(). Include questions so the orchestrator/tools
    can condition retrieval and synthesis on the actual user ask.
    """
    payload: Dict[str, Any] = {}
    if property_name.strip():
        payload["property"] = property_name.strip()
    if seed_psmiles.strip():
        payload["psmiles"] = seed_psmiles.strip()
    if literature_query.strip():
        payload["literature_query"] = literature_query.strip()
        payload["query"] = literature_query.strip()
    if target_value is not None:
        payload["target_value"] = float(target_value)
    payload["num_samples"] = int(DEFAULT_NUM_GEN_SAMPLES)
    if isinstance(questions, str) and questions.strip():
        payload["questions"] = questions.strip()
    return payload


def _maybe_add_artifacts(
    orch: PolymerOrchestrator,
    report: Dict[str, Any],
    seed_psmiles_fallback: Optional[str] = None,
    property_name_fallback: Optional[str] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    imgs: List[str] = []
    extras: Dict[str, Any] = {}

    # Generation grid
    try:
        gen = (report.get("summary", {}) or {}).get("generation", {})
        if isinstance(gen, dict) and gen.get("generated_psmiles"):
            grid = orch._run_gen_grid({}, {"polymer_generation": gen})
            if isinstance(grid, dict) and grid.get("png_path") and Path(grid["png_path"]).exists():
                imgs.append(grid["png_path"])
                extras["gen_grid"] = grid
    except Exception as e:
        extras["gen_grid_error"] = str(e)

    # Polymer render (seed)
    try:
        seed_psmiles = ((report.get("summary", {}) or {}).get("property_prediction", {}) or {}).get("psmiles")
        if not seed_psmiles:
            seed_psmiles = seed_psmiles_fallback
        if seed_psmiles:
            mol_png = orch._run_mol_render({}, {"psmiles": seed_psmiles, "view": "2d"})
            if isinstance(mol_png, dict) and mol_png.get("png_path") and Path(mol_png["png_path"]).exists():
                imgs.append(mol_png["png_path"])
                extras["mol_render"] = mol_png
    except Exception as e:
        extras["mol_render_error"] = str(e)

    # Explainability heatmap
    try:
        summary = report.get("summary", {}) or {}
        tool_outputs = report.get("tool_outputs", {}) or {}

        prop_pred = summary.get("property_prediction", {}) or {}
        data_ex = summary.get("data_extraction", {}) or tool_outputs.get("data_extraction", {}) or {}

        seed_psmiles = (
            prop_pred.get("psmiles")
            or data_ex.get("canonical_psmiles")
            or seed_psmiles_fallback
        )

        prop_name = (
            prop_pred.get("property")
            or property_name_fallback
            or DEFAULT_PROPERTY_NAME
        )

        if seed_psmiles:
            expl_payload = {"psmiles": seed_psmiles, "top_k_atoms": 12, "property": prop_name}
            expl = orch._run_prop_attribution({}, expl_payload)
            if isinstance(expl, dict) and expl.get("png_path") and Path(expl["png_path"]).exists():
                imgs.append(expl["png_path"])
                extras["prop_attribution"] = expl
            else:
                extras["prop_attribution_error"] = expl.get("error") if isinstance(expl, dict) else "unknown"
        else:
            extras["prop_attribution_error"] = "No seed pSMILES available for attribution."
    except Exception as e:
        extras["prop_attribution_error"] = str(e)

    return imgs, extras

def _requested_citation_count(questions: str, default_n: int = 10) -> int:
    """
    If the user explicitly asks for N citations/papers/sources/references, honor that.
    Otherwise, default to 10.
    """
    q = (questions or "").lower()

    patterns = [
        r"(?:at\s+least\s+)?(\d{1,3})\s*(?:citations|citation|papers|paper|sources|source|references|reference)\b",
        r"\bcite\s+(\d{1,3})\s*(?:papers|paper|sources|source|references|reference|citations|citation)\b",
        r"\b(\d{1,3})\s*(?:papers|paper|sources|source|references|reference|citations|citation)\s*(?:minimum|min)\b",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
                return max(1, min(n, 200))
            except Exception:
                pass
    return max(1, default_n)


def _collect_citations(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect citations from report['citation_index']['sources'] if present; otherwise walk the report.
    Deduplicate by DOI (preferred) or URL.
    """
    if not isinstance(report, dict):
        return []

    sources = []
    ci = report.get("citation_index")
    if isinstance(ci, dict) and isinstance(ci.get("sources"), list):
        for s in ci["sources"]:
            if isinstance(s, dict):
                sources.append(s)

    if not sources:
        def walk(node: Any):
            if isinstance(node, dict):
                if "url" in node or "doi" in node:
                    doi = normalize_doi(node.get("doi", "")) or ""
                    url = None
                    if doi:
                        url = doi_to_url(doi)
                    else:
                        url = node.get("url")
                    sources.append({
                        "domain": node.get("source_domain") or _url_to_domain(node.get("url") or ""),
                        "title": node.get("title") or node.get("name") or "Untitled",
                        "url": url,
                        "doi": doi,
                        "tag": url,
                    })
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)
        walk(report)

    # normalize + dedupe
    dedup: Dict[str, Dict[str, Any]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        url = s.get("url")
        doi = normalize_doi(s.get("doi", "")) or ""

        # Requirement: label should be COMPLETE DOI URL (preferred) else URL.
        tag = s.get("tag")
        if doi:
            cite_url = doi_to_url(doi)
        elif isinstance(url, str) and url.strip():
            cite_url = url.strip()
        else:
            continue

        key = None
        if doi:
            key = "doi:" + doi.lower()
        elif isinstance(cite_url, str) and cite_url.strip():
            key = "url:" + cite_url.strip()
        else:
            continue

        title = s.get("title") or "Untitled"

        dedup[key] = {
            "domain": cite_url,
            "title": title,
            "url": cite_url,
            "doi": doi,
            "tag": cite_url if isinstance(cite_url, str) else tag,
        }

    # stable-ish ordering: prefer items that have a URL and non-generic domain
    def _rank(x: Dict[str, Any]) -> Tuple[int, int, str]:
        dom = (x.get("domain") or "").lower()
        url = x.get("url") or ""
        generic = int(dom in ("source", "doi.org"))
        has_url = 0 if (isinstance(url, str) and url.startswith("http")) else 1
        return (generic, has_url, dom)

    out = list(dedup.values())
    out.sort(key=_rank)
    return out


def _build_sources_section(citations: List[Dict[str, Any]], n_needed: int) -> str:
    """
    Deterministic clickable source list.
    Requirement:
      - link text must be the COMPLETE DOI URL (preferred) else URL.
    Bullet format:
      - [https://doi.org/...](https://doi.org/...) — Title
    """
    if n_needed < 1:
        n_needed = 1

    picked: List[Dict[str, Any]] = []
    seen_urls: set = set()
    for c in citations:
        url = c.get("url")
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        picked.append(c)
        if len(picked) >= n_needed:
            break

    lines = []
    lines.append("\n\n---\n\n### Sources (clickable)\n")
    if not picked:
        lines.append("_No citable web/RAG sources were available in the report output._\n")
        return "".join(lines)

    if len(picked) < n_needed:
        lines.append(f"_Only {len(picked)} unique sources were available; target was {n_needed}._\n\n")

    for c in picked:
        cite_text = (c.get("domain") or c.get("url") or "source").strip()
        url = c.get("url")
        title = (c.get("title") or "Untitled").strip()
        lines.append(f"- [{cite_text}]({url}) — {title}\n")

    return "".join(lines)


def _augment_questions_for_grounding(questions: str, n_citations: int) -> str:
    """
    Updated grounding constraints:
      - Tool citations MUST be [T] only.
      - Paper citations MUST be clickable hyperlinks whose link text is the COMPLETE DOI URL (preferred).
      - Ensure at least n_citations unique citations unless user asked otherwise.
      - Do not repeat the same DOI/URL more than once.
    """
    constraints = (
        "\n\nCONSTRAINTS FOR THE ANSWER:\n"
        "- Do NOT manufacture DOIs or sources. Use only URLs/DOIs present in the provided report.\n"
        "- Tool-derived facts: cite inline using [T] (exactly; do NOT use [T1], [T2], etc.).\n"
        "- Literature/web/RAG citations: cite as clickable hyperlinks where the bracket text is the COMPLETE DOI URL "
        "(https://doi.org/...) when DOI is available; otherwise use the best available URL.\n"
        "- Do NOT use numbered bracket citations like [1], [2].\n"
        "- You are FORBIDDEN from adding a separate references list/section (e.g., 'References', 'Sources').\n"
        "- All literature citations must be inline hyperlinks: [https://doi.org/...](https://doi.org/...) placed immediately after the claim.\n"
        "- Distribute citations across the answer (do not cluster them in one place).\n"
        "- NON-DUPLICATES: Do not repeat the same paper link. Each DOI/URL may appear at most once in the entire answer.\n"
        "- Each major section should include at least 1 inline literature citation when relevant.\n"
        "- Numeric values: only use numeric values that appear in tool outputs; otherwise state 'not available'.\n"
        "- Qualitative expectations are allowed when numeric outputs are not available; label them clearly as qualitative.\n"
        "- When presenting polymer_generation outputs (e.g., generated_psmiles), reproduce them verbatim exactly as returned.\n"
        "- Polymer endpoint tokens: preserve attachment-point placeholders exactly as '[*]' in any pSMILES/SMILES shown.\n"
        "  Do NOT drop the '*' or render it as empty brackets '[]'.\n"
        f"- Citation minimum: include at least {int(n_citations)} NON-DUPLICATE literature citations (unique by URL/DOI), "
        "unless the user explicitly requested a different number.\n"
    )
    q = (questions or "").rstrip()
    return q + constraints


def _assign_tool_tags(plan: Dict[str, Any], exec_res: Dict[str, Any], report: Dict[str, Any]) -> None:
    """
    Tool tags are ALWAYS [T] (single tag only).
    """
    try:
        steps_executed = (exec_res or {}).get("steps", []) or []
        for s in steps_executed:
            if isinstance(s, dict):
                s["cite_tag"] = "[T]"
    except Exception:
        pass

    try:
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        if isinstance(summary, dict):
            for k, v in list(summary.items()):
                if isinstance(v, dict):
                    v["cite_tag"] = "[T]"
    except Exception:
        pass

    try:
        tool_outputs = report.get("tool_outputs", {}) if isinstance(report, dict) else {}
        if isinstance(tool_outputs, dict):
            for _, v in tool_outputs.items():
                if isinstance(v, dict):
                    v["cite_tag"] = "[T]"
    except Exception:
        pass


# -----------------------------------------------------------------------------
# PolyAgent Console
# -----------------------------------------------------------------------------
def run_agent(state: Dict[str, Any], questions: str) -> Tuple[str, List[str]]:
    orch, ctx = ensure_orch(state)

    # ---------- AUTO-DETECTION ----------
    qtxt = questions or ""

    inferred_prop = _infer_property_from_questions(qtxt) or DEFAULT_PROPERTY_NAME

    inferred_seed = _infer_seed_psmiles_from_questions(qtxt)
    seed_psmiles = _convert_at_to_star(inferred_seed) if inferred_seed else _convert_at_to_star(DEFAULT_SEED_PSMILES)

    want_generation = _infer_generate_intent(qtxt)

    inferred_target = _infer_target_value_from_questions(qtxt, inferred_prop)

    # Only default a target when the user appears to want generation but omitted an explicit value
    if inferred_target is None and want_generation:
        inferred_target = float(DEFAULT_TARGET_BY_PROPERTY.get(inferred_prop, DEFAULT_TARGET_VALUE))

    target_value: Optional[float] = float(inferred_target) if inferred_target is not None else None

    # Literature query
    literature_query_default = DEFAULT_LITERATURE_QUERY
    case_brief = DEFAULT_CASE_BRIEF
    property_name = inferred_prop

    # Planner prompt
    planner_prompt = _compose_planner_prompt(
        case_brief=case_brief,
        questions=qtxt,
        property_name=property_name,
        seed_psmiles=seed_psmiles,
        literature_query=literature_query_default,
        target_value=target_value,
    )
    plan = orch.analyze_query(planner_prompt)
    ctx["last_plan"] = plan

    # Execute plan with inferred inputs
    exec_inputs = _seed_inputs(
        property_name=property_name,
        seed_psmiles=seed_psmiles,
        literature_query=literature_query_default,
        target_value=target_value,
        questions=qtxt,
    )
    exec_res = orch.execute_plan(plan, user_inputs=exec_inputs)
    ctx["last_exec"] = exec_res

    # IMPORTANT: Prefer report_generation output from execute_plan (plan-driven)
    report = _extract_tool_output(exec_res, "report_generation")

    # Fallback if orchestrator didn't include report_generation in the executed plan
    if report is None:
        qhint = (qtxt or "").strip()
        if len(qhint) >= 20:
            lit_query = qhint
        else:
            lit_query = literature_query_default

        rep_inputs: Dict[str, Any] = {
            "questions": qtxt,
            "literature_query": lit_query,
            "query": lit_query,
            "psmiles": seed_psmiles,
            "property": property_name,
            "rows": int(DEFAULT_SEARCH_ROWS),
            "fetch_top_n": int(DEFAULT_FETCH_TOP_N),
            "fetch_top_n_arxiv": 1,
            "num_samples": int(DEFAULT_NUM_GEN_SAMPLES),
        }

        # Only request generation if we have a target_value (or generation intent + fallback target above)
        if target_value is not None:
            rep_inputs["generate"] = True
            rep_inputs["target_value"] = float(target_value)

        report = orch.generate_report(rep_inputs)

    if not isinstance(report, dict):
        report = {"summary": {"report_generation": {"text": str(report)}}}

    # Attach domains/citations
    report = _attach_source_domains(report)
    report = _index_citable_sources(report)

    # Tool tags: ALWAYS [T]
    _assign_tool_tags(plan=plan, exec_res=exec_res, report=report)

    # Normalize seed-related PSMILES for display only
    report = _normalize_seed_inputs_for_display(report)
    ctx["last_report"] = report

    # Artifacts
    imgs, extras = _maybe_add_artifacts(
        orch,
        report,
        seed_psmiles_fallback=seed_psmiles,
        property_name_fallback=property_name,
    )
    ctx.update(extras)

    # Decide required citation count (default 10 unless user asked otherwise)
    n_citations = _requested_citation_count(qtxt, default_n=10)
    ctx["required_citations"] = n_citations

    # Collect citations deterministically for an explicit clickable list
    citations = _collect_citations(report)
    ctx["citations_collected"] = len(citations)

    # Compose final answer with strict constraints
    guarded_questions = _augment_questions_for_grounding(qtxt, n_citations=n_citations)
    final_md, composer_imgs = orch.compose_gpt_style_answer(
        report,
        case_brief=case_brief,
        questions=guarded_questions,
    )

    final_md = _escape_endpoint_tokens_for_markdown(final_md)

    # Append deterministic source list to GUARANTEE explicit clickable citations
    # final_md = final_md.rstrip() + _build_sources_section(citations, n_needed=n_citations)

    for p in composer_imgs:
        if p not in imgs and Path(p).exists():
            imgs.append(p)

    return final_md, imgs


# ----------------------------- Advanced Tools ----------------------------- #
def tool_data_extraction(state: Dict[str, Any], psmiles: str) -> Tuple[str, List[str]]:
    orch, ctx = ensure_orch(state)
    psmiles = _convert_at_to_star(psmiles)
    out = orch._run_data_extraction({"step": 1}, {"psmiles": psmiles})
    ctx["data_extraction"] = out
    images: List[str] = []

    if isinstance(out, dict) and out.get("canonical_psmiles"):
        mimg = orch._run_mol_render({}, {"psmiles": out["canonical_psmiles"], "view": "2d"})
        if isinstance(mimg, dict) and mimg.get("png_path") and Path(mimg["png_path"]).exists():
            images.append(mimg["png_path"])

        expl = orch._run_prop_attribution({}, {"psmiles": out["canonical_psmiles"], "top_k_atoms": 12})
        if isinstance(expl, dict) and expl.get("png_path") and Path(expl["png_path"]).exists():
            images.append(expl["png_path"])

    return pretty_json(out), images


def tool_property_prediction(state: Dict[str, Any], property_name: str, psmiles: Optional[str]) -> str:
    orch, ctx = ensure_orch(state)
    payload: Dict[str, Any] = {"property": property_name}
    if psmiles:
        payload["psmiles"] = _convert_at_to_star(psmiles)
    if ctx.get("data_extraction"):
        payload["data_extraction"] = ctx["data_extraction"]
    if ctx.get("cl_encoding"):
        payload["cl_encoding"] = ctx["cl_encoding"]
    out = orch._run_property_prediction({"step": 3}, payload)
    ctx["property_prediction"] = out
    return pretty_json(out)


def tool_polymer_generation(
    state: Dict[str, Any], property_name: str, target_value: float, num_samples: int
) -> Tuple[str, List[str]]:
    orch, ctx = ensure_orch(state)
    payload: Dict[str, Any] = {
        "property": property_name,
        "target_value": float(target_value),
        "num_samples": int(num_samples),
    }
    out = orch._run_polymer_generation({"step": 4}, payload)
    ctx["polymer_generation"] = out

    images: List[str] = []
    try:
        grid = orch._run_gen_grid({}, {"polymer_generation": out})
        if isinstance(grid, dict) and grid.get("png_path") and Path(grid["png_path"]).exists():
            images.append(grid["png_path"])
    except Exception:
        pass

    return pretty_json(out), images


def tool_web_search(state: Dict[str, Any], source: str, query: str, rows: int) -> Tuple[str, List[str]]:
    orch, ctx = ensure_orch(state)
    out = orch._run_web_search({"step": 5}, {"source": source, "query": query, "rows": rows})
    out = _attach_source_domains(out)
    out = _index_citable_sources(out) if isinstance(out, dict) else out
    ctx.setdefault("web_search", {})[source] = out
    return pretty_json(out), []


def tool_rag_retrieval(state: Dict[str, Any], query: str) -> str:
    orch, ctx = ensure_orch(state)
    out = orch._run_rag_retrieval({"step": 7}, {"query": query})
    out = _attach_source_domains(out)
    out = _index_citable_sources(out) if isinstance(out, dict) else out
    ctx["rag_retrieval"] = out
    return pretty_json(out)


def tool_explainability(state: Dict[str, Any], psmiles: str, property_name: str) -> Tuple[str, List[str]]:
    orch, ctx = ensure_orch(state)
    psmiles = _convert_at_to_star(psmiles)
    payload: Dict[str, Any] = {"psmiles": psmiles, "top_k_atoms": 12}
    if property_name:
        payload["property"] = property_name
    out = orch._run_prop_attribution({"step": 8}, payload)
    images: List[str] = []
    if isinstance(out, dict) and out.get("png_path") and Path(out["png_path"]).exists():
        images.append(out["png_path"])
    return pretty_json(out), images


def tool_openai_probe(state: Dict[str, Any]) -> str:
    orch, _ = ensure_orch(state)
    if getattr(orch, "openai_client", None) is None or orch.openai_client is None:
        return pretty_json({"ok": False, "reason": getattr(orch, "_openai_unavailable_reason", "OpenAI client not available")})

    try:
        resp = orch.openai_client.chat.completions.create(
            model=orch.config.model,
            messages=[
                {"role": "system", "content": 'Return a tiny JSON object {"ok":true} and nothing else.'},
                {"role": "user", "content": "ping"},
            ],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content
    except Exception as e:
        return pretty_json({"ok": False, "error": str(e)})


# ----------------------------- GPT-only ----------------------------- #
def gpt_only_answer(state: Dict[str, Any], prompt: str) -> str:
    """
    Pure GPT-only responses. This function will not call orchestrator tools or perform web search.
    """
    orch, _ = ensure_orch(state)
    if getattr(orch, "openai_client", None) is None or orch.openai_client is None:
        return pretty_json({"ok": False, "reason": getattr(orch, "_openai_unavailable_reason", "OpenAI client not available")})

    p = (prompt or "").strip()
    if not p:
        return "Please provide a prompt."

    try:
        resp = orch.openai_client.chat.completions.create(
            model=orch.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a polymer R&D assistant. Answer directly and clearly. "
                        "Do not call tools or run web searches. If you are uncertain, state uncertainty."
                    ),
                },
                {"role": "user", "content": p},
            ],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return pretty_json({"ok": False, "error": str(e)})


# ----------------------------- Other LLMs (Hugging Face Inference) ----------------------------- #
def llm_only_answer(state: Dict[str, Any], model_name: str, prompt: str) -> str:
    """
    LLM-only responses using Hugging Face Inference API for non-GPT models.
    """
    ensure_orch(state)

    import os
    from huggingface_hub import InferenceClient

    HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip()
    if not HF_TOKEN:
        return pretty_json(
            {
                "ok": False,
                "error": "HF_TOKEN is not set. Add HF_TOKEN=hf_... to your .env or env vars.",
            }
        )

    HF_MODEL_MAP = {
        "mixtral-8x22b-instruct": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    }

    m = (model_name or "").strip()
    p = (prompt or "").strip()

    if not p:
        return "Please provide a prompt."
    if not m:
        return "Please select a model."

    model_id = HF_MODEL_MAP.get(m)
    if not model_id:
        return pretty_json(
            {
                "ok": False,
                "error": f"Unsupported model selection: {m}",
                "supported": list(HF_MODEL_MAP.keys()),
            }
        )

    client = InferenceClient(model=model_id, token=HF_TOKEN)

    try:
        resp = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a polymer R&D assistant. Answer directly and clearly. "
                        "Do not call tools or run web searches. If you are uncertain, state uncertainty."
                    ),
                },
                {"role": "user", "content": p},
            ],
            max_tokens=900,
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return pretty_json({"ok": False, "error": str(e), "model_id": model_id})


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        css="""
        .mono {font-family: ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace}
        .info-bubble {
            border: 1px solid rgba(15, 23, 42, 0.18);
            background: rgba(15, 23, 42, 0.04);
            border-radius: 18px;
            padding: 16px 18px;
            margin: 10px 0 14px 0;
        }
        """
    ) as demo:
        state = gr.State({})

        gr.Markdown("## PolyAgent 🧪\n")

        # Big bubble shown on load and retained (no dismiss / no state gating).
        gr.Markdown(RUN_INSTRUCTIONS_MD, elem_classes=["info-bubble"])

        with gr.Tabs():
            with gr.Tab("PolyAgent Console"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Questions")
                        questions = gr.Textbox(
                            label="Ask your questions",
                            lines=16,
                            placeholder=(
                                "Example:\n"
                                "1) For high-barrier flexible packaging films, what polymer design strategies improve OTR/WVTR?\n"
                                "2) What recent (2015–2025) literature supports these strategies? (cite 10 papers)\n"
                                "3) Suggest candidate polyester families and practical next experiments.\n"
                            ),
                        )
                        btn_run = gr.Button("Run PolyAgent", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### PolyAgent Answer")
                        final_answer = gr.Markdown("PolyAgent will respond here with a single structured answer.")
                        gr.Markdown("### PNG Artifacts (Molecule, Grid, Explainability)")
                        ev_imgs = gr.Gallery(label="", columns=3, height=260)

                btn_run.click(
                    fn=run_agent,
                    inputs=[state, questions],
                    outputs=[final_answer, ev_imgs],
                )

            with gr.Tab("Tools"):
                gr.Markdown("Run individual tools for debugging/ad-hoc usage. Visuals are PNG-only.")

                with gr.Accordion("Data Extraction", open=True):
                    psm_in = gr.Textbox(label="pSMILES")
                    btn_ex = gr.Button("Extract", variant="primary")
                    ex_json = gr.Code(label="Output", language="json", elem_classes=["mono"])
                    ex_imgs = gr.Gallery(label="PNG (molecule + explainability)", columns=3, height=220)
                    btn_ex.click(tool_data_extraction, [state, psm_in], [ex_json, ex_imgs])

                with gr.Accordion("Property Prediction", open=False):
                    prop = gr.Dropdown(
                        label="Property",
                        choices=["density", "glass transition", "melting", "thermal decomposition"],
                        value="glass transition",
                    )
                    psm_pred = gr.Textbox(label="Optional pSMILES (if not using previous extraction)")
                    btn_pred = gr.Button("Predict", variant="primary")
                    pred_json = gr.Code(label="Output", language="json", elem_classes=["mono"])
                    btn_pred.click(tool_property_prediction, [state, prop, psm_pred], [pred_json])

                with gr.Accordion("Polymer Generation (inverse design)", open=False):
                    prop_g = gr.Dropdown(
                        label="Property (select generator)",
                        choices=["density", "glass transition", "melting", "thermal decomposition"],
                        value="glass transition",
                    )
                    tgt = gr.Number(label="target_value (required)", value=60.0, precision=4)
                    ns = gr.Slider(1, 24, value=4, step=1, label="# Samples")
                    btn_gen = gr.Button("Generate", variant="primary")
                    gen_json = gr.Code(label="Output", language="json", elem_classes=["mono"])
                    gen_imgs = gr.Gallery(label="PNG (generation grid)", columns=3, height=220)
                    btn_gen.click(tool_polymer_generation, [state, prop_g, tgt, ns], [gen_json, gen_imgs])

                with gr.Accordion("Web / RAG", open=False):
                    src = gr.Dropdown(
                        label="Source",
                        choices=["crossref", "openalex", "epmc", "arxiv", "semanticscholar", "springer", "internetarchive", "all"],
                        value="all",
                    )
                    query = gr.Textbox(label="Query")
                    rows = gr.Slider(1, 50, value=12, step=1, label="rows")
                    btn_ws = gr.Button("Search", variant="primary")
                    ws_json = gr.Code(label="Output", language="json", elem_classes=["mono"])
                    ws_imgs = gr.Gallery(label="(not used)", columns=3, height=10)
                    btn_ws.click(tool_web_search, [state, src, query, rows], [ws_json, ws_imgs])

                    rag_q = gr.Textbox(label="RAG query (local polymer KB)")
                    btn_rag = gr.Button("Retrieve (RAG)", variant="secondary")
                    rag_json = gr.Code(label="Output", language="json", elem_classes=["mono"])
                    btn_rag.click(tool_rag_retrieval, [state, rag_q], [rag_json])

                with gr.Accordion("Explainability (top-K atom occlusion)", open=False):
                    psm_expl = gr.Textbox(label="pSMILES")
                    prop_expl = gr.Dropdown(
                        label="Property (for attribution)",
                        choices=["density", "glass transition", "melting", "thermal decomposition"],
                        value="glass transition",
                    )
                    btn_expl = gr.Button("Explain", variant="primary")
                    expl_json = gr.Code(label="Attribution data (JSON)", language="json", elem_classes=["mono"])
                    expl_imgs = gr.Gallery(label="PNG (heatmap)", columns=2, height=220)
                    btn_expl.click(tool_explainability, [state, psm_expl, prop_expl], [expl_json, expl_imgs])

                with gr.Accordion("Diagnostics", open=False):
                    btn_probe = gr.Button("Probe OpenAI (JSON ping)")
                    probe_json = gr.Code(label="Result", language="json", elem_classes=["mono"])
                    btn_probe.click(tool_openai_probe, [state], [probe_json])

            with gr.Tab("Other LLMs"):
                gr.Markdown("Run a direct LLM-only response (no tools, no web search) using a non-GPT model name.")

                llm_model = gr.Dropdown(
                    label="Model",
                    choices=["mixtral-8x22b-instruct", "llama-3.1-8b-instruct"],
                    value="mixtral-8x22b-instruct",
                )
                llm_prompt = gr.Textbox(label="Prompt", lines=10, placeholder="Enter your polymer question/prompt.")
                llm_btn = gr.Button("Run LLM", variant="primary")
                llm_out = gr.Markdown("The model response will appear here.")
                llm_btn.click(fn=llm_only_answer, inputs=[state, llm_model, llm_prompt], outputs=[llm_out])

        return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", type=str, default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=True)


if __name__ == "__main__":
    main()
