from __future__ import annotations

import os
import re
import time
import json
import hashlib
import pathlib
import tempfile
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Vector store, loaders, splitters
# --------------------------------------------------------------------------------------
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# --------------------------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------------------------
from langchain_openai import OpenAIEmbeddings

# --------------------------------------------------------------------------------------
# Tokenizer for true token-based multi-scale segmentation
# --------------------------------------------------------------------------------------
import tiktoken


def sanitize_text(text: str) -> str:
    """
    Remove surrogate pairs and invalid Unicode characters.
    Prevents UnicodeEncodeError when adding documents to ChromaDB.
    """
    if not text:
        return text
    # Replace surrogates and invalid chars with empty string
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


# --------------------------------------------------------------------------------------
# ARXIV, OPENALEX, EPMC API URLS
# --------------------------------------------------------------------------------------
ARXIV_SEARCH_URL = "http://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

DEFAULT_PERSIST_DIR = "chroma_polymer_db"
DEFAULT_TMP_DOWNLOAD_DIR = os.path.join(tempfile.gettempdir(), "polymer_rag_pdfs")
MANIFEST_NAME = "manifest.jsonl"

# --------------------------------------------------------------------------------------
# Balanced target distribution (total ~2000 PDFs)
# --------------------------------------------------------------------------------------
TARGET_CURATED = 100
TARGET_JOURNALS = 200
TARGET_ARXIV = 800
TARGET_OPENALEX = 600
TARGET_EPMC = 200
TARGET_DATABASES = 100

# --------------------------------------------------------------------------------------
# Polymer keywords
# --------------------------------------------------------------------------------------
POLYMER_KEYWORDS = [
    "polymer",
    "macromolecule",
    "macromolecular",
    "polymeric",
    "polymer informatics",
    "polymer chemistry",
    "polymer physics",
    "PSMILES",
    "pSMILES",
    "BigSMILES",
    "polymer SMILES",
    "polymer sequence",
    "polymer electrolyte",
    "polymer morphology",
    "polymer dielectric",
    "polymer electrolyte membrane",
    "block copolymer",
    "biopolymer",
    "polymer nanocomposite",
    "polymer foundation model",
    "self-supervised polymer",
    "masked language model polymer",
    "polymer transformer",
    "generative polymer",
    "copolymer",
    "polymerization",
    "polymer synthesis",
    "polymer characterization",
]

# --------------------------------------------------------------------------------------
# IUPAC Guidelines & Standards (polymer nomenclature and terminology standards)
# --------------------------------------------------------------------------------------
CURATED_IUPAC_STANDARDS: List[Dict[str, Any]] = [
    {
        "url": "https://iupac.org/wp-content/uploads/2019/07/140-Brief-Guide-to-Polymer-Nomenclature-Web-Final-d.pdf",
        "name": "IUPAC - Brief Guide to Polymer Nomenclature",
        "meta": {
            "title": "A Brief Guide to Polymer Nomenclature (IUPAC Technical Report)",
            "year": "2012",
            "venue": "IUPAC Pure and Applied Chemistry",
            "source": "curated_iupac_standard",
        },
    },
    {
        "url": "https://rseq.org/wp-content/uploads/2022/10/20220816-English-BriefGuidePolymerTerminology-IUPAC.pdf",
        "name": "IUPAC - Brief Guide to Polymerization Terminology",
        "meta": {
            "title": "A Brief Guide to Polymerization Terminology (IUPAC Recommendations)",
            "year": "2022",
            "venue": "IUPAC",
            "source": "curated_iupac_standard",
        },
    },
    {
        "url": "https://www.rsc.org/images/richard-jones-naming-polymers_tcm18-243646.pdf",
        "name": "RSC - Naming Polymers",
        "meta": {
            "title": "Naming Polymers (RSC Educational Resource)",
            "year": "2020",
            "venue": "Royal Society of Chemistry",
            "source": "curated_iupac_standard",
        },
    },
]

# --------------------------------------------------------------------------------------
# ISO/ASTM Standards (polymer testing and characterization standards)
# --------------------------------------------------------------------------------------
CURATED_ISO_ASTM_STANDARDS: List[Dict[str, Any]] = [
    {
        "url": "https://cdn.standards.iteh.ai/samples/76910/29c8e7af07bd4188b297c39684ada79e/ISO-ASTM-52925-2022.pdf",
        "name": "ISO/ASTM 52925:2022 - Additive Manufacturing Polymers",
        "meta": {
            "title": "ISO/ASTM 52925:2022 Additive manufacturing of polymers - Feedstock materials",
            "year": "2022",
            "venue": "ISO/ASTM",
            "source": "curated_iso_astm_standard",
        },
    },
    {
        "url": "https://cdn.standards.iteh.ai/samples/76909/b9883b2f204248aca175e2f574bd879c/ISO-ASTM-52924-2023.pdf",
        "name": "ISO/ASTM 52924:2023 - Additive Manufacturing Qualification",
        "meta": {
            "title": "ISO/ASTM 52924:2023 Additive manufacturing of polymers - Qualification principles",
            "year": "2023",
            "venue": "ISO/ASTM",
            "source": "curated_iso_astm_standard",
        },
    },
    {
        "url": "https://nvlpubs.nist.gov/nistpubs/ir/2015/NIST.IR.8059.pdf",
        "name": "NIST IR 8059 - Materials Testing Standards for Additive Manufacturing",
        "meta": {
            "title": "Materials Testing Standards for Additive Manufacturing of Polymer Materials",
            "year": "2015",
            "venue": "NIST",
            "source": "curated_iso_astm_standard",
        },
    },
]

# --------------------------------------------------------------------------------------
# Foundational polymer informatics papers
# --------------------------------------------------------------------------------------
CURATED_POLYMER_INFORMATICS: List[Dict[str, Any]] = [
    {
        "url": "https://ramprasad.mse.gatech.edu/wp-content/uploads/2021/01/polymer-informatics.pdf",
        "name": "Polymer Informatics - Current Status and Critical Next Steps",
        "meta": {
            "title": "Polymer informatics: Current status and critical next steps",
            "year": "2020",
            "venue": "Materials Science and Engineering: R",
            "source": "curated_review_informatics",
        },
    },
    {
        "url": "https://arxiv.org/pdf/2011.00508.pdf",
        "name": "Polymer Informatics - Current Status (arXiv)",
        "meta": {
            "title": "Polymer Informatics: Current Status and Critical Next Steps",
            "year": "2020",
            "venue": "arXiv:2011.00508",
            "source": "curated_review_informatics",
        },
    },
]

# --------------------------------------------------------------------------------------
# BigSMILES notation papers (polymer representation standards)
# --------------------------------------------------------------------------------------
CURATED_BIGSMILES: List[Dict[str, Any]] = [
    {
        "url": "https://pubs.acs.org/doi/pdf/10.1021/acscentsci.9b00476",
        "name": "BigSMILES - Structurally-Based Line Notation",
        "meta": {
            "title": "BigSMILES: A Structurally-Based Line Notation for Describing Macromolecules",
            "year": "2019",
            "venue": "ACS Central Science",
            "source": "curated_bigsmiles",
        },
    },
    {
        "url": "https://www.rsc.org/suppdata/d3/dd/d3dd00147d/d3dd00147d1.pdf",
        "name": "Generative BigSMILES - Supplementary Information",
        "meta": {
            "title": "Generative BigSMILES: an extension for polymer informatics (SI)",
            "year": "2024",
            "venue": "RSC Digital Discovery",
            "source": "curated_bigsmiles",
        },
    },
]

# --------------------------------------------------------------------------------------
# Combine all curated sources
# --------------------------------------------------------------------------------------
CURATED_POLYMER_PDF_SOURCES = (
    CURATED_IUPAC_STANDARDS
    + CURATED_ISO_ASTM_STANDARDS
    + CURATED_POLYMER_INFORMATICS
    + CURATED_BIGSMILES
)

# --------------------------------------------------------------------------------------
# Major polymer journals with OA content
# --------------------------------------------------------------------------------------
POLYMER_JOURNAL_QUERIES = [
    # ACS Journals
    {"journal": "Macromolecules", "issn": "0024-9297", "publisher": "ACS"},
    {"journal": "ACS Polymers Au", "issn": "2768-1939", "publisher": "ACS"},
    {"journal": "ACS Applied Polymer Materials", "issn": "2637-6105", "publisher": "ACS"},
    {"journal": "Biomacromolecules", "issn": "1525-7797", "publisher": "ACS"},
    {"journal": "ACS Macro Letters", "issn": "2161-1653", "publisher": "ACS"},
    # RSC Journals
    {"journal": "Polymer Chemistry", "issn": "1759-9954", "publisher": "RSC"},
    {"journal": "RSC Applied Polymers", "issn": "2755-0656", "publisher": "RSC"},
    {"journal": "Soft Matter", "issn": "1744-683X", "publisher": "RSC"},
    # Springer/Nature Journals
    {"journal": "Polymer Journal", "issn": "0032-3896", "publisher": "Nature"},
    {"journal": "Journal of Polymer Science", "issn": "2642-4169", "publisher": "Wiley"},
    # Additional OA Journals
    {"journal": "Polymer Science and Technology", "issn": "2837-0341", "publisher": "ACS"},
    {"journal": "Polymers", "issn": "2073-4360", "publisher": "MDPI"},
]

DEFAULT_MAILTO = "kaur-m43@webmail.uwinnipeg.ca"  # polite defaults


# --------------------------------------------------------------------------------------
# DEDUPLICATION, DOWNLOAD, MANIFEST HELPERS
# --------------------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    name = str(name or "").strip().replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^a-zA-Z0-9._\-]", "_", name)
    return name[:200]


def is_probably_pdf(raw: bytes, content_type: str) -> bool:
    if not raw:
        return False
    if raw[:4] == b"%PDF":
        return True
    return "pdf" in (content_type or "").lower()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_manifest(out_dir: str, record: Dict[str, Any]) -> None:
    try:
        ensure_dir(out_dir)
        with open(os.path.join(out_dir, MANIFEST_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_manifest(out_dir: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    try:
        mpath = os.path.join(out_dir, MANIFEST_NAME)
        if not os.path.exists(mpath):
            return data
        with open(mpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    p = rec.get("path")
                    sha = rec.get("sha256")
                    if p:
                        data[p] = rec
                    if sha:
                        data[sha] = rec
                except Exception:
                    continue
    except Exception:
        pass
    return data


# --------------------------------------------------------------------------------------
# DOWNLOAD SINGLE PDF
# --------------------------------------------------------------------------------------
def download_pdf(
    url: str,
    out_dir: str,
    suggested_name: Optional[str] = None,
    timeout: int = 60,
    meta: Optional[Dict[str, Any]] = None,
    manifest: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Download a PDF and return local file path, or None on failure.
    Deduplicates by SHA256 content hash.
    Writes manifest record if meta provided.
    """
    try:
        headers = {"User-Agent": f"polymer-rag/1.0 ({DEFAULT_MAILTO})"}
        with requests.get(
            url, headers=headers, timeout=timeout, stream=True, allow_redirects=True
        ) as r:
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "")
            raw = r.content
            if not raw or not is_probably_pdf(raw, content_type):
                return None

            sha = sha256_bytes(raw)
            ensure_dir(out_dir)

            # Check manifest for existing SHA
            if manifest and sha in manifest:
                existing_path = manifest[sha].get("path")
                if existing_path and os.path.exists(existing_path):
                    return existing_path

            # Check filesystem for existing files with this hash
            existing = list(pathlib.Path(out_dir).glob(f"{sha[:16]}*.pdf"))
            if existing:
                path = str(existing[0])
                if meta:
                    rec = dict(meta)
                    rec.update({"sha256": sha, "path": path})
                    append_manifest(out_dir, rec)
                return path

            base = suggested_name or pathlib.Path(url).name or "paper.pdf"
            base = safe_filename(base)
            if not base.lower().endswith(".pdf"):
                base += ".pdf"
            fname = f"{sha[:16]}_{base}"
            fpath = os.path.join(out_dir, fname)

            with open(fpath, "wb") as f:
                f.write(raw)

            if meta:
                rec = dict(meta)
                rec.update({"sha256": sha, "path": fpath})
                append_manifest(out_dir, rec)

            return fpath
    except Exception:
        return None


def retry(fn, args, retries=3, sleep=0.6, **kwargs):
    for i in range(retries):
        out = fn(*args, **kwargs)
        if out:
            return out
        time.sleep(sleep * (2**i))
    return None


def download_one(entry: Union[str, Dict[str, Any]], out_dir: str, manifest: Dict):
    if isinstance(entry, dict):
        return download_pdf(
            entry["url"],
            out_dir,
            suggested_name=entry.get("name"),
            meta=entry.get("meta"),
            manifest=manifest,
        )
    return download_pdf(entry, out_dir, manifest=manifest)


def parallel_download_pdfs(
    entries: List[Union[str, Dict[str, Any]]],
    out_dir: str,
    manifest: Dict[str, Dict[str, Any]],
    max_workers: int = 12,
    desc: str = "Downloading PDFs",
) -> List[str]:
    ensure_dir(out_dir)
    results: List[str] = []
    if not entries:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(retry, download_one, (e, out_dir, manifest)) for e in entries]
        for f in tqdm(as_completed(futs), total=len(futs), desc=desc):
            p = f.result()
            if p:
                results.append(p)
    return results


# --------------------------------------------------------------------------------------
# ARXIV
# --------------------------------------------------------------------------------------
def arxiv_query_from_keywords(keywords: List[str]) -> str:
    kw = [k.replace(" ", "+") for k in keywords]
    terms = " OR ".join([f"ti:{k}" for k in kw] + [f"abs:{k}" for k in kw])
    cats = (
        "cat:cond-mat.mtrl-sci OR cat:cond-mat.soft OR cat:physics.chem-ph OR cat:cs.LG OR cat:stat.ML"
    )
    return f"({terms}) AND ({cats})"


def fetch_arxiv_pdf_urls(keywords: List[str], max_results: int = 800) -> List[str]:
    """
    Extract explicit pdf links and fallback to building from id entries.
    """
    query = arxiv_query_from_keywords(keywords)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": f"polymer-rag/1.0 ({DEFAULT_MAILTO})"}
    try:
        resp = requests.get(ARXIV_SEARCH_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        xml = resp.text
    except Exception:
        return []

    pdfs: List[str] = []
    seen = set()

    # explicit pdf hrefs
    for p in re.findall(r'href="(https?://arxiv\.org/pdf[^"]*)"', xml):
        if p not in seen:
            pdfs.append(p)
            seen.add(p)

    # fallback: build from id entries
    for aid in re.findall(r'<id>(https?://arxiv\.org/abs[^<]*)</id>', xml):
        m = re.search(r"arxiv\.org/abs/([^?v]+)", aid)
        if m:
            identifier = m.group(1)
            pdf = f"https://arxiv.org/pdf/{identifier}.pdf"
            if pdf not in seen:
                pdfs.append(pdf)
                seen.add(pdf)

    return pdfs


def fetch_arxiv_pdfs(
    keywords: List[str],
    out_dir: str,
    manifest: Dict[str, Dict[str, Any]],
    max_results: int = 800,
) -> List[str]:
    urls = fetch_arxiv_pdf_urls(keywords, max_results=max_results)
    entries = [
        {
            "url": u,
            "name": u.rstrip("/").split("/")[-1],
            "meta": {"source": "arxiv", "url": u},
        }
        for u in urls
    ]
    paths = parallel_download_pdfs(entries, out_dir, manifest, max_workers=8, desc="arXiv PDFs")
    return paths


# --------------------------------------------------------------------------------------
# OPENALEX
# --------------------------------------------------------------------------------------
def openalex_fetch_works_try(
    search: str,
    filter_str: str,
    per_page: int,
    page: int,
    mailto: Optional[str],
) -> Dict[str, Any]:
    headers = {"User-Agent": f"polymer-rag/1.0 ({mailto or DEFAULT_MAILTO})"}
    params: Dict[str, Any] = {
        "search": search,
        "per-page": per_page,
        "per_page": per_page,
        "page": page,
        "sort": "publication_date:desc",
    }
    if filter_str:
        params["filter"] = filter_str
    if mailto:
        params["mailto"] = mailto

    resp = requests.get(OPENALEX_WORKS_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


def openalex_fetch_works(
    keywords: List[str],
    max_results: int = 600,
    per_page: int = 200,
    mailto: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Try multiple query forms with relaxed filters if needed.
    """
    kws = sorted(set(keywords or []), key=str.lower)
    combined = " ".join(kws)
    or_query = " OR ".join(kws)

    attempts = [
        {"q": combined, "filter": "is_oa:true,language:en"},
        {"q": or_query, "filter": "is_oa:true,language:en"},
        {"q": or_query, "filter": "is_oa:true"},
        {"q": or_query, "filter": ""},
    ]

    works: List[Dict[str, Any]] = []
    for attempt in attempts:
        search = attempt["q"]
        filter_str = attempt["filter"]
        page = 1
        while len(works) < max_results:
            try:
                data = openalex_fetch_works_try(
                    search, filter_str, per_page, page, mailto or DEFAULT_MAILTO
                )
            except Exception as e:
                print(f"[WARN] OpenAlex request failed: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            works.extend(results)
            if len(results) < per_page:
                break
            page += 1
            time.sleep(0.12)

        if len(works) >= max_results:
            break
        if works:
            break

    return works[:max_results]


def openalex_extract_pdf_entries(
    works: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract candidate PDF URLs and metadata from OpenAlex works.
    """
    out: List[Dict[str, Any]] = []
    seen_urls = set()

    for w in works:
        pdf = ""
        best = w.get("best_oa_location") or {}
        if isinstance(best, dict):
            pdf = best.get("pdf_url") or best.get("url_for_pdf") or best.get("url") or ""
        if not pdf:
            pl = w.get("primary_location") or {}
            if isinstance(pl, dict):
                pdf = (
                    pl.get("pdf_url")
                    or pl.get("url_for_pdf")
                    or pl.get("landing_page_url")
                    or ""
                )
        if not pdf:
            oa = w.get("open_access") or {}
            if isinstance(oa, dict):
                pdf = oa.get("oa_url") or oa.get("oa_url_for_pdf") or ""
        if not pdf or pdf in seen_urls:
            continue
        seen_urls.add(pdf)

        title = (w.get("title") or w.get("display_name") or "").strip()
        year = w.get("publication_year") or w.get("publication_date") or ""
        venue = ""
        pl = w.get("primary_location") or {}
        if isinstance(pl, dict):
            venue = (pl.get("source") or {}).get("display_name") or ""
        if not venue:
            venue = (w.get("host_venue") or {}).get("display_name") or "".strip()

        name = " - ".join([s for s in [title, venue, str(year) or ""] if s])

        meta = {"title": title, "year": year, "venue": venue, "source": "openalex"}
        out.append({"url": pdf, "name": name, "meta": meta})

    return out


def fetch_openalex_pdfs(
    keywords: List[str],
    out_dir: str,
    manifest: Dict[str, Dict[str, Any]],
    max_results: int = 600,
    mailto: Optional[str] = None,
) -> List[str]:
    works = openalex_fetch_works(keywords, max_results=max_results, mailto=mailto)
    if not works:
        print("[INFO] OpenAlex returned no works for given queries/filters.")
        return []

    entries = openalex_extract_pdf_entries(works)
    if not entries:
        print("[INFO] OpenAlex works found, but no PDF links extracted.")
        return []

    paths = parallel_download_pdfs(
        entries, out_dir, manifest, max_workers=16, desc="OpenAlex PDFs"
    )
    return paths


# --------------------------------------------------------------------------------------
# EUROPE PMC
# --------------------------------------------------------------------------------------
def epmc_query_from_keywords(keywords: List[str]) -> str:
    return " OR ".join([f'"{k}"' for k in keywords])


def epmc_extract_pdf_entries_from_results(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()

    for r in results:
        ftl = r.get("fullTextUrlList") or {}
        urls: List[str] = []
        if isinstance(ftl, dict):
            for ful in ftl.get("fullTextUrl") or []:
                if isinstance(ful, dict):
                    u = ful.get("url") or ""
                    if u:
                        urls.append(u)
        if not urls:
            fu = r.get("fullTextUrl")
            if isinstance(fu, str) and fu:
                urls.append(fu)

        for u in urls:
            if not u or u in seen:
                continue
            seen.add(u)

            title = r.get("title") or "".strip()
            year = r.get("firstPublicationDate") or r.get("pubYear") or ""
            name = " - ".join([s for s in [title, str(year) or ""] if s])

            out.append(
                {
                    "url": u,
                    "name": name,
                    "meta": {"title": title, "year": year, "source": "epmc"},
                }
            )

    return out


def fetch_epmc_pdfs(
    keywords: List[str],
    out_dir: str,
    manifest: Dict[str, Dict[str, Any]],
    max_results: int = 200,
    page_size: int = 25,
) -> List[str]:
    """
    Query Europe PMC and extract fullTextUrlList entries.
    """
    q = epmc_query_from_keywords(keywords)
    params = {
        "query": q,
        "format": "json",
        "pageSize": page_size,
        "sort": "FIRST_PDATE desc",
    }
    headers = {"User-Agent": f"polymer-rag/1.0 ({DEFAULT_MAILTO})"}
    saved: List[str] = []
    cursor = 1
    total_fetched = 0

    while total_fetched < max_results:
        params["page"] = cursor
        try:
            resp = requests.get(EPMC_SEARCH_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[WARN] Europe PMC request failed: {e}")
            break

        results = (data.get("resultList") or {}).get("result") or []
        if not results:
            break

        entries = epmc_extract_pdf_entries_from_results(results)
        if not entries:
            cursor += 1
            total_fetched += len(results)
            time.sleep(0.2)
            continue

        paths = parallel_download_pdfs(entries, out_dir, manifest, max_workers=8, desc="Europe PMC PDFs")
        saved.extend(paths)

        total_fetched += len(results)
        cursor += 1
        time.sleep(0.2)

    return saved


# --------------------------------------------------------------------------------------
# POLYMER JOURNALS OA
# --------------------------------------------------------------------------------------
def fetch_polymer_journal_pdfs(
    journal_queries: List[Dict[str, Any]],
    out_dir: str,
    manifest: Dict[str, Dict[str, Any]],
    max_per_journal: int = 50,
    mailto: Optional[str] = None,
) -> List[str]:
    """
    Fetch OA papers from specific polymer journals via OpenAlex.
    """
    all_paths: List[str] = []
    for jq in journal_queries:
        journal_name = jq["journal"]
        issn = jq.get("issn", "")
        publisher = jq.get("publisher", "")
        print(f"→ Fetching from {journal_name} ({publisher})...")

        # Build OpenAlex filter for this journal
        filter_parts = ["is_oa:true", "language:en"]
        if issn:
            filter_parts.append(f"primary_location.source.issn:{issn}")
        filter_str = ",".join(filter_parts)

        # Search for polymer-related content in this journal
        search_query = "polymer OR macromolecule OR copolymer"
        page = 1
        journal_works = []
        while len(journal_works) < max_per_journal:
            try:
                data = openalex_fetch_works_try(
                    search_query, filter_str, 25, page, mailto or DEFAULT_MAILTO
                )
            except Exception as e:
                print(f"[WARN] Failed to fetch {journal_name}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break
            journal_works.extend(results)
            if len(results) < 25:
                break
            page += 1
            time.sleep(0.15)

        if journal_works:
            entries = openalex_extract_pdf_entries(journal_works[:max_per_journal])
            # Tag with journal source
            for e in entries:
                e["meta"]["journal"] = journal_name
                e["meta"]["publisher"] = publisher
                e["meta"]["source"] = f"{journal_name}_{publisher}".lower()

            paths = parallel_download_pdfs(
                entries, out_dir, manifest, max_workers=8, desc=f"{journal_name} PDFs"
            )
            all_paths.extend(paths)
            print(f"  → Downloaded {len(paths)} PDFs from {journal_name}")
            time.sleep(0.3)

    return all_paths


# --------------------------------------------------------------------------------------
# WRAPPER FOR OPENAI EMBEDDINGS
# --------------------------------------------------------------------------------------
class PolymerStyleOpenAIEmbeddings(OpenAIEmbeddings):
    """
    OpenAI embeddings wrapper for polymer RAG.
    Default model: text-embedding-3-small (1536-D) ← FIXED
    """

    def __init__(self, model: str = "text-embedding-3-small", **kwargs):
        super().__init__(model=model, **kwargs)


# --------------------------------------------------------------------------------------
# TOKENIZER FOR TRUE TOKEN-BASED SEGMENTATION
# --------------------------------------------------------------------------------------
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    if not text:
        return 0
    return len(TOKENIZER.encode(text))


# --------------------------------------------------------------------------------------
# METADATA ENRICHMENT FROM MANIFEST
# --------------------------------------------------------------------------------------
def attach_extra_metadata_from_manifest(
    docs: List[Any], manifest: Dict[str, Dict[str, Any]]
) -> None:
    """
    Enrich Document metadata with manifest data for later citation.
    """
    for d in docs:
        src_path = d.metadata.get("source", "")
        if not src_path:
            continue

        rec = manifest.get(src_path)
        if not rec:
            for k, v in manifest.items():
                if os.path.basename(k) == os.path.basename(src_path):
                    rec = v
                    break
        if rec:
            for k in ["title", "year", "venue", "url", "source", "journal", "publisher"]:
                if k in rec:
                    d.metadata[k] = rec[k]


# --------------------------------------------------------------------------------------
# MULTI-SCALE CHUNKING
# --------------------------------------------------------------------------------------
def multiscale_chunk_documents(
    docs: List[Any], min_chunk_tokens: int = 32
) -> List[Any]:
    """
    Multi-scale segmentation at TOKEN level: 512, 256, 128 token windows.
    """
    splitter_specs = [
        ("tokens=512", 512, 64),  # 50% tokens overlap
        ("tokens=256", 256, 48),
        ("tokens=128", 128, 32),
    ]

    all_chunks: List[Any] = []
    seg_id = 0

    for scale_label, chunk_size, overlap in splitter_specs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        splits = splitter.split_documents(docs)
        for d in splits:
            if token_length(d.page_content or "") < min_chunk_tokens:
                continue
            d.metadata = dict(d.metadata or {})
            d.metadata["segment_scale"] = scale_label
            d.metadata["segment_id"] = seg_id
            seg_id += 1
            all_chunks.append(d)

    return all_chunks


# --------------------------------------------------------------------------------------
# BUILD RETRIEVER FROM LOCAL PDFs
# --------------------------------------------------------------------------------------
def _split_and_build_retriever(
    documents_dir: str,
    persist_dir: Optional[str] = None,
    k: int = 10,
    embedding_model: str = "text-embedding-3-small",
    vector_backend: str = "chroma",
    min_chunk_tokens: int = 32,
    api_key: Optional[str] = None,
):
    """
    Load PDFs, chunk multi-scale, build dense retriever.
    FIXED: Always uses text-embedding-3-small (1536-D) and handles existing DB correctly.
    """
    print(f"→ Loading PDFs from {documents_dir}...")
    try:
        loader = DirectoryLoader(
            documents_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True,
        )
    except TypeError:
        loader = DirectoryLoader(
            documents_dir,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )

    docs = loader.load()
    if not docs:
        raise RuntimeError("No PDF documents found to index.")

    manifest = load_manifest(documents_dir)
    attach_extra_metadata_from_manifest(docs, manifest)

    documents = multiscale_chunk_documents(docs, min_chunk_tokens=min_chunk_tokens)
    print(
        f"→ Created {len(documents)} multi-scale segments from {len(docs)} PDFs (512/256/128-token windows)."
    )

    print(f"→ Using OpenAI embeddings model: {embedding_model}")
    embeddings = PolymerStyleOpenAIEmbeddings(model=embedding_model, api_key=api_key)

    if vector_backend.lower() == "chroma":
        if persist_dir and os.path.exists(persist_dir):
            print(f"→ Deleting existing Chroma database at {persist_dir} to prevent dimension mismatch...")
            import shutil
            shutil.rmtree(persist_dir)
            print(f"→ Existing database deleted. Creating fresh database...")

        # Sanitize all text content to prevent Unicode errors
        for doc in documents:
            doc.page_content = sanitize_text(doc.page_content or "")
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    doc.metadata[key] = sanitize_text(value)

        # Process in batches to avoid rate limiting and memory issues
        batch_size = 500  # Adjust based on your document sizes (500 is safe for most cases)
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"→ Processing {len(documents)} documents in {total_batches} batches of {batch_size}...")

        vector_store = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"  → Embedding batch {batch_num}/{total_batches} ({len(batch)} documents)...")

            if vector_store is None:
                # First batch: create the vector store
                if persist_dir:
                    print(f"    → Creating new Chroma database at {persist_dir}")
                    vector_store = Chroma.from_documents(
                        batch, embeddings, persist_directory=persist_dir
                    )
                else:
                    # In-memory mode also needs batching
                    vector_store = Chroma.from_documents(batch, embeddings)
            else:
                # Subsequent batches: add to existing store
                vector_store.add_documents(batch)

            time.sleep(0.5)  # Small delay to avoid rate limiting

    elif vector_backend.lower() == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except Exception as e:
            raise RuntimeError("FAISS requested but not available") from e

        # Sanitize all text content
        for doc in documents:
            doc.page_content = sanitize_text(doc.page_content or "")
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    doc.metadata[key] = sanitize_text(value)

        # FAISS also needs batching
        batch_size = 500
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"→ Processing {len(documents)} documents in {total_batches} batches of {batch_size}...")

        vector_store = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            print(f"  → Embedding batch {batch_num}/{total_batches} ({len(batch)} documents)...")

            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                batch_store = FAISS.from_documents(batch, embeddings)
                vector_store.merge_from(batch_store)

            time.sleep(0.5)

    else:
        raise ValueError("vector_backend must be 'chroma' or 'faiss'")

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": k})
    print("→ RAG KB ready (dense retriever over multi-scale segments).")
    return vector_retriever


# --------------------------------------------------------------------------------------
# PUBLIC API: BUILD RETRIEVER FROM WEB
# --------------------------------------------------------------------------------------
def build_retriever_from_web(
    polymer_keywords: Optional[List[str]] = None,
    target_curated: int = TARGET_CURATED,
    target_journals: int = TARGET_JOURNALS,
    target_arxiv: int = TARGET_ARXIV,
    target_openalex: int = TARGET_OPENALEX,
    target_epmc: int = TARGET_EPMC,
    extra_pdf_urls: Optional[List[str]] = None,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    tmp_download_dir: str = DEFAULT_TMP_DOWNLOAD_DIR,
    k: int = 10,
    embedding_model: str = "text-embedding-3-small",
    vector_backend: str = "chroma",
    mailto: Optional[str] = None,
    include_curated: bool = True,
):
    """
    Fetch balanced polymer corpus across multiple sources.

    Target distribution (~2000 PDFs):
    - Curated guidelines/standards: 100
    - Polymer journals OA: 200
    - arXiv: 800
    - OpenAlex: 600
    - Europe PMC: 200
    - Extra/databases: 100
    """
    polymer_keywords = sorted(set(polymer_keywords or POLYMER_KEYWORDS), key=str.lower)
    print("=" * 70)
    print("Fetching polymer PDFs from balanced sources...")
    print(
        f"Target: {target_curated} curated + {target_journals} journals + "
        f"{target_arxiv} arXiv + {target_openalex} OpenAlex + {target_epmc} EPMC"
    )

    ensure_dir(tmp_download_dir)
    manifest = load_manifest(tmp_download_dir)
    source_stats = defaultdict(int)
    all_paths: List[str] = []

    # --------------------------------------------------------------------------------------
    # 1) Curated sources (IUPAC, ISO/ASTM, polymer informatics reviews)
    # --------------------------------------------------------------------------------------
    if include_curated and CURATED_POLYMER_PDF_SOURCES:
        print(f"[1/6] Downloading {len(CURATED_POLYMER_PDF_SOURCES)} curated PDFs...")
        curated_paths = parallel_download_pdfs(
            CURATED_POLYMER_PDF_SOURCES[:target_curated],
            tmp_download_dir,
            manifest,
            max_workers=4,
            desc="Curated PDFs",
        )
        for p in curated_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["curated"] += 1
        print(f"  → {len(curated_paths)} curated PDFs downloaded")

    # --------------------------------------------------------------------------------------
    # 2) Polymer journals OA
    # --------------------------------------------------------------------------------------
    try:
        print(f"[2/6] Fetching polymer journal PDFs (target: {target_journals})...")
        journal_paths = fetch_polymer_journal_pdfs(
            POLYMER_JOURNAL_QUERIES,
            tmp_download_dir,
            manifest,
            max_per_journal=target_journals // len(POLYMER_JOURNAL_QUERIES) + 1,
            mailto=mailto,
        )
        for p in journal_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["journal"] += 1
        print(f"  → {len(journal_paths)} journal PDFs downloaded")
    except Exception as e:
        print(f"[WARN] Polymer journal fetch error: {e}")

    # --------------------------------------------------------------------------------------
    # 3) arXiv polymer-focused categories
    # --------------------------------------------------------------------------------------
    try:
        print(f"[3/6] Fetching arXiv PDFs (target: {target_arxiv})...")
        arxiv_paths = fetch_arxiv_pdfs(
            polymer_keywords, tmp_download_dir, manifest, max_results=target_arxiv
        )
        for p in arxiv_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["arxiv"] += 1
        print(f"  → {len(arxiv_paths)} arXiv PDFs downloaded")
    except Exception as e:
        print(f"[WARN] arXiv fetch error: {e}")

    # --------------------------------------------------------------------------------------
    # 4) OpenAlex broad polymer search
    # --------------------------------------------------------------------------------------
    try:
        print(f"[4/6] Fetching OpenAlex PDFs (target: {target_openalex})...")
        openalex_paths = fetch_openalex_pdfs(
            polymer_keywords,
            tmp_download_dir,
            manifest,
            max_results=target_openalex,
            mailto=mailto,
        )
        for p in openalex_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["openalex"] += 1
        print(f"  → {len(openalex_paths)} OpenAlex PDFs downloaded")
    except Exception as e:
        print(f"[WARN] OpenAlex fetch error: {e}")

    # --------------------------------------------------------------------------------------
    # 5) Europe PMC biopolymers/materials
    # --------------------------------------------------------------------------------------
    try:
        print(f"[5/6] Fetching Europe PMC PDFs (target: {target_epmc})...")
        epmc_paths = fetch_epmc_pdfs(
            polymer_keywords, tmp_download_dir, manifest, max_results=target_epmc
        )
        for p in epmc_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["epmc"] += 1
        print(f"  → {len(epmc_paths)} Europe PMC PDFs downloaded")
    except Exception as e:
        print(f"[WARN] Europe PMC fetch error: {e}")

    # --------------------------------------------------------------------------------------
    # 6) Extra URLs (user-provided, database exports, etc.)
    # --------------------------------------------------------------------------------------
    if extra_pdf_urls:
        print(f"[6/6] Downloading {len(extra_pdf_urls)} extra PDFs...")
        extra_entries = [
            {"url": u, "name": None, "meta": {"url": u, "source": "extra"}}
            for u in extra_pdf_urls
        ]
        extra_paths = parallel_download_pdfs(
            extra_entries, tmp_download_dir, manifest, max_workers=8, desc="Extra PDFs"
        )
        for p in extra_paths:
            if p not in all_paths:
                all_paths.append(p)
                source_stats["extra"] += 1
        print(f"  → {len(extra_paths)} extra PDFs downloaded")

    # --------------------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------------------
    total = len(all_paths)
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Total unique PDFs downloaded: {total}")
    print(" by source:")
    for source, count in sorted(source_stats.items()):
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {source:20s} {count:4d} PDFs ({pct:5.1f}%)")
    print("=" * 70)

    if total == 0:
        raise RuntimeError(
            "No PDFs fetched. Adjust keywords, targets, or add extra_pdf_urls."
        )

    print("Building knowledge base from downloaded PDFs...")
    retriever = _split_and_build_retriever(
        documents_dir=tmp_download_dir,
        persist_dir=persist_dir,
        k=k,
        embedding_model=embedding_model,
        vector_backend=vector_backend,
    )

    return retriever


# --------------------------------------------------------------------------------------
# PUBLIC API: BUILD RETRIEVER FROM LOCAL PAPERS
# --------------------------------------------------------------------------------------
def build_retriever(
    papers_path: str,
    persist_dir: Optional[str] = DEFAULT_PERSIST_DIR,
    k: int = 10,
    embedding_model: str = "text-embedding-3-small",
    vector_backend: str = "chroma",
):
    """
    Build polymer RAG KB from local PDFs.
    """
    print("Building RAG knowledge base from local PDFs...")
    return _split_and_build_retriever(
        documents_dir=papers_path,
        persist_dir=persist_dir,
        k=k,
        embedding_model=embedding_model,
        vector_backend=vector_backend,
    )


# --------------------------------------------------------------------------------------
# CONVENIENCE WRAPPER: POLYMER FOUNDATION MODELS
# --------------------------------------------------------------------------------------
def build_retriever_polymer_foundation_models(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    k: int = 10,
    vector_backend: str = "chroma",
):
    """
    Convenience wrapper for polymer foundation model corpus.
    """
    fm_kw = list(
        set(POLYMER_KEYWORDS)
        | {
            "BigSMILES",
            "PSMILES",
            "polymer SMILES",
            "polymer language model",
            "foundation model polymer",
            "masked language model polymer",
            "self-supervised polymer",
            "generative polymer",
            "polymer sequence modeling",
            "representation learning polymer",
        }
    )
    return build_retriever_from_web(
        polymer_keywords=fm_kw,
        target_curated=100,
        target_journals=200,
        target_arxiv=800,
        target_openalex=600,
        target_epmc=200,
        persist_dir=persist_dir,
        k=k,
        embedding_model="text-embedding-3-small",
        vector_backend=vector_backend,
    )


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    retriever = build_retriever_from_web(
        polymer_keywords=POLYMER_KEYWORDS,
        target_curated=100,
        target_journals=200,
        target_arxiv=800,
        target_openalex=600,
        target_epmc=200,
        persist_dir="chroma_polymer_db_balanced",
        tmp_download_dir=DEFAULT_TMP_DOWNLOAD_DIR,
        k=10,
        embedding_model="text-embedding-3-small",
        vector_backend="chroma",
        mailto=DEFAULT_MAILTO,
        include_curated=True,
    )

    print("\n" + "=" * 70)
    print("Testing retrieval with sample query")
    docs = retriever.get_relevant_documents("PSMILES polymer electrolyte design")
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        title = meta.get("title") or os.path.basename(meta.get("source", "")) or "document"
        year = meta.get("year", "")
        src = meta.get("source", "unknown")
        journal = meta.get("journal", "")
        scale = meta.get("segment_scale", "")
        source_str = f"{src}"
        if journal:
            source_str = f"{journal} ({src})"
        print(f"\n[{i}] {title}")
        print(f"    Year: {year} | Source: {source_str} | Scale: {scale}")
        print(f"    Content: {(d.page_content or '')[:200]}...")
