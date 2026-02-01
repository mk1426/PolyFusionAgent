# PolyFusionAgent: a multimodal foundation model and an autonomous AI assistant for polymer informatics

**PolyFusionAgent** is an interactive framework that couples a **multimodal polymer foundation model (PolyFusion)** with a **tool-augmented, literature-grounded design agent (PolyAgent)** for polymer property prediction, inverse design, and evidence-linked scientific reasoning.

> **PolyFusion** aligns complementary polymer views—**PSMILES sequence**, **2D topology**, **3D structural proxies**, and **chemical fingerprints**—into a shared latent space that transfers across chemistries and data regimes.  
> **PolyAgent** closes the design loop by connecting **prediction + generation + retrieval + visualization** so recommendations are contextualized with explicit supporting precedent.

---

## Authors & Affiliation

**Manpreet Kaur**¹, **Qian Liu**¹\*  

¹ Department of Applied Computer Science, The University of Winnipeg, Winnipeg, MB, Canada

### Contact
- **Qian Liu** — qi.liu@uwinnipeg.ca

---

## Abstract

Polymers underpin technologies from energy storage to biomedicine, yet discovery remains constrained by an astronomically large design space and fragmented representations of polymer structure, properties, and prior knowledge. Although machine learning has advanced property prediction and candidate generation, most models remain disconnected from the physical and experimental context needed for actionable materials design.  

Here we introduce **PolyFusionAgent**, an interactive framework that couples a multimodal polymer foundation model (**PolyFusion**) with a tool-augmented, literature-grounded design agent (**PolyAgent**). PolyFusion aligns complementary polymer views—sequence, topology, three-dimensional structural proxies, and chemical fingerprints—across millions of polymers to learn a shared latent space that transfers across chemistries and data regimes. Using this unified representation, PolyFusion improves prediction of key thermophysical properties and enables property-conditioned generation of chemically valid, structurally novel polymers that extend beyond the reference design space.  

PolyAgent closes the design loop by coupling prediction and inverse design to evidence retrieval from the polymer literature, so that hypotheses are proposed, evaluated, and contextualized with explicit supporting precedent in a single workflow. Together, **PolyFusionAgent** establishes a route toward interactive, evidence-linked polymer discovery that combines large-scale representation learning, multimodal chemical knowledge, and verifiable scientific reasoning.

---

<p align="center">
  <img src="assets/PP1.png" alt="PolyFusionAgent Overview" width="800" height="1000"/>
</p>

## Contents

- [1. Repository Overview](#1-repository-overview)
- [2. Dependencies & Environment](#2-dependencies--environment)
  - [2.1 Installation](#21-installation)
  - [2.2 Optional Chemistry & GPU Notes](#22-optional-chemistry--gpu-notes)
- [3. Data, Modalities, and Preprocessing](#3-data-modalities-and-preprocessing)
  - [3.1 Input CSV schema](#31-input-csv-schema)
  - [3.2 Generate multimodal columns (graph/geometry/fingerprints)](#32-generate-multimodal-columns-graphgeometryfingerprints)
  - [3.3 What “graph”, “geometry”, and “fingerprints” look like](#33-what-graph-geometry-and-fingerprints-look-like)
- [4. Models & Artifacts](#4-models--artifacts)
- [5. Running the Code](#5-running-the-code)
  - [5.1 Multimodal contrastive pretraining (PolyFusion)](#51-multimodal-contrastive-pretraining-polyfusion)
  - [5.2 Downstream property prediction](#52-downstream-property-prediction)
  - [5.3 Inverse design / polymer generation](#53-inverse-design--polymer-generation)
  - [5.4 PolyAgent (Gradio UI)](#54-polyagent-gradio-ui)
- [6. Results & Reproducibility](#6-results--reproducibility)
- [7. Citation](#7-citation)

---

## 1. Repository Overview

### PolyFusionAgent has three tightly coupled layers:  
**(i) PolyFusion** learns a transferable multimodal embedding space; **(ii) task heads** perform property prediction and property-conditioned generation using that embedding; and **(iii) PolyAgent** orchestrates tools (prediction, generation, retrieval, visualization) to produce grounded, audit-ready design outputs.
---

### A. PolyFusion — multimodal polymer foundation model (FM) 
**Modalities + encoders:** 
- **PSMILES (D)** → DeBERTaV2-style encoder (`PolyFusion/DeBERTav2.py`)  
- **2D molecular graph (G)** → **GINE** (Graph Isomorphism Network with Edge features) (`PolyFusion/GINE.py`)  
- **3D geometry proxy (S)** → **SchNet** (continuous-filter network for 3D structures) (`PolyFusion/SchNet.py`) 
- **Fingerprints (T)** → Transformer encoder (`PolyFusion/Transformer.py`)

**Pretraining objective:**  
PolyFusion forms a **fused structural anchor** from (D, G, S) and contrastively aligns it to the **fingerprint target** (T) using an **InfoNCE** loss over a cross-similarity matrix. (`PolyFusion/CL.py`) 

Key entrypoint:
- `PolyFusion/CL.py` — multimodal contrastive pretraining (anchor–target InfoNCE)
---

### B. Downstream tasks — prediction + inverse design
These scripts adapt PolyFusion embeddings for two core tasks:

- **Property prediction (structure → properties)**  
  `Downstream Tasks/Property_Prediction.py`  
  Trains lightweight regression heads on top of (typically frozen) PolyFusion embeddings for thermophysical properties (e.g., density (ρ), glass transition temperature (Tg), melting temperature (Tm), and thermal decomposition temperature (Td)). 

- **Inverse design / generation (target properties → candidate polymers)**  
  `Downstream Tasks/Polymer_Generation.py`  
  Performs property-conditioned generation using PolyFusion embeddings as the conditioning interface with a pretrained SELFIES-based encoder–decoder (SELFIES-TED) and latent guidance.
---

### C. PolyAgent — tool-augmented AI assistant (controller + tools)

**Goal:** convert open-ended polymer design prompts into **grounded, constraint-consistent, evidence-linked** outputs by coupling PolyFusion with tool-mediated verification and retrieval.

**What PolyAgent does (system-level):**  
- decomposes a user request into typed sub-tasks  
- calls tools for **prediction**, **generation**, **retrieval (local RAG + web)**, and **visualization**  
- returns a final response with explicit evidence/citations and an experiment-ready validation plan

Main files:
- `PolyAgent/orchestrator.py` — planning + tool routing (controller)
- `PolyAgent/rag_pipeline.py` — local retrieval utilities (PDF → chunks → embeddings → vector store)
- `PolyAgent/gradio_interface.py` — Gradio UI entrypoint

D. Datasets
Data

This repo is designed to work with large-scale pretraining corpora (for PolyFusion) plus experiment-backed downstream sets (for finetuning/evaluation). It does not redistribute these datasets—please download them from the original sources and follow their licenses/terms.

Pretraining corpora (examples used in the paper)

PI1M: “PI1M: A Benchmark Database for Polymer Informatics.”

DOI page: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726

(Often mirrored/linked via PubMed)

polyOne: “polyOne Data Set – 100 million hypothetical polymers …” (Zenodo record).

Zenodo: https://zenodo.org/records/7766806

Downstream / evaluation data (example)

PoLyInfo (NIMS Polymer Database) provides experimental/literature polymer properties and metadata.

Main site: https://polymer.nims.go.jp/en/

Overview/help: https://polymer.nims.go.jp/PoLyInfo/guide/en/what_is_polyinfo.html

Tip: For reproducibility, document: export query, filtering rules, property units/conditions, and train/val/test splits in data/README.md.

2. Dependencies & Environment

PolyFusionAgent spans three compute modes:

Data preprocessing (RDKit-heavy; CPU-friendly but parallelizable)

Model training/inference (PyTorch; GPU strongly recommended for PolyFusion pretraining)

PolyAgent runtime (Gradio UI + retrieval stack; GPU optional but helpful for throughput)

2.1 Supported platforms

OS: Linux recommended (Ubuntu 20.04/22.04 tested most commonly in similar stacks), macOS/Windows supported for lightweight inference but may require extra care for RDKit/FAISS.

Python: 3.9–3.11 recommended (keep Python/PyTorch/CUDA consistent for reproducibility).

GPU: NVIDIA recommended for training. Manuscript pretraining used mixed precision and ran on NVIDIA A100 GPUs 

 

.

2.2 Installation (base)
git clone https://github.com/manpreet88/PolyFusionAgent.git
cd PolyFusionAgent

# Option A: venv
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Option B: conda (recommended if you use RDKit/FAISS)
# conda create -n polyfusion python=3.10 -y
# conda activate polyfusion

pip install -r requirements.txt


Tip (recommended): split installs by “extras” so users don’t pull GPU/RAG dependencies unless needed.

requirements.txt → core + inference

requirements-train.txt → training + distributed / acceleration

requirements-agent.txt → gradio + retrieval + PDF tooling

(If you keep a single requirements file, clearly label optional dependencies as such.)

2.3 Core ML stack (PolyFusion / downstream)

Required

torch (GPU build strongly recommended for training)

numpy, pandas, scikit-learn (downstream regression uses standard scaling + CV; manuscript uses 5-fold CV 

 

)

transformers (PSMILES encoder + assorted NLP utilities)

Recommended

accelerate (multi-GPU / fp16 ergonomics)

sentencepiece (PSMILES tokenization uses SentencePiece with a fixed 265-token vocab 

 

)

tqdm, rich (logging)

GPU check

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| torch:', torch.__version__, '| cuda_ver:', torch.version.cuda)"

2.4 Chemistry stack (strongly recommended)

A large fraction of the pipeline depends on RDKit:

building graphs / fingerprints

conformer generation

canonicalization + validity checks

PolyAgent visualization

Install RDKit via conda-forge:

conda install -c conda-forge rdkit -y


Wildcard endpoint handling (important):
For RDKit-derived modalities, the pipeline converts polymer repeat units into a pseudo-molecule by replacing the repeat-unit wildcard attachment token [*] with [At] (Astatine) to ensure chemical sanitization and tool compatibility 

 

.

2.5 Graph / 3D stacks (optional, depending on your implementation)

If your GINE implementation uses PyTorch Geometric, install the wheels that match your exact PyTorch + CUDA combination.

PyG install instructions differ by CUDA version; pin your environment carefully.

If you use SchNet via a third-party implementation, confirm the dependency (e.g., schnetpack, torchmd-net, or a local SchNet module). In the manuscript, SchNet uses a neighbor list with radial cutoff 10 Å and ≤64 neighbors/atom, with 6 interaction layers and hidden size 600 

 

 

 

.

2.6 Retrieval stack (PolyAgent)

PolyAgent combines:

Local RAG over PDFs (chunking + embeddings + vector index)

Web augmentation (optional)

Reranking (cross-encoder)

In the manuscript implementation, the local knowledge base is constructed from 1108 PDFs, chunked at 512/256/128 tokens with overlaps 64/48/32, embedded with OpenAI text-embedding-3-small (1536-d), and indexed using FAISS HNSW (M=64, efconstruction=200) 

 

. Retrieved chunks are reranked with ms-marco-MiniLM-L-12-v2 

 

.

Typical dependencies

gradio

faiss-cpu (or faiss-gpu if desired)

pypdf / pdfminer.six (PDF text extraction)

tiktoken (chunking tokens; manuscript references TikToken cl100k 

 

)

trafilatura (web page extraction; used in manuscript web augmentation 

 

)

transformers (reranker and query rewrite model; manuscript uses T5 for rewriting in web augmentation 

 

)

2.7 Environment variables

PolyAgent is a tool-orchestrated system. At minimum, set:

export OPENAI_API_KEY="YOUR_KEY"


Optional (if your configs support them):

export OPENAI_MODEL="gpt-4.1"     # controller model (manuscript uses GPT-4.1) :contentReference[oaicite:11]{index=11}
export HF_TOKEN="YOUR_HF_TOKEN"   # to pull hosted weights/tokenizers if applicable


Recommended .env pattern
Create a .env (do not commit) and load it in the Gradio entrypoint:

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1

3. Data, Modalities, and Preprocessing
3.1 Datasets (what the manuscript uses)

Pretraining uses PI1M + polyOne, at two scales: 2M and 5M polymers 

 

.

Downstream fine-tuning / evaluation uses PolyInfo (≈ 1.8×10⁴ experimental polymers) 

 

.

PolyInfo is held out from pretraining 

 

.

Where are the links? The uploaded manuscript describes these datasets but does not include canonical URLs in the excerpted sections available here. Add the official dataset links in this README once you finalize where you host or reference them.

3.2 Minimum CSV schema

Your raw CSV must include:

psmiles (required) — polymer repeat unit string with [*] endpoints

Optional:

source — dataset tag (PI1M/polyOne/PolyInfo/custom)

property columns — e.g., density, Tg, Tm, Td (names can be mapped)

Example:

psmiles,source,density,Tg,Tm,Td
[*]CC(=O)OCCO[*],PolyInfo,1.21,55,155,350


Endpoint note: when generating RDKit-dependent modalities, the code may internally replace [*] with [At] to sanitize repeat-unit molecules 

 

.

3.3 Modalities produced per polymer

PolyFusion represents each polymer using four complementary modalities 

 

:

PSMILES sequences (D)

SentencePiece tokenization with fixed vocab size 265 (kept fixed during downstream) 

 

2D molecular graph (G)

nodes = atoms, edges = bonds, with chemically meaningful node/edge features

3D conformational proxy (S)

conformer embedding + optimization pipeline (ETKDG/UFF described in Methods)

SchNet neighbor cutoff and layer specs given in Supplementary 

 

 

 

Fingerprints (T)

ECFP6 (radius r=3) with 2048 bits 

 

3.4 Preprocessing script

Use your preprocessing utility (e.g., Data_Modalities.py) to append multimodal columns:

python Data_Modalities.py \
  --csv_file /path/to/polymers.csv \
  --chunk_size 1000 \
  --num_workers 24


Expected outputs:

*_processed.csv with new columns: graph, geometry, fingerprints (as JSON blobs)

*_failures.jsonl for failed rows (index + error)

4. Models & Artifacts

This repository typically produces three artifact families:

4.1 PolyFusion checkpoints (pretraining)

PolyFusion maps each modality into a shared embedding space of dimension d=600 

 

.
Pretraining uses:

unified masking with pmask = 0.15 and an 80/10/10 corruption rule 

 

anchor–target contrastive learning where the fused structural anchor is aligned to the fingerprint target (InfoNCE with τ = 0.07) 

 

Store:

encoder weights per modality

projection heads

training config + tokenizer artifacts (SentencePiece model)

4.2 Downstream predictors (property regression)

Downstream uses:

fused 600-d embedding

a lightweight regressor (2-layer MLP, hidden width 300, dropout 0.1) 

 

Training protocol:

5-fold CV, inner validation (10%) with early stopping 

 

Save:

best weights per property per fold

scalers used for standardization

4.3 Inverse design generator (SELFIES-TED conditioning)

Inverse design conditions a SELFIES-based encoder–decoder (SELFIES-TED) on PolyFusion’s 600-d embedding 

 

.
Implementation details from the manuscript include:

conditioning via K=4 learned memory tokens 

 

training-time latent noise σtrain = 0.10 

 

decoding uses top-p (0.92), temperature 1.0, repetition penalty 1.05, max length 256 

 

property targeting via generate-then-filter using a GP oracle and acceptance threshold τs = 0.5 (standardized units) 

 

Save:

decoder weights + conditioning projection

tokenization assets (if applicable)

property oracle artifacts (GP models / scalers)

5. Running the Code

Several scripts may contain path placeholders. Centralize them into one config file (recommended) or update the constants in each entrypoint.

5.1 Multimodal contrastive pretraining (PolyFusion)

Entrypoint:

PolyFusion/CL.py

Manuscript-grounded defaults:

AdamW, lr=1e-4, weight_decay=1e-2, batch=16, grad accum=4 (effective 64), up to 25 epochs, early stopping patience 10, FP16 

 

Run:

python PolyFusion/CL.py


Sanity tip: start with a smaller subset (e.g., 50k–200k rows) to validate preprocessing + training stability before scaling to millions.

5.2 Downstream property prediction

Entrypoint:

Downstream Tasks/Property_Prediction.py

What it does:

loads a modality-augmented CSV

loads pretrained PolyFusion weights

trains property heads with K-fold CV

Run:

python "Downstream Tasks/Property_Prediction.py"

5.3 Inverse design / polymer generation

Entrypoint:

Downstream Tasks/Polymer_Generation.py

What it does:

conditions SELFIES-TED on PolyFusion embeddings

generates candidates and filters to target using the manuscript-style oracle loop 

 

Run:

python "Downstream Tasks/Polymer_Generation.py"

5.4 PolyAgent (Gradio UI)

Core components:

PolyAgent/orchestrator.py (controller + tool router)

PolyAgent/rag_pipeline.py (local RAG)

PolyAgent/gradio_interface.py (UI)

Manuscript controller:

GPT-4.1 controller with planning temperature τplan=0.2 

 

Run:

cd PolyAgent
python gradio_interface.py --server-name 0.0.0.0 --server-port 7860

6. Results & Reproducibility
6.1 What “reproducible” means in this repo

To help others reproduce your paper-level results:

Pin versions: Python, PyTorch, CUDA, RDKit, FAISS, Transformers

Fix seeds across Python/NumPy/Torch

Log configs per run (JSON/YAML dumped beside checkpoints)

Record dataset snapshots (hashes of CSVs and modality JSON columns)

6.2 Manuscript training protocol highlights

PolyFusion shared latent dimension: 600 

 

Unified corruption: pmask = 0.15, 80/10/10 rule 

 

Contrastive alignment uses InfoNCE with τ = 0.07 

 

Pretraining optimization and schedule: AdamW, lr 1e-4, wd 1e-2, eff batch 64, FP16, early stopping 

 

PolyAgent retrieval index: 1108 PDFs; chunking and FAISS HNSW params as described 

 

7. Citation

If you use this repository in your work, please cite the accompanying manuscript:

@article{kaur2026polyfusionagent,
  title   = {PolyFusionAgent: a multimodal foundation model and autonomous AI assistant for polymer informatics},
  author  = {Kaur, Manpreet and Liu, Qian},
  year    = {2026},
  note    = {Manuscript / preprint}
}
PI1M (JCIM): https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726

polyOne (Zenodo): https://zenodo.org/records/7766806

PoLyInfo (NIMS): https://polymer.nims.go.jp/en/
