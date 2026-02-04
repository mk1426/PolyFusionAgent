# PolyFusionAgent: A Multimodal Foundation Model and Autonomous AI Assistant for Polymer Property Prediction and Inverse Design

**PolyFusionAgent** is an interactive framework that couples a **multimodal polymer foundation model (PolyFusion)** with a **tool-augmented, literature-grounded design agent (PolyAgent)** for polymer property prediction, inverse design, and evidence-linked scientific reasoning.

> **PolyFusion** aligns complementary polymer views—**PSMILES sequence**, **2D topology**, **3D structural proxies**, and **chemical fingerprints**—into a shared latent space that transfers across chemistries and data regimes.  
> **PolyAgent** closes the design loop by connecting **prediction + generation + retrieval + visualization** so recommendations are contextualized with explicit supporting precedent.

---

## Authors & Affiliation

**Manpreet Kaur**¹, **Qian Liu**¹*  
¹ Department of Applied Computer Science, The University of Winnipeg, Winnipeg, MB, Canada

### Contact
- **Qian Liu** — qi.liu@uwinnipeg.ca

---

## Links 
- **Live Space:** [kaurm43/PolyFusionAgent](https://huggingface.co/spaces/kaurm43/PolyFusionAgent)
- **Weights repo:** [kaurm43/polyfusionagent-weights](https://huggingface.co/kaurm43/polyfusionagent-weights)
- **Weights file browser:** [weights/tree/main](https://huggingface.co/kaurm43/polyfusionagent-weights/tree/main)

---

## Abstract

Polymers underpin technologies from energy storage to biomedicine, yet discovery remains constrained by an astronomically large design space and fragmented representations of polymer structure, properties, and prior knowledge. Although machine learning has advanced property prediction and candidate generation, most models remain disconnected from the physical and experimental context needed for actionable materials design.

Here we introduce **PolyFusionAgent**, an interactive framework that couples a multimodal polymer foundation model (**PolyFusion**) with a tool-augmented, literature-grounded design agent (**PolyAgent**). PolyFusion aligns complementary polymer views—sequence, topology, three-dimensional structural proxies, and chemical fingerprints—across millions of polymers to learn a shared latent space that transfers across chemistries and data regimes. Using this unified representation, PolyFusion improves prediction of key thermophysical properties and enables property-conditioned generation of chemically valid, structurally novel polymers that extend beyond the reference design space.

PolyAgent closes the design loop by coupling prediction and inverse design to evidence retrieval from the polymer literature, so that hypotheses are proposed, evaluated, and contextualized with explicit supporting precedent in a single workflow. Together, **PolyFusionAgent** establishes a route toward interactive, evidence-linked polymer discovery that combines large-scale representation learning, multimodal chemical knowledge, and verifiable scientific reasoning.

---

<p align="center">
  <img src="assets/PP1.png" alt="PolyFusionAgent Overview" width="800" height="1000"/>
</p>

---

## Contents

- [1. Repository Overview](#1-repository-overview)
  - [A. PolyFusion — multimodal polymer foundation model (FM)](#a-polyfusion--multimodal-polymer-foundation-model-fm)
  - [B. Downstream tasks — prediction + inverse design](#b-downstream-tasks--prediction--inverse-design)
  - [C. PolyAgent — tool-augmented AI assistant (controller + tools)](#c-polyagent--tool-augmented-ai-assistant-controller--tools)
- [2. Dependencies & Environment](#2-dependencies--environment)
  - [2.1 Supported platforms](#21-supported-platforms)
  - [2.2 Installation (base)](#22-installation-base)
  - [2.3 Core ML stack (Pretraining / Downstream)](#23-core-ml-stack-pretraining--downstream)
  - [2.4 Chemistry stack (Strongly Recommended)](#24-chemistry-stack-strongly-recommended)
  - [2.5 Retrieval Stack (PolyAgent)](#25-retrieval-stack-polyagent)
  - [2.6 Environment variables](#26-environment-variables)
- [3. Data, Modalities, and Preprocessing](#3-data-modalities-and-preprocessing)
  - [3.1 Datasets](#31-datasets)
  - [3.2 Minimum CSV schema](#32-minimum-csv-schema)
  - [3.3 Modalities produced per polymer](#33-modalities-produced-per-polymer)
  - [3.4 Preprocessing script](#34-preprocessing-script)
- [4. Models & Artifacts](#4-models--artifacts)
  - [4.1 PolyFusion Checkpoints (pretraining)](#41-polyfusion-checkpoints-pretraining)
  - [4.2 Downstream Predictors (property regression)](#42-downstream-predictors-property-regression)
  - [4.3 Inverse Design Generator (SELFIES-TED conditioning)](#43-inverse-design-generator-selfies-ted-conditioning)
  - [4.4 PolyAgent (Gradio UI)](#44-polyagent-gradio-ui)
- [5.Pretraining Weights / PolyAgent UI](#5-pretraining-weights-polyagent-ui)
---

## 1. Repository Overview

PolyFusionAgent has three tightly coupled layers:
- **(i) PolyFusion** learns a transferable multimodal embedding space.
- **(ii) Task heads** perform property prediction and property-conditioned generation using that embedding.
- **(iii) PolyAgent** orchestrates tools (prediction, generation, retrieval, visualization) to produce grounded, audit-ready design outputs.

### A. PolyFusion — multimodal polymer foundation model (FM)

**Modalities + encoders:**
- **PSMILES (D)** → DeBERTaV2-style encoder (`PolyFusion/DeBERTav2.py`)
- **2D molecular graph (G)** → **GINE** (Graph Isomorphism Network with Edge features) (`PolyFusion/GINE.py`)
- **3D geometry proxy (S)** → **SchNet** (continuous-filter network for 3D structures) (`PolyFusion/SchNet.py`)
- **Fingerprints (T)** → Transformer encoder (`PolyFusion/Transformer.py`)

**Pretraining objective:**  
PolyFusion forms a **fused structural anchor** from (D, G, S) and contrastively aligns it to the **fingerprint target** (T) using an **InfoNCE** loss over a cross-similarity matrix (`PolyFusion/CL.py`).

**Key entrypoint:**
- `PolyFusion/CL.py` — multimodal contrastive pretraining (anchor–target InfoNCE)

### B. Downstream tasks — prediction + inverse design

These scripts adapt PolyFusion embeddings for two core tasks:

- **Property prediction (structure → properties)**  
  `Downstream Tasks/Property_Prediction.py`  
  Trains lightweight regression heads on top of (typically frozen) PolyFusion embeddings for thermophysical properties (e.g., density (ρ), glass transition temperature (Tg), melting temperature (Tm), and thermal decomposition temperature (Td)).

- **Inverse design / generation (target properties → candidate polymers)**  
  `Downstream Tasks/Polymer_Generation.py`  
  Performs property-conditioned generation using PolyFusion embeddings as the conditioning interface with a pretrained SELFIES-based encoder–decoder (SELFIES-TED) and latent guidance.

### C. PolyAgent — tool-augmented AI assistant (controller + tools)

**Goal:** Convert open-ended polymer design prompts into **grounded, constraint-consistent, evidence-linked** outputs by coupling PolyFusion with tool-mediated verification and retrieval.

**What PolyAgent does (system-level):**
- Decomposes a user request into typed sub-tasks.
- Calls tools for **prediction**, **generation**, **retrieval (local RAG + web)**, and **visualization**.
- Returns a final response with explicit evidence/citations and an experiment-ready validation plan.

**Main files:**
- `PolyAgent/orchestrator.py` — planning + tool routing (controller)
- `PolyAgent/rag_pipeline.py` — local retrieval utilities (PDF → chunks → embeddings → vector store)
- `PolyAgent/gradio_interface.py` — Gradio UI entrypoint

---

## 2. Dependencies & Environment

PolyFusionAgent spans three compute modes:
- **Data preprocessing** (RDKit-heavy; CPU-friendly but parallelizable)
- **Model training/inference** (PyTorch; GPU strongly recommended for PolyFusion pretraining)
- **PolyAgent runtime** (Gradio UI + retrieval stack; GPU optional but helpful for throughput)

### 2.1 Supported platforms

- **Python:** 3.9–3.11 recommended (keep Python/PyTorch/CUDA consistent for reproducibility).
- **GPU:** NVIDIA recommended for training. Manuscript pretraining used mixed precision and ran on NVIDIA A100 GPUs.

### 2.2 Installation (base)

```bash
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
```

### 2.3 Core ML stack (Pretraining / Downstream)

**Required:**
- torch (GPU build strongly recommended for training)
- numpy, pandas, scikit-learn (downstream regression uses standard scaling + CV; manuscript uses 5-fold CV)
- transformers (PSMILES encoder + assorted NLP utilities)

**Recommended:**
- accelerate (fp16 ergonomics)
- sentencepiece (PSMILES tokenization uses SentencePiece with a 265-token vocab)
- tqdm, rich (logging)

**GPU Check:**
```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), '| torch:', torch.__version__, '| cuda_ver:', torch.version.cuda)"
```

### 2.4 Chemistry stack (Strongly Recommended)

A large fraction of the pipeline depends on RDKit:
- building graphs / fingerprints
- conformer generation
- canonicalization + validity checks
- PolyAgent visualization

**RDKit Installation**
```bash
# Conda (recommended)
conda install -c conda-forge rdkit -y

# OR: pip (use a fresh venv; best effort)
python -m pip install -U pip
python -m pip install rdkit

# If rdkit isn't available/works poorly on your platform via pip:
# use the community wheels instead (common fallback):
python -m pip install rdkit-pypi
```

**Wildcard endpoint (*) handling:**  
For RDKit-derived modalities, the pipeline converts polymer repeat units into a pseudo-molecule by replacing the repeat-unit wildcard attachment token `[*]` with `[At]` (Astatine) to ensure chemical sanitization and tool compatibility.

### 2.5 Retrieval Stack (PolyAgent)

PolyAgent combines:
- Local RAG over PDFs (chunking + embeddings + vector index)
- Web augmentation 
- Reranking (cross-encoder)

In the manuscript implementation, the local knowledge base is constructed from 1108 PDFs, chunked at 512/256/128 tokens with overlaps 64/48/32, embedded with OpenAI text-embedding-3-small (1536-d), and indexed using FAISS HNSW (M=64, efconstruction=200). Retrieved chunks are reranked with ms-marco-MiniLM-L-12-v2.

**Typical dependencies:**
- gradio
- faiss-cpu (or faiss-gpu if desired)
- pypdf / pdfminer.six (PDF text extraction)
- tiktoken (chunking tokens; manuscript references TikToken cl100k)
- trafilatura (web page extraction)
- transformers (reranker and query rewrite model; manuscript uses T5 for rewriting in web augmentation)

### 2.6 Environment variables

PolyAgent is a tool-orchestrated system. At minimum, set:
```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Additionally:
```bash
export OPENAI_MODEL="gpt-4.1"     # controller model (manuscript uses GPT-4.1)
export HF_TOKEN="YOUR_HF_TOKEN"   # Baseline LLMs
```

---

## 3. Data, Modalities, and Preprocessing

### 3.1 Datasets
This repo is designed to work with large-scale pretraining corpora (for PolyFusion at two scales: 2M and 5M) plus experiment-backed downstream sets (for finetuning/evaluation). It does not redistribute these datasets—please download them from the original sources and follow their licenses/terms.

**Pretraining corpora:**
- **PI1M:** “PI1M: A Benchmark Database for Polymer Informatics.”  
  DOI page: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00726  
  (Often mirrored/linked via PubMed)
- **polyOne:** “polyOne Data Set – 100 million hypothetical polymers …” (Zenodo record).  
  Zenodo: https://zenodo.org/records/7766806

**Downstream / evaluation data:**
- **PoLyInfo (NIMS Polymer Database)** provides experimental/literature polymer properties and metadata.  
  Main site: https://polymer.nims.go.jp/en/  
  Overview/help: https://polymer.nims.go.jp/PoLyInfo/guide/en/what_is_polyinfo.html

### 3.2 Minimum CSV schema

The raw CSV for pretraining must include:
- `psmiles` (required) — polymer repeat unit string with `[*]` endpoints

Additionally, for fine-tuning:
- property columns — e.g., ρ, Tg, Tm, Td 

**Example:**
```csv
psmiles,ρ,Tg,Tm,Td
[*]CC(=O)OCCO[*],1.21,55,155,350
```

**Endpoint Note:** When generating RDKit-dependent modalities, the code may internally replace `[*]` with `[At]` to sanitize repeat-unit molecules.

### 3.3 Modalities produced per polymer

PolyFusion represents each polymer using four complementary modalities:

- **PSMILES sequences (D)**  
  SentencePiece tokenization with vocab size 265 (kept fixed during downstream).
- **2D molecular graph (G)**  
  Nodes = atoms, edges = bonds, with chemically meaningful node/edge features.
- **3D conformational proxy (S)**  
  Conformer embedding + optimization pipeline (ETKDG/UFF).
- **Fingerprints (T)**  
  ECFP6 (radius r=3) with 2048 bits.

### 3.4 Preprocessing script

Use the preprocessing utility (Data_Modalities.py) to append multimodal columns to both the pretraining and fine-tuning CSV files:

```bash
python Data_Modalities.py \
  --csv_file /path/to/polymers.csv \
  --chunk_size 1000 \
  --num_workers 24
```

---

## 4. Models & Artifacts

This repository typically produces three artifact families:

### 4.1 PolyFusion Checkpoints (pretraining)

PolyFusion maps each modality into a shared embedding space of dimension **d = 600**. Pretraining uses:
- unified masking with **pmask = 0.15** and an **80/10/10** corruption rule
- anchor–target contrastive learning where the fused structural anchor is aligned to the fingerprint target (InfoNCE with **τ = 0.07**)

**Store:**
- encoder weights per modality
- projection heads
- training config + tokenizer artifacts (SentencePiece model)

**Run:**
```bash
python PolyFusion/CL.py
```

### 4.2 Downstream Predictors (property regression)

Downstream uses:
- fused 600-d embedding
- a lightweight regressor (2-layer MLP, hidden width 300, dropout 0.1)

**Training protocol:**
- 5-fold CV, inner validation (10%) with early stopping

**Save:**
- best weights per property per fold
- scalers used for standardization

**Run:**
```bash
python "Downstream Tasks/Property_Prediction.py"
```

### 4.3 Inverse Design Generator (SELFIES-TED conditioning)

Inverse design conditions a SELFIES-based encoder–decoder (SELFIES-TED) on PolyFusion’s 600-d embedding. Implementation details include:
- conditioning via **K = 4** learned memory tokens
- training-time latent noise **σ_train = 0.10**
- decoding uses top-p (0.92), temperature 1.0, repetition penalty 1.05, max length 256
- property targeting via generate-then-filter using a Gaussian Process Regression (GPR) oracle and acceptance threshold **τ_s = 0.5** (standardized units)

**Save:**
- decoder weights + conditioning projection
- tokenization assets
- property oracle artifacts (GP models / scalers)

**Run:**
```bash
python "Downstream Tasks/Polymer_Generation.py"
```

### 4.4 PolyAgent (Gradio UI)

**Core components:**
- `PolyAgent/orchestrator.py` (controller + tool router)
- `PolyAgent/rag_pipeline.py` (local RAG)
- `PolyAgent/gradio_interface.py` (UI)

**Manuscript controller:**
- GPT-4.1 controller with planning temperature τ_plan = 0.2

**Run:**
```bash
cd PolyAgent
python rag_pipeline.py
python gradio_interface.py --server-name 0.0.0.0 --server-port 7860
```
---

## 5. Pretraining Weights / PolyAgent UI

All pretrained checkpoints, tokenizers, and downstream heads are versioned in the PolyFusionAgent weights repository:

- **Weights repo:** [kaurm43/polyfusionagent-weights](https://huggingface.co/kaurm43/polyfusionagent-weights)  
- **Weights file browser:** [weights/tree/main](https://huggingface.co/kaurm43/polyfusionagent-weights/tree/main)

The interactive PolyAgent interface is available as a Hugging Face Space:

- **Live Space:** [kaurm43/PolyFusionAgent](https://huggingface.co/spaces/kaurm43/PolyFusionAgent)
