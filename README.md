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
- [8. Contact](#8-contact)
- [9. License & Disclaimer](#9-license--disclaimer)

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

## 2. Dependencies & Environment

### 2.1 Installation

```bash
git clone https://github.com/manpreet88/PolyFusionAgent.git
cd PolyFusionAgent

# Recommended: create a fresh environment (conda or venv), then:
pip install -r requirements.txt
2.2 Optional Chemistry & GPU Notes
RDKit (recommended)
Data_Modalities.py and many optional visual/validation steps in generation/agent workflows work best with RDKit.
Recommended installation:

conda install -c conda-forge rdkit
GPU (recommended for training & large runs)
PyTorch + CUDA should match your GPU driver. If you use torch-geometric, install it following the official wheels for your CUDA/PyTorch build.

3. Data, Modalities, and Preprocessing
3.1 Input CSV schema
At minimum, your dataset CSV should include a polymer string column:

psmiles (required): polymer SMILES / PSMILES string (often contains [*] endpoints)

Optional:

source (optional): any identifier/source tag

property columns (optional): e.g., density, Tg, Tm, Td, etc. (names vary—see downstream scripts’ column matching)

Example:

psmiles,source,density,glass transition,melting,thermal decomposition
[*]CC(=O)OCCO[*],PI1M,1.21,55,155,350
...
Wildcard handling: this code replaces * (atomicNum 0) with Astatine (At, Z=85) internally for RDKit robustness, while preserving endpoint semantics.

3.2 Generate multimodal columns (graph/geometry/fingerprints)
Use Data_Modalities.py to process a CSV and append JSON blobs for:

graph

geometry

fingerprints

python Data_Modalities.py \
  --csv_file /path/to/your/polymers.csv \
  --chunk_size 1000 \
  --num_workers 24
Outputs:

/path/to/your/polymers_processed.csv (same rows + new modality columns)

/path/to/your/polymers_failures.jsonl (failures with index/smiles/error)

3.3 What “graph”, “geometry”, and “fingerprints” look like
Each processed row stores modalities as JSON strings.

graph contains:

node_features: atomic_num, degree, formal_charge, hybridization, aromatic/ring flags, chirality, etc.

edge_indices + edge_features (bond_type, stereo, conjugation, etc.)

adjacency_matrix

graph_features (MolWt, LogP, TPSA, rings, rotatable bonds, HBA/HBD, ...)

geometry contains:

ETKDG-generated conformers, optimized via MMFF/UFF (best energy chosen)

best_conformer: atomic_numbers + coordinates + energy + optional 3D descriptors

falls back to 2D coords if 3D fails

fingerprints contains:

Morgan fingerprints (bitstrings + counts) for radii up to 3 (default)

e.g., morgan_r3_bits, morgan_r3_counts, plus smaller radii

4. Models & Artifacts
This repo is organized so you can train and export artifacts for:

PolyFusion (pretraining)
multimodal CL checkpoint bundle (e.g., multimodal_output/best/...)

unimodal encoder checkpoints (optional, used by some scripts)

Downstream (best weights per property)
saved best checkpoint per property (CV selection)

directory example: multimodal_downstream_bestweights/...

Inverse design generator artifacts
decoder bundles + scalers + (optionally) SentencePiece tokenizer assets

directory example: multimodal_inverse_design_output/.../best_models

Important: Several scripts include placeholder paths at the top (e.g., /path/to/...). You must update them for your filesystem.

5. Running the Code
5.1 Multimodal contrastive pretraining (PolyFusion)
Main entry:

PolyFusion/CL.py

What it does (high-level):

Streams a large CSV (CSV_PATH) and writes per-sample .pt files to avoid RAM spikes.

Encodes polymer modalities with DeBERTaV2 (PSMILES), GINE (2D), SchNet (3D), Transformer (fingerprints).

Projects each modality embedding into a shared space.

Trains with contrastive alignment (InfoNCE) + optional reconstruction objectives.

Steps

Edit path placeholders in PolyFusion/CL.py, e.g.:

CSV_PATH

SPM_MODEL

PREPROC_DIR

OUTPUT_DIR and BEST_*_DIR locations (if used)

Run:

python PolyFusion/CL.py
Tip: Start with a smaller TARGET_ROWS (e.g., 100k) to validate pipeline correctness before scaling.

5.2 Downstream property prediction
Script:

Downstream Tasks/Property_Prediction.py

This script:

loads your dataset CSV with modalities (e.g., polyinfo_with_modalities.csv)

loads pretrained encoders / CL fused backbone

trains a fusion + regression head for each requested property

evaluates using true K-fold (NUM_RUNS = 5) and saves best weights

Steps

Update placeholders near the top of the script:

POLYINFO_PATH

PRETRAINED_MULTIMODAL_DIR

optional: BEST_*_DIR (if needed)

output paths: OUTPUT_RESULTS, BEST_WEIGHTS_DIR

Run:

python "Downstream Tasks/Property_Prediction.py"
Requested properties (default)

REQUESTED_PROPERTIES = [
  "density",
  "glass transition",
  "melting",
  "specific volume",
  "thermal decomposition"
]
The script includes a robust column-matching function that tries to map these names to your dataframe’s actual column headers.

5.3 Inverse design / polymer generation
Script:

Downstream Tasks/Polymer_Generation.py

Core idea:

condition a SELFIES-TED-style decoder on PolyFusion embeddings,

guide sampling toward target property values (with optional latent noise and verification)

Steps

Update placeholders in the Config dataclass:

POLYINFO_PATH

pretrained weights directories (CL + downstream + tokenizer)

output directory OUTPUT_DIR

Run:

python "Downstream Tasks/Polymer_Generation.py"
Notes

If RDKit and SELFIES are installed, the script can:

validate chemistry constraints more robustly

convert polymer endpoints safely (e.g., [*] ↔ [At] internal representation)

5.4 PolyAgent (Gradio UI)
Files:

PolyAgent/orchestrator.py (core engine)

PolyAgent/gradio_interface.py (UI)

PolyAgent/rag_pipeline.py (local RAG utilities)

What you configure
In PolyAgent/orchestrator.py, update the PathsConfig placeholders, e.g.:

cl_weights_path

downstream_bestweights_5m_dir

inverse_design_5m_dir

spm_model_path, spm_vocab_path

chroma_db_path (if using local RAG store)

Environment variables

OPENAI_API_KEY (required for planning/composition)

Optional (improves retrieval coverage):

OPENAI_MODEL (defaults set in config)

HF_TOKEN (if pulling HF artifacts)

SPRINGER_NATURE_API_KEY, SEMANTIC_SCHOLAR_API_KEY

Run the UI

cd PolyAgent
python gradio_interface.py --server-name 0.0.0.0 --server-port 7860
Prompting tips

To trigger inverse design: include “generate” / “inverse design” and a target value:

target_value=60 or Tg 60

Provide a seed polymer pSMILES in a code block:

[*]CC(=O)OCCOCCOC(=O)C[*]
If you need more citations, ask explicitly:

“cite 10 papers”

6. Results & Reproducibility
PolyFusion is designed for scalable multimodal alignment across large polymer corpora.

Downstream scripts perform K-fold evaluation per property and save best weights.

PolyAgent produces evidence-linked answers with tool outputs and DOI-style links (when available).

Reproducibility reminder: Several scripts currently use in-file configuration constants (placeholders). For a clean workflow, keep a consistent folder layout for datasets and checkpoints and update paths in one place (or refactor into a shared config module).

7. Citation
If you use this repository in your work, please cite the accompanying manuscript:

@article{kaur2026polyfusionagent,
  title   = {PolyFusionAgent: a multimodal foundation model and autonomous AI assistant for polymer informatics},
  author  = {Kaur, Manpreet and Liu, Qian},
  year    = {2026},
  note    = {Manuscript / preprint},
}
Replace the BibTeX entry above with the final venue DOI/citation when available.

8. Contact
Corresponding author: Qian Liu — qi.liu@uwinnipeg.ca

Contributing author: Manpreet Kaur — kaur-m43@webmail.uwinnipeg.ca

9. License & Disclaimer
License: (Add your license file here; e.g., MIT / Apache-2.0 / CC BY-NC for models)

Disclaimer: This codebase is provided for research and development use. Polymer generation outputs and suggested candidates should be validated with domain expertise, safety constraints, and experimental verification before deployment.
