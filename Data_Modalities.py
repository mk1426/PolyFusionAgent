import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdDepictor
from rdkit.Chem import Crippen, Descriptors3D
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import os
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Tuple

# ----------------------------------------------------------------------
# Logging / RDKit hygiene
# ----------------------------------------------------------------------

# RDKit can be chatty; we silence logs above via RDLogger.DisableLog.
# We also suppress Python warnings (set above).

# ----------------------------------------------------------------------
# Wildcard ("*") handling utilities
# ----------------------------------------------------------------------

ATOMIC_NUM_AT = 85  # Astatine (At) used as a placeholder for wildcard atoms


def process_star_atoms(mol: Chem.Mol) -> Chem.Mol:
    """
    Replace all wildcard atoms ("*" or atomicNum == 0) with Astatine (At, Z=85).

    Rationale:
    - Polymer SMILES often contain '*' to indicate attachment points.
    - Many RDKit operations fail or sanitize differently with atomicNum == 0.
    - Mapping '*' -> At allows sanitization and downstream featurization while
      keeping a consistent placeholder identity.
    """
    if mol is None:
        return mol

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() == "*":
            atom.SetAtomicNum(ATOMIC_NUM_AT)
    return mol


# ----------------------------------------------------------------------
                      # Per-polymer worker function
# ----------------------------------------------------------------------

def process_single_polymer(args) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Worker that processes one row (one polymer) and returns:
      (polymer_data, failed_info)

    polymer_data is a dict containing serialized multimodal outputs.
    failed_info is a dict with index/smiles/error if anything fails.
    """
    idx, row_dict, extractor = args
    polymer_data = None
    failed_info = None

    try:
        smiles = row_dict.get("psmiles", None)
        source = row_dict.get("source", None)

        if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles.strip()) == 0:
            failed_info = {"index": idx, "smiles": str(smiles), "error": "Empty or invalid SMILES"}
            return polymer_data, failed_info

        canonical_smiles = extractor.validate_and_standardize_smiles(smiles)
        if canonical_smiles is None:
            failed_info = {"index": idx, "smiles": smiles, "error": "Invalid SMILES or cannot be standardized"}
            return polymer_data, failed_info

        polymer_data = {
            "original_index": idx,
            "psmiles": canonical_smiles,
            "source": source,
            "smiles": canonical_smiles,
        }

        # Graph
        try:
            polymer_data["graph"] = extractor.generate_molecular_graph(canonical_smiles)
        except Exception:
            polymer_data["graph"] = {}

        # Geometry
        try:
            polymer_data["geometry"] = extractor.optimize_3d_geometry(canonical_smiles)
        except Exception:
            polymer_data["geometry"] = {}

        # Fingerprints
        try:
            polymer_data["fingerprints"] = extractor.calculate_morgan_fingerprints(canonical_smiles)
        except Exception:
            polymer_data["fingerprints"] = {}

        return polymer_data, failed_info

    except Exception as e:
        failed_info = {"index": idx, "smiles": row_dict.get("psmiles", ""), "error": str(e)}
        return polymer_data, failed_info


# ----------------------------------------------------------------------
# Main extractor class
# ----------------------------------------------------------------------

class AdvancedPolymerMultimodalExtractor:
    """
    Multimodal extractor that reads a CSV of polymers and adds:
      - graph: node/edge features + adjacency + summary graph features
      - geometry: best 3D conformer (or fallback 2D coords) + 3D descriptors
      - fingerprints: Morgan fingerprints (bitstrings + counts) for multiple radii

    Output:
      - <input>_processed.csv (appended chunk-by-chunk)
      - <input>_failures.jsonl (one JSON per failure)
    """

    def __init__(self, csv_file: str):
        self.csv_file = str(csv_file)

    # ------------------------------
    # SMILES validation/standardization
    # ------------------------------
    def validate_and_standardize_smiles(self, smiles: str) -> Optional[str]:
        """
        Parse, sanitize, replace '*' with At, and return canonical SMILES.
        Returns None if parsing/sanitization fails.
        """
        try:
            if not smiles or pd.isna(smiles):
                return None

            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None

            mol = process_star_atoms(mol)  # pass 1
            Chem.SanitizeMol(mol)
            mol = process_star_atoms(mol)  # pass 2 (robust)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            if not canonical_smiles:
                return None
            return canonical_smiles

        except Exception:
            return None

    # ------------------------------
    # Molecular graph (RDKit -> JSONable dict)
    # ------------------------------
    def generate_molecular_graph(self, smiles: str) -> Dict:
        """
        Build a molecular graph representation with atom/bond features and
        global graph descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = process_star_atoms(mol)
        if mol is None:
            return {}

        # Explicit hydrogens for atom-level features 
        mol = Chem.AddHs(mol)

        node_features = []
        for atom in mol.GetAtoms():
            node_features.append(
                {
                    "atomic_num": atom.GetAtomicNum(),
                    "degree": atom.GetDegree(),
                    "formal_charge": atom.GetFormalCharge(),
                    "hybridization": int(atom.GetHybridization()),
                    "is_aromatic": atom.GetIsAromatic(),
                    "is_in_ring": atom.IsInRing(),
                    "chirality": int(atom.GetChiralTag()),
                    "mass": atom.GetMass(),
                    "valence": atom.GetTotalValence(),
                    "num_radical_electrons": atom.GetNumRadicalElectrons(),
                }
            )

        edge_features = []
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_features.append(
                {
                    "bond_type": int(bond.GetBondType()),
                    "is_aromatic": bond.GetIsAromatic(),
                    "is_in_ring": bond.IsInRing(),
                    "stereo": int(bond.GetStereo()),
                    "is_conjugated": bond.GetIsConjugated(),
                }
            )

            # Undirected -> store both directions for GNN-style edge lists
            edge_indices.extend([[i, j], [j, i]])

        graph_features = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
            "num_h_donors": rdMolDescriptors.CalcNumHBD(mol),
        }

        adj = Chem.GetAdjacencyMatrix(mol).tolist()

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "edge_indices": edge_indices,
            "graph_features": graph_features,
            "adjacency_matrix": adj,
        }

    # ------------------------------
    # 3D geometry (ETKDG + MMFF/UFF) 
    # ------------------------------
    def optimize_3d_geometry(self, smiles: str, num_conformers: int = 10) -> Dict:
        """
        Generate multiple conformers, optimize (MMFF if available else UFF),
        and return the lowest-energy conformer coordinates + 3D descriptors.

        If no conformer is generated/optimized, fall back to 2D coordinates.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > 200:
            return {}

        mol = process_star_atoms(mol)
        mol_h = Chem.AddHs(mol)

        # Atomic numbers aligned to coordinate ordering (mol_h atoms)
        atomic_numbers = [atom.GetAtomicNum() for atom in mol_h.GetAtoms()]

        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            conformer_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        except Exception:
            conformer_ids = []

        best_conformer = None
        best_energy = float("inf")

        for conf_id in conformer_ids:
            try:
                mmff_ok = AllChem.MMFFHasAllMoleculeParams(mol_h)

                if mmff_ok:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)
                    props = AllChem.MMFFGetMoleculeProperties(mol_h)
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=conf_id)
                else:
                    AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id)
                    ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)

                energy = ff.CalcEnergy() if ff is not None else None
                if energy is None or energy >= best_energy:
                    continue

                conf = mol_h.GetConformer(conf_id)
                coords = [
                    [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol_h.GetNumAtoms())
                ]

                descriptors_3d = {}
                try:
                    descriptors_3d = {
                        "asphericity": Descriptors3D.Asphericity(mol_h, confId=conf_id),
                        "eccentricity": Descriptors3D.Eccentricity(mol_h, confId=conf_id),
                        "inertial_shape_factor": Descriptors3D.InertialShapeFactor(mol_h, confId=conf_id),
                        "radius_of_gyration": Descriptors3D.RadiusOfGyration(mol_h, confId=conf_id),
                        "spherocity_index": Descriptors3D.SpherocityIndex(mol_h, confId=conf_id),
                    }
                except Exception:
                    pass

                best_conformer = {
                    "conformer_id": int(conf_id),
                    "coordinates": coords,
                    "atomic_numbers": atomic_numbers,
                    "energy": float(energy),
                    "descriptors_3d": descriptors_3d,
                }
                best_energy = energy

            except Exception:
                continue

        if best_conformer is not None:
            return {
                "best_conformer": best_conformer,
                "num_conformers_generated": int(len(conformer_ids)),
                "converted_smiles": Chem.MolToSmiles(mol),
            }

        # Fallback: 2D coordinates
        try:
            rdDepictor.Compute2DCoords(mol)
            coords_2d = mol.GetConformer().GetPositions().tolist()
            atomic_numbers_2d = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            return {
                "best_conformer": {
                    "conformer_id": -1,
                    "coordinates": coords_2d,
                    "atomic_numbers": atomic_numbers_2d,
                    "energy": None,
                    "descriptors_3d": {},
                },
                "num_conformers_generated": 0,
                "converted_smiles": Chem.MolToSmiles(mol),
            }
        except Exception:
            return {}

    # ------------------------------
    # Morgan fingerprints (multi-radius)
    # ------------------------------
    def calculate_morgan_fingerprints(self, smiles: str, radius: int = 3, n_bits: int = 2048) -> Dict:
        """
        Compute Morgan fingerprints:
          - bitstring (as list of '0'/'1' chars) at radius=radius
          - counts (as dict) at radius=radius
        Also includes all radii r in [1, radius-1].
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = process_star_atoms(mol)
        if mol is None:
            return {}

        fingerprints = {}

        # Main radius
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_bitvect = generator.GetFingerprint(mol)
        fingerprints[f"morgan_r{radius}_bits"] = list(fp_bitvect.ToBitString())
        fingerprints[f"morgan_r{radius}_counts"] = dict(AllChem.GetMorganFingerprint(mol, radius).GetNonzeroElements())

        # Additional radii
        for r in range(1, radius):
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=n_bits)
            bitvect = gen.GetFingerprint(mol)
            fingerprints[f"morgan_r{r}_bits"] = list(bitvect.ToBitString())
            fingerprints[f"morgan_r{r}_counts"] = dict(AllChem.GetMorganFingerprint(mol, r).GetNonzeroElements())

        return fingerprints

    # ------------------------------
    # Chunked parallel processing over CSV
    # ------------------------------
    def process_all_polymers_parallel(self, chunk_size: int = 100, num_workers: int = 40) -> str:
        """
        Read the input CSV in chunks, fill missing multimodal columns, and process
        only rows that are missing any of: graph/geometry/fingerprints.

        Appends processed chunks to <input>_processed.csv and failures to <input>_failures.jsonl.
        """
        chunk_iterator = pd.read_csv(self.csv_file, chunksize=chunk_size, engine="python")

        for chunk in chunk_iterator:
            # Ensure expected output columns exist and are object dtype (for JSON strings)
            for col in ["graph", "geometry", "fingerprints"]:
                if col not in chunk.columns:
                    chunk[col] = None
                chunk[col] = chunk[col].astype(object)

            # Only process rows missing any modality
            chunk_to_process = chunk[chunk[["graph", "geometry", "fingerprints"]].isnull().any(axis=1)].copy()

            # If all rows already done, just persist chunk and continue
            if len(chunk_to_process) == 0:
                self.save_chunk_to_csv(chunk)
                continue

            rows = list(chunk_to_process.iterrows())
            argslist = [(i, row.to_dict(), self) for i, row in rows]

            with mp.Pool(num_workers) as pool:
                results = pool.map(process_single_polymer, argslist)

            failed_list = []
            for n, (output, fail) in enumerate(results):
                idx = rows[n][0]
                if output:
                    chunk.at[idx, "graph"] = json.dumps(output["graph"])
                    chunk.at[idx, "geometry"] = json.dumps(output["geometry"])
                    chunk.at[idx, "fingerprints"] = json.dumps(output["fingerprints"])
                if fail:
                    failed_list.append(fail)

            self.save_chunk_to_csv(chunk)
            self.save_failed_to_json(failed_list)

        return "Processing Done"

    # ------------------------------
    # Output helpers
    # ------------------------------
    def save_chunk_to_csv(self, chunk: pd.DataFrame) -> None:
        """
        Append processed chunk to <input>_processed.csv.
        """
        out_csv = self.csv_file.replace(".csv", "_processed.csv")
        if not os.path.exists(out_csv):
            chunk.to_csv(out_csv, index=False, mode="w")
        else:
            chunk.to_csv(out_csv, index=False, mode="a", header=False)

    def save_failed_to_json(self, failed_list) -> None:
        """
        Append failures to <input>_failures.jsonl (JSON lines).
        """
        if not failed_list:
            return
        fail_json = self.csv_file.replace(".csv", "_failures.jsonl")
        with open(fail_json, "a", encoding="utf-8") as f:
            for fail in failed_list:
                json.dump(fail, f)
                f.write("\n")

    def save_results(self, output_file: str = "polymer_multimodal_data.json"):
        pass

    def generate_summary_statistics(self) -> Dict:
        return {}


# ----------------------------------------------------------------------
# CLI / entry-point helpers
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Command-line arguments:
      --csv_file: path to input CSV (required)
      --chunk_size: rows per chunk
      --num_workers: multiprocessing workers
    """
    parser = argparse.ArgumentParser(description="Polymer multimodal feature extraction (RDKit).")
    parser.add_argument(
        "--csv_file",
        type=str,
        default="/path/to/polymer_structures_unified.csv",
        help="Path to the input CSV file containing at least a 'psmiles' column.",
    )
    parser.add_argument("--chunk_size", type=int, default=1000, help="Rows per chunk for streaming CSV processing.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of parallel worker processes.")
    return parser.parse_args()


def main() -> Tuple[AdvancedPolymerMultimodalExtractor, Optional[object]]:
    """
    Script entry point.
    Reads arguments, constructs the extractor, and runs chunked parallel processing.
    """
    args = parse_args()
    csv_file = args.csv_file

    extractor = AdvancedPolymerMultimodalExtractor(csv_file)
    try:
        extractor.process_all_polymers_parallel(chunk_size=args.chunk_size, num_workers=args.num_workers)
    except KeyboardInterrupt:
        return extractor, None
    except Exception as e:
        print(f"CRASH! Error: {e}")
        return extractor, None

    print("\n=== Processing Complete ===")
    return extractor, None


if __name__ == "__main__":
    extractor, results = main()
