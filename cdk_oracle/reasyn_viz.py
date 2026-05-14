# Copyright (c) 2026, NVIDIA CORPORATION. Licensed under the Apache License, Version 2.0 (the "License") you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

"""ReaSyn synthesis pathway visualization helpers."""

import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator
from IPython.display import HTML, display


# ---------------------------------------------------------------------------
# Route validation via Tanimoto fingerprint analysis
# ---------------------------------------------------------------------------

def _get_morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    """Return a Morgan fingerprint bit vector, or None if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def _extract_building_blocks(synthesis_str: str) -> List[str]:
    """Extract SMILES of building blocks from a synthesis string (skip Rxx template IDs)."""
    if not synthesis_str:
        return []
    return [
        p.strip() for p in synthesis_str.split(";")
        if p.strip() and not (p.strip().startswith("R") and p.strip()[1:].isdigit())
    ]


def validate_route(target_smi: str, synthesis_str: str) -> Dict[str, float]:
    """Validate a synthesis route by comparing building block fingerprints to the target.

    Returns dict with:
      - fp_coverage:  Tanimoto(target_fp, union_of_building_block_fps)
                      Measures what fraction of the target's chemical features
                      are present in the building blocks. High → route is
                      structurally consistent.
      - atom_ratio:   sum(heavy_atoms in building blocks) / heavy_atoms in target.
                      Should be ≥ 1.0 (building blocks have leaving groups).
                      Much > 1 or < 1 is suspicious.
      - bb_count:     Number of valid building blocks parsed.
      - max_bb_sim:   Highest single-building-block Tanimoto to target
                      (shows the most "target-like" fragment).
    """
    target_fp = _get_morgan_fp(target_smi)
    if target_fp is None:
        return {"fp_coverage": None, "atom_ratio": None, "bb_count": 0, "max_bb_sim": None}

    target_mol = Chem.MolFromSmiles(target_smi)
    target_heavy = target_mol.GetNumHeavyAtoms()

    bb_smiles = _extract_building_blocks(synthesis_str)
    bb_fps = []
    bb_heavy_total = 0
    max_sim = 0.0

    for smi in bb_smiles:
        fp = _get_morgan_fp(smi)
        if fp is None:
            continue
        bb_fps.append(fp)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            bb_heavy_total += mol.GetNumHeavyAtoms()
        sim = DataStructs.TanimotoSimilarity(target_fp, fp)
        max_sim = max(max_sim, sim)

    if not bb_fps:
        return {"fp_coverage": None, "atom_ratio": None, "bb_count": 0, "max_bb_sim": None}

    # Union (bitwise OR) of all building block fingerprints
    union_fp = bb_fps[0]
    for fp in bb_fps[1:]:
        union_fp = union_fp | fp

    coverage = DataStructs.TanimotoSimilarity(target_fp, union_fp)
    atom_ratio = bb_heavy_total / target_heavy if target_heavy > 0 else 0

    return {
        "fp_coverage": round(coverage, 3),
        "atom_ratio": round(atom_ratio, 2),
        "bb_count": len(bb_fps),
        "max_bb_sim": round(max_sim, 3),
    }


def parse_reasyn_results(reasyn_results: dict, compound_ids: List[str],
                         validate: bool = True) -> pd.DataFrame:
    """Convert raw ReaSyn JSON into a summary DataFrame.

    If validate=True, runs fingerprint coverage analysis on each best route.
    """
    records = []
    for i, result in enumerate(reasyn_results.get("results", [])):
        pathways = result.get("pathways", [])
        best = pathways[0] if pathways else {}
        target_smi = result.get("target", "")
        best_synthesis = best.get("synthesis", "")

        rec = {
            "compound_id": compound_ids[i] if i < len(compound_ids) else f"compound_{i}",
            "smiles": target_smi,
            "synthesizable": result.get("success", False),
            "num_pathways": len(pathways),
            "best_score": best.get("score"),
            "best_steps": best.get("num_steps"),
            "best_synthesis": best_synthesis,
        }

        if validate and best_synthesis:
            val = validate_route(target_smi, best_synthesis)
            rec.update(val)
        elif validate:
            rec.update({"fp_coverage": None, "atom_ratio": None, "bb_count": 0, "max_bb_sim": None})

        records.append(rec)
    return pd.DataFrame(records)


def _smi_to_img_tag(smi: str, size=(220, 160)) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return f'<code style="font-size:10px">{smi[:40]}</code>'
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" width="{size[0]}" height="{size[1]}">'


def _synthesis_to_html(synthesis_str: str) -> str:
    """Parse 'SMILES;SMILES;Rxx;SMILES;Rxx;...' into reactant images + reaction labels."""
    if not synthesis_str:
        return "—"
    parts = [p.strip() for p in synthesis_str.split(";") if p.strip()]
    steps, current_reactants = [], []
    for part in parts:
        if part.startswith("R") and part[1:].isdigit():
            steps.append((current_reactants, part))
            current_reactants = []
        else:
            current_reactants.append(part)
    if current_reactants:
        steps.append((current_reactants, None))

    html_parts = []
    for i, (reactants, rxn_id) in enumerate(steps):
        mol_imgs = [_smi_to_img_tag(smi, size=(140, 100)) for smi in reactants]
        reactant_html = ' <span style="font-size:18px;vertical-align:middle;">+</span> '.join(mol_imgs)
        rxn_label = (
            f' <span style="background:#e8e8e8;padding:2px 6px;border-radius:3px;'
            f'font-size:11px;vertical-align:middle;">{rxn_id}</span>'
            if rxn_id else ""
        )
        if i > 0:
            html_parts.append('<span style="font-size:20px;vertical-align:middle;margin:0 4px;">→</span>')
        html_parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;">'
            f'{reactant_html}{rxn_label}</span>'
        )
    return "".join(html_parts)


def _coverage_badge(val) -> str:
    """Color-coded badge for fingerprint coverage or atom ratio."""
    if val is None:
        return "—"
    if isinstance(val, float):
        if val >= 0.8:
            color = "#2e7d32"  # green
        elif val >= 0.5:
            color = "#f57f17"  # amber
        else:
            color = "#c62828"  # red
        return f'<span style="color:{color};font-weight:bold;">{val:.2f}</span>'
    return str(val)


def _atom_ratio_badge(val) -> str:
    if val is None:
        return "—"
    # Ideal range: 1.0 – 1.5 (building blocks slightly heavier due to leaving groups)
    if 0.9 <= val <= 1.8:
        color = "#2e7d32"
    elif 0.7 <= val <= 2.5:
        color = "#f57f17"
    else:
        color = "#c62828"
    return f'<span style="color:{color};font-weight:bold;">{val:.1f}x</span>'


def display_synthesis_table(df: pd.DataFrame, total_submitted: int = None):
    """Render a rich HTML table with 2D structures, synthesis route sketches, and validation."""
    total = total_submitted or len(df)
    has_validation = "fp_coverage" in df.columns

    print("=" * 70)
    print("ReaSyn Synthesis Pathway Summary")
    print("=" * 70)
    print(f"  Total compounds submitted:  {total}")
    print(f"  Synthesizable:              {df['synthesizable'].sum()}")
    print(f"  Not synthesizable:          {(~df['synthesizable']).sum()}")
    if has_validation:
        valid_cov = df["fp_coverage"].dropna()
        if len(valid_cov) > 0:
            print(f"  Avg FP coverage:            {valid_cov.mean():.3f}  (min={valid_cov.min():.3f}, max={valid_cov.max():.3f})")
    print("=" * 70)

    rows_html = ""
    for _, row in df.iterrows():
        synth_badge = (
            '<span style="color:green;font-weight:bold;font-size:18px">✓</span>'
            if row["synthesizable"]
            else '<span style="color:red;font-weight:bold;font-size:18px">✗</span>'
        )
        score_str = f"{row['best_score']:.2f}" if row["best_score"] is not None else "—"
        steps_str = str(row["best_steps"]) if row["best_steps"] is not None else "—"
        pathway_html = _synthesis_to_html(row["best_synthesis"]) if row["best_synthesis"] else "—"
        td = 'style="padding:8px;vertical-align:middle;"'
        tdc = 'style="padding:8px;text-align:center;vertical-align:middle;"'

        val_cells = ""
        if has_validation:
            val_cells = (
                f'<td {tdc}>{_coverage_badge(row.get("fp_coverage"))}</td>'
                f'<td {tdc}>{_atom_ratio_badge(row.get("atom_ratio"))}</td>'
                f'<td {tdc}>{_coverage_badge(row.get("max_bb_sim"))}</td>'
            )

        rows_html += (
            f'<tr style="border-bottom:1px solid #ddd;">'
            f'<td {td}><b>{row["compound_id"]}</b></td>'
            f'<td {td}>{_smi_to_img_tag(row["smiles"])}</td>'
            f'<td {tdc}>{synth_badge}</td>'
            f'<td {tdc}>{row["num_pathways"]}</td>'
            f'<td {tdc}>{score_str}</td>'
            f'<td {tdc}>{steps_str}</td>'
            f'{val_cells}'
            f'<td {td}>{pathway_html}</td>'
            f'</tr>'
        )

    th = 'style="padding:8px 12px; border:1px solid #ddd; text-align:center;"'
    val_headers = ""
    if has_validation:
        val_headers = (
            f'<th {th} title="Tanimoto similarity between target FP and union of building block FPs. '
            f'Higher = building blocks cover more of the target structure.">FP Coverage</th>'
            f'<th {th} title="Sum of heavy atoms in building blocks / heavy atoms in target. '
            f'Ideal: 1.0-1.5x (leaving groups add mass).">Atom Ratio</th>'
            f'<th {th} title="Highest Tanimoto similarity between any single building block and the target. '
            f'Shows the most target-like fragment.">Max BB Sim</th>'
        )
    html = (
        f'<table style="border-collapse:collapse; font-family:sans-serif; font-size:13px;">'
        f'<thead><tr style="background:#f0f0f0;">'
        f'<th {th}>Compound</th><th {th}>Target Structure</th>'
        f'<th {th}>Synth?</th><th {th}># Pathways</th>'
        f'<th {th}>Best Score</th><th {th}>Steps</th>'
        f'{val_headers}'
        f'<th {th}>Best Synthesis Route</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
    )

    if has_validation:
        legend = (
            '<div style="margin-top:12px;font-size:12px;color:#555;font-family:sans-serif;">'
            '<b>Validation columns:</b> '
            '<b>FP Coverage</b> = Tanimoto(target, union of building block fingerprints) — '
            'do the building blocks collectively contain the target\'s chemical features? '
            '<b>Atom Ratio</b> = total heavy atoms in building blocks / target heavy atoms — '
            'ideal 1.0–1.5x (leaving groups add mass). '
            '<b>Max BB Sim</b> = highest single building block similarity to target — '
            'shows the most target-like fragment. '
            '<span style="color:#2e7d32">■</span> good '
            '<span style="color:#f57f17">■</span> marginal '
            '<span style="color:#c62828">■</span> suspicious'
            '</div>'
        )
    else:
        legend = ""

    display(HTML(html + legend))


def print_pathway_details(reasyn_results: dict, compound_ids: List[str], top_n: int = 3):
    """Print step-by-step retrosynthetic pathways for each compound."""
    print(f"Showing top {top_n} retrosynthetic pathways per compound.\n")
    print("Legend:")
    print("  Score  — ReaSyn confidence (0→1, higher = more feasible route)")
    print("  Steps  — Number of sequential reactions to synthesize the target")
    print("  Each step shows: reactant SMILES → [reaction template ID]\n")

    for i, result in enumerate(reasyn_results.get("results", [])):
        target_smi = result.get("target", "")
        pathways = result.get("pathways", [])
        success = result.get("success", False)
        cid = compound_ids[i] if i < len(compound_ids) else f"compound_{i}"
        status = "✓ synthesizable" if success else "✗ no route found"

        print(f"{'=' * 70}")
        print(f"Compound: {cid}  [{status}, {len(pathways)} pathways]")
        print(f"Target:   {target_smi[:80]}{'...' if len(target_smi) > 80 else ''}")

        if pathways:
            for j, pathway in enumerate(pathways[:top_n]):
                score = pathway.get("score", 0)
                steps = pathway.get("num_steps", "?")
                synthesis = pathway.get("synthesis", "")
                print(f"\n  Route {j+1}/{min(top_n, len(pathways))}:  score={score:.2f}  steps={steps}")
                if synthesis:
                    parts = [p.strip() for p in synthesis.split(";") if p.strip()]
                    step_num, reactants = 1, []
                    for part in parts:
                        if part.startswith("R") and part[1:].isdigit():
                            print(f"    Step {step_num}: {' + '.join(reactants)}  →  [{part}]")
                            reactants = []
                            step_num += 1
                        else:
                            reactants.append(part)
                    if reactants:
                        print(f"    Remaining building blocks: {' + '.join(reactants)}")
        else:
            print("  No synthesis pathways found.")
        print()

    print("=" * 70)
    print("Done.")
