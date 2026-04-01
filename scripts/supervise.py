#!/usr/bin/env python3
"""
Interface de supervision via Argilla (HuggingFace Space).
Un record Argilla par désaccord de span inter-annotateurs.
"""

import argparse, json, os, sys, copy, webbrowser
from difflib import SequenceMatcher

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import pandas as pd
import argilla as rg

EMOTIONS = [
    "Colère", "Dégoût", "Joie", "Peur", "Surprise", "Tristesse",
    "Admiration", "Culpabilité", "Embarras", "Fierté", "Jalousie", "Autre",
]

_CAT_NORMALIZE = {e: e for e in EMOTIONS}


# ═══ DATA LOADING ══════════════════════════════════════════════════════════

def _aggregate_sitemo_to_emotions(sitemo_units):
    emo_dict = {e: 0 for e in EMOTIONS}
    for unit in sitemo_units:
        if not isinstance(unit, dict):
            continue
        for field in ("categorie", "categorie2"):
            cat = unit.get(field)
            if cat and cat in _CAT_NORMALIZE:
                emo_dict[_CAT_NORMALIZE[cat]] = 1
    return emo_dict


def load_run(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = {"idx": rec["idx"], "row_id": rec.get("row_id"),
                   "json_ok": rec.get("json_ok", False),
                   "raw_text": rec.get("raw_text", "{}")}
            pj = rec.get("parsed_json")
            row["parsed_json"] = pj if isinstance(pj, dict) else {}

            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            row["target_name"] = meta.get("target_name", "")
            row["target_role"] = meta.get("target_role", "")
            row["thematique"] = meta.get("thematique", "")

            if rec["json_ok"] and isinstance(pj, dict):
                if "sitemo_units" in pj:
                    emo = _aggregate_sitemo_to_emotions(
                        pj.get("sitemo_units", []))
                    for e in EMOTIONS:
                        row[e] = int(emo.get(e, 0))
                elif "emotions" in pj:
                    emo = pj.get("emotions", {})
                    for e in EMOTIONS:
                        row[e] = int(emo.get(e, 0))
                else:
                    for e in EMOTIONS:
                        row[e] = None
            else:
                for e in EMOTIONS:
                    row[e] = None
            rows.append(row)
    return pd.DataFrame(rows)


# ═══ SPAN MATCHING ═════════════════════════════════════════════════════════

def _match_spans(units_r1, units_r2, threshold=0.85):
    """Match spans between R1 and R2 by text similarity.
    Returns (matched_pairs, unmatched_r1, unmatched_r2)."""
    matched = []
    used_r2 = set()
    unmatched_r1_list = []

    for u1 in units_r1:
        s1 = u1.get("span_text", "").strip().lower()
        best_j, best_ratio = -1, 0.0
        for j, u2 in enumerate(units_r2):
            if j in used_r2:
                continue
            s2 = u2.get("span_text", "").strip().lower()
            if s1 == s2:
                ratio = 1.0
            elif s1 in s2 or s2 in s1:
                ratio = 0.95
            else:
                ratio = SequenceMatcher(None, s1, s2).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_j = j
        if best_ratio >= threshold:
            matched.append((u1, units_r2[best_j]))
            used_r2.add(best_j)
        else:
            unmatched_r1_list.append(u1)

    unmatched_r2_list = [u for j, u in enumerate(units_r2)
                         if j not in used_r2]
    return matched, unmatched_r1_list, unmatched_r2_list


def _has_annotation_diff(u1, u2):
    """True if two matched spans differ in category or mode."""
    return (u1.get("categorie") != u2.get("categorie") or
            u1.get("categorie2") != u2.get("categorie2") or
            u1.get("mode") != u2.get("mode"))


def compute_disagreements(row):
    """Return list of disagreement dicts for a merged row."""
    pj_r1 = row.get("parsed_json_r1", {})
    pj_r2 = row.get("parsed_json_r2", {})
    if not isinstance(pj_r1, dict):
        pj_r1 = {}
    if not isinstance(pj_r2, dict):
        pj_r2 = {}

    units_r1 = pj_r1.get("sitemo_units", [])
    units_r2 = pj_r2.get("sitemo_units", [])
    if not isinstance(units_r1, list):
        units_r1 = []
    if not isinstance(units_r2, list):
        units_r2 = []

    matched, unmatched_r1, unmatched_r2 = _match_spans(
        units_r1, units_r2)

    disagreements = []
    for u1, u2 in matched:
        if _has_annotation_diff(u1, u2):
            disagreements.append({
                "type": "mismatch",
                "span_text": u1.get("span_text", ""),
                "r1": u1, "r2": u2,
            })

    for u1 in unmatched_r1:
        disagreements.append({
            "type": "only_r1",
            "span_text": u1.get("span_text", ""),
            "r1": u1, "r2": None,
        })

    for u2 in unmatched_r2:
        disagreements.append({
            "type": "only_r2",
            "span_text": u2.get("span_text", ""),
            "r1": None, "r2": u2,
        })

    return disagreements


def _rebuild_message_spans(row, disagreements, decisions_map,
                           corrections_map=None):
    """Rebuild final sitemo_units from agreed spans + decisions."""
    if corrections_map is None:
        corrections_map = {}
    pj_r1 = row.get("parsed_json_r1", {})
    pj_r2 = row.get("parsed_json_r2", {})
    if not isinstance(pj_r1, dict):
        pj_r1 = {}
    if not isinstance(pj_r2, dict):
        pj_r2 = {}

    units_r1 = pj_r1.get("sitemo_units", [])
    units_r2 = pj_r2.get("sitemo_units", [])
    if not isinstance(units_r1, list):
        units_r1 = []
    if not isinstance(units_r2, list):
        units_r2 = []

    matched, _, _ = _match_spans(units_r1, units_r2)

    final = []
    # Agreed matched spans → keep R1's version
    for u1, u2 in matched:
        if not _has_annotation_diff(u1, u2):
            final.append(copy.deepcopy(u1))

    # Apply disagreement decisions
    for i, dis in enumerate(disagreements):
        decision = decisions_map.get(i)
        corr = corrections_map.get(i)
        if decision == "Autre" and corr:
            # Build corrected unit from the span text
            base = dis.get("r1") or dis.get("r2") or {}
            for cat in corr.get("categories", []):
                unit = copy.deepcopy(base)
                unit["categorie"] = cat
                unit["categorie2"] = ""
                if corr.get("mode"):
                    unit["mode"] = corr["mode"]
                final.append(unit)
        elif decision == "R1" and dis.get("r1"):
            final.append(copy.deepcopy(dis["r1"]))
        elif decision == "R2" and dis.get("r2"):
            final.append(copy.deepcopy(dis["r2"]))
        elif decision == "Aucun":
            pass  # discard
        elif decision is None:
            # No response → default: keep R1, skip only_r2
            if dis["type"] in ("mismatch", "only_r1"):
                if dis.get("r1"):
                    final.append(copy.deepcopy(dis["r1"]))

    return final


# ═══ FORMATTING ════════════════════════════════════════════════════════════

MODES = ["Désignée", "Comportementale", "Suggérée", "Montrée"]


def _unit_md_block(label, unit):
    """Format one unit as a markdown block."""
    if unit is None:
        return f"**{label} :** *(non détecté)*"
    cat = unit.get("categorie", "—")
    cat2 = unit.get("categorie2", "")
    if cat2:
        cat = f"{cat} / {cat2}"
    mode = unit.get("mode", "—")
    justif = unit.get("justification", "—")
    return (
        f"**{label}** — Catégorie : **{cat}** · "
        f"Mode : **{mode}**  \n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;*{justif}*"
    )


def format_disagreement_md(dis):
    """Format a disagreement as compact markdown."""
    span = dis["span_text"]
    lines = [
        f"**Span :** `{span}`",
        "",
        _unit_md_block("R1", dis["r1"]),
        "",
        "---",
        "",
        _unit_md_block("R2", dis["r2"]),
    ]
    return "\n".join(lines)


# ═══ ARGILLA CONNECTION ════════════════════════════════════════════════════

def connect_argilla(api_url, api_key, proxy=None):
    kwargs = {}
    if proxy:
        kwargs["proxy"] = proxy
    return rg.Argilla(api_url=api_url, api_key=api_key, **kwargs)


# ═══ PUSH TO ARGILLA ══════════════════════════════════════════════════════

def push_to_argilla(merged, df_orig, args):
    client = connect_argilla(args.api_url, args.api_key, args.proxy)
    dataset_name = args.dataset
    workspace = args.workspace

    settings = rg.Settings(
        fields=[
            rg.TextField(name="message", title="Message",
                         use_markdown=True),
            rg.TextField(name="desaccord",
                         title="Désaccord inter-annotateurs",
                         use_markdown=True),
            rg.TextField(name="contexte",
                         title="Contexte (locuteur)",
                         use_markdown=True, required=False),
        ],
        questions=[
            rg.LabelQuestion(
                name="decision",
                title="Quelle annotation est correcte ?",
                description=(
                    "R1 / R2 = garder cette annotation, "
                    "Aucun = rejeter le span, "
                    "Autre = corriger manuellement"),
                labels=["R1", "R2", "Aucun", "Autre"],
                required=True,
            ),
            rg.MultiLabelQuestion(
                name="correction_categories",
                title="Correction : catégories",
                description=(
                    "Si 'Autre' : sélectionnez la ou les "
                    "catégories correctes pour ce span."),
                labels=EMOTIONS,
                required=False,
            ),
            rg.LabelQuestion(
                name="correction_mode",
                title="Correction : mode",
                description=(
                    "Si 'Autre' : sélectionnez le mode "
                    "correct pour ce span."),
                labels=MODES,
                required=False,
            ),
            rg.TextQuestion(
                name="notes",
                title="Notes",
                required=False,
            ),
        ],
        metadata=[
            rg.IntegerMetadataProperty(name="idx",
                                       title="Index message"),
            rg.IntegerMetadataProperty(
                name="n_disagreements",
                title="Nb désaccords (message)"),
            rg.TermsMetadataProperty(
                name="type_desaccord",
                title="Type de désaccord"),
        ],
    )

    # Check existing dataset
    existing = None
    try:
        existing = client.datasets(name=dataset_name,
                                   workspace=workspace)
    except Exception:
        pass

    if existing is not None:
        try:
            _ = existing.id
            if args.force:
                print(f"⚠ Suppression du dataset "
                      f"'{dataset_name}'...")
                existing.delete()
            else:
                print(f"⚠ Le dataset '{dataset_name}' existe.")
                print("  --force pour recréer, "
                      "--mode export pour exporter.")
                return
        except Exception:
            pass

    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace,
        settings=settings,
        client=client,
    )
    dataset.create()
    print(f"✓ Dataset '{dataset_name}' créé")

    # Build records — one per span disagreement
    records = []
    n_msg_with_dis = 0
    for i in range(len(merged)):
        row = merged.iloc[i]
        orig_idx = int(row["idx"])

        disagreements = compute_disagreements(row)
        if not disagreements:
            continue

        n_msg_with_dis += 1
        text = str(row.get("TEXT", ""))
        if not text or text == "nan":
            text = str(row.get("raw_text_r1", ""))
        name = str(row.get("NAME",
                           row.get("target_name_r1", "")))
        role = str(row.get("ROLE",
                           row.get("target_role_r1", "")))

        message_md = f"> {text}" if text else "> *(vide)*"
        contexte_md = (
            f"**Locuteur :** {name} | **Rôle :** {role}")

        for di, dis in enumerate(disagreements):
            desaccord_md = format_disagreement_md(dis)

            record = rg.Record(
                id=f"{orig_idx}_d{di}",
                fields={
                    "message": message_md,
                    "desaccord": desaccord_md,
                    "contexte": contexte_md,
                },
                metadata={
                    "idx": orig_idx,
                    "n_disagreements": len(disagreements),
                    "type_desaccord": dis["type"],
                },
            )
            records.append(record)

    dataset.records.log(records)
    print(f"✓ {len(records)} désaccords poussés "
          f"({n_msg_with_dis} messages concernés)")

    url = args.api_url
    webbrowser.open(url)
    print(f"Interface Argilla : {url}")


# ═══ EXPORT FROM ARGILLA ══════════════════════════════════════════════════

def export_from_argilla(merged, df_orig, args):
    client = connect_argilla(args.api_url, args.api_key, args.proxy)

    try:
        dataset = client.datasets(name=args.dataset,
                                  workspace=args.workspace)
        _ = dataset.id
    except Exception:
        print(f"⚠ Dataset '{args.dataset}' non trouvé.")
        return

    # Collect decisions: (orig_idx, disagreement_index) → decision
    record_decisions = {}
    for record in dataset.records:
        rid = record.id
        if not rid or "_d" not in rid:
            continue
        parts = rid.rsplit("_d", 1)
        try:
            orig_idx = int(parts[0])
            di = int(parts[1])
        except (ValueError, IndexError):
            continue

        decision = None
        correction = None
        if record.responses:
            user_resp = record.responses[-1]
            cats, mode = None, None
            for resp in user_resp.responses:
                if resp.question_name == "decision":
                    decision = resp.value
                elif resp.question_name == "correction_categories":
                    cats = resp.value
                elif resp.question_name == "correction_mode":
                    mode = resp.value
            if decision == "Autre" and cats:
                correction = {"categories": cats,
                              "mode": mode}
        record_decisions[(orig_idx, di)] = (
            decision, correction)

    # Build export rows
    rows_out = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        orig_idx = int(row["idx"])
        out = {"idx": orig_idx}

        if df_orig is not None and orig_idx < len(df_orig):
            for col in df_orig.columns:
                out[col] = df_orig.iloc[orig_idx][col]

        disagreements = compute_disagreements(row)
        decisions_map = {}
        has_any_response = False
        corrections_map = {}
        for di in range(len(disagreements)):
            entry = record_decisions.get((orig_idx, di))
            if entry is not None:
                dec, corr = entry
                if dec is not None:
                    has_any_response = True
                decisions_map[di] = dec
                if corr:
                    corrections_map[di] = corr
            else:
                decisions_map[di] = None

        final_spans = _rebuild_message_spans(
            row, disagreements, decisions_map,
            corrections_map)

        emo_dict = _aggregate_sitemo_to_emotions(final_spans)
        for e in EMOTIONS:
            out[e] = emo_dict.get(e, 0)

        out["reviewed"] = (has_any_response
                           or len(disagreements) == 0)
        out["n_spans"] = len(final_spans)
        out["spans_json"] = json.dumps(
            final_spans, ensure_ascii=False)

        for si, span in enumerate(final_spans[:5]):
            if isinstance(span, dict):
                out[f"span{si+1}_text"] = span.get(
                    "span_text", "")
                out[f"span{si+1}_cat"] = span.get(
                    "categorie", "")
                out[f"span{si+1}_mode"] = span.get(
                    "mode", "")

        for e in EMOTIONS:
            out[f"{e}_run1"] = (
                int(row[f"{e}_r1"])
                if pd.notna(row[f"{e}_r1"]) else None)
            out[f"{e}_run2"] = (
                int(row[f"{e}_r2"])
                if pd.notna(row[f"{e}_r2"]) else None)
        out["n_disagreements"] = len(disagreements)
        rows_out.append(out)

    df_out = pd.DataFrame(rows_out)

    lead = [c for c in ["idx"] if c in df_out.columns]
    if df_orig is not None:
        lead += [c for c in df_orig.columns
                 if c in df_out.columns and c not in lead]
    ordered = (
        lead + EMOTIONS
        + ["reviewed", "n_spans", "spans_json"]
        + [f"span{i}_text" for i in range(1, 6)]
        + [f"span{i}_cat"  for i in range(1, 6)]
        + [f"span{i}_mode" for i in range(1, 6)]
        + ["n_disagreements"]
        + [f"{e}_{s}" for e in EMOTIONS for s in ("run1", "run2")]
    )
    rest = [c for c in df_out.columns if c not in ordered]
    df_out = df_out[
        [c for c in ordered + rest if c in df_out.columns]]

    export_path = args.out_xlsx or os.path.join(
        os.path.dirname(os.path.abspath(args.run1)),
        "annotations_validees.xlsx")

    df_out.to_excel(export_path, index=False, engine="openpyxl")

    n_rev = sum(1 for r in rows_out if r.get("reviewed"))
    print(f"✓ Exporté → {export_path}")
    print(f"  {n_rev}/{len(merged)} messages avec décision")


# ═══ PARSE ARGS ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Supervision des annotations via Argilla")
    p.add_argument("--run1", required=True, help="JSONL du run 1")
    p.add_argument("--run2", required=True, help="JSONL du run 2")
    p.add_argument("--xlsx", default=None,
                   help="XLSX original (pour TEXT/NAME/ROLE)")
    p.add_argument("--api_url", required=True,
                   help="URL Argilla")
    p.add_argument("--api_key", required=True,
                   help="Clé API Argilla")
    p.add_argument("--dataset", default="supervision",
                   help="Nom du dataset (défaut: supervision)")
    p.add_argument("--workspace", default="argilla",
                   help="Workspace Argilla (défaut: argilla)")
    p.add_argument("--mode", choices=["push", "export"],
                   default="push",
                   help="push = envoyer, export = récupérer")
    p.add_argument("--force", action="store_true",
                   help="Recréer le dataset s'il existe")
    p.add_argument("--out_xlsx", default=None,
                   help="Fichier d'export XLSX (défaut: auto)")
    p.add_argument("--proxy", default=None,
                   help="Proxy HTTP")
    return p.parse_args()


# ═══ MAIN ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    # Auto-detect xlsx from run1 filename
    if not args.xlsx:
        base = os.path.basename(args.run1).lower()
        for part in base.replace(".jsonl", "").split("_"):
            candidate = os.path.join(REPO_ROOT, "data",
                                     f"{part}.xlsx")
            if os.path.exists(candidate):
                args.xlsx = candidate
                print(f"  XLSX auto-détecté : {args.xlsx}")
                break

    df_r1 = load_run(args.run1)
    df_r2 = load_run(args.run2)

    merged = pd.merge(
        df_r1, df_r2, on="idx", how="inner",
        suffixes=("_r1", "_r2"))
    merged = merged[
        merged["json_ok_r1"] & merged["json_ok_r2"]
    ].reset_index(drop=True)

    df_orig = None
    if args.xlsx and os.path.exists(args.xlsx):
        df_orig = pd.read_excel(args.xlsx).reset_index(drop=True)
        for col in ["TEXT", "NAME", "ROLE", "TIME"]:
            merged[col] = merged["idx"].apply(
                lambda i, c=col: str(df_orig.iloc[i].get(c, ""))
                if i < len(df_orig) else "")

    for e in EMOTIONS:
        merged[f"{e}_div"] = merged[f"{e}_r1"] != merged[f"{e}_r2"]
    merged["n_div"] = sum(
        merged[f"{e}_div"].astype(int) for e in EMOTIONS)

    N_TOTAL = len(merged)
    N_DIVERG = int((merged["n_div"] > 0).sum())
    print(f"✓ {N_TOTAL} messages comparables")
    print(f"  dont {N_DIVERG} avec ≥1 divergence (émotions)")

    if N_TOTAL == 0:
        print("⚠ Aucun message comparable")
        return

    if args.mode == "push":
        push_to_argilla(merged, df_orig, args)
    elif args.mode == "export":
        export_from_argilla(merged, df_orig, args)


if __name__ == "__main__":
    main()
