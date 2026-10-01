#!/usr/bin/env python3
"""Build the MedWeb retrieval files used by the paper from the NTCIR-13 MedWeb data.

usage: python build_medweb_csv.py --medweb_dir <ntcir13_MedWeb_TestCollection> [--out_dir <dir>]

Input: the six files of the NTCIR-13 MedWeb task (obtained under the NTCIR data
agreement), NTCIR-13_MedWeb_{en,ja,zh}_{training,test}.xlsx. CSV exports of the
same sheets are accepted too (any file whose name starts with the xlsx name and
ends in .csv, e.g. "NTCIR-13_MedWeb_en_training.xlsx - en_train.csv").

Output: medweb_rag_{en,ja,zh}_fixed.csv with columns document_id, ReviewTitle,
ReviewText, and medweb_rag_all_fixed.csv (all three). Training and test tweets
are pooled (2,560 per language). ReviewTitle is the label string of the tweet's
positive symptom labels, which is both the query and the prefix of the indexed
document text:

    en  "Patient symptoms: Hayfever, Runnynose."     no label: "Routine health status."
    ja  "臨床症状： 花粉症, 鼻水・鼻づまり。"           no label: "特定の症状なし。"
    zh  "临床症状： 花粉症, 鼻涕。"                    no label: "无特定症状。"

Every tweet with the same label string is relevant to that query
(rag_dataload.UniversalRAGLoader.load_medweb and title_groups() in
camera_ready_extras.py).
"""
import argparse
import glob
import os

import pandas as pd

SYMPTOMS = ["Influenza", "Diarrhea", "Hayfever", "Cough", "Headache", "Fever", "Runnynose", "Cold"]

# Label names per language, following the NTCIR-13 MedWeb task
SYMPTOM_MAP = {
    "ja": {"Influenza": "インフルエンザ", "Diarrhea": "下痢", "Hayfever": "花粉症", "Cough": "咳・たん",
           "Headache": "頭痛", "Fever": "熱", "Runnynose": "鼻水・鼻づまり", "Cold": "風邪"},
    "zh": {"Influenza": "流感", "Diarrhea": "腹泻", "Hayfever": "花粉症", "Cough": "咳嗽",
           "Headache": "头痛", "Fever": "发烧", "Runnynose": "鼻涕", "Cold": "感冒"},
    "en": {s: s for s in SYMPTOMS},
}
NO_SYMPTOM = {"en": "Routine health status.", "ja": "特定の症状なし。", "zh": "无特定症状。"}
PREFIX = {"en": "Patient symptoms: ", "ja": "臨床症状： ", "zh": "临床症状： "}


def read_sheet(medweb_dir, stem):
    """Read <stem>.xlsx, or a CSV export of it."""
    csvs = sorted(glob.glob(os.path.join(medweb_dir, f"{stem}.xlsx*.csv")))
    if csvs:
        return pd.read_csv(csvs[0], encoding="utf-8-sig", engine="python")
    book = pd.ExcelFile(os.path.join(medweb_dir, f"{stem}.xlsx"))
    sheet = [n for n in book.sheet_names if n.lower() != "readme"][0]   # e.g. en_train
    return book.parse(sheet)


def label_string(row, lang):
    active = [SYMPTOM_MAP[lang][s] for s in SYMPTOMS
              if s in row and str(row[s]).strip().lower() == "p"]
    if not active:
        return NO_SYMPTOM[lang]
    suffix = "。" if lang in ("ja", "zh") else "."
    return f"{PREFIX[lang]}{', '.join(active)}{suffix}"


def build(medweb_dir, lang):
    df = pd.concat([read_sheet(medweb_dir, f"NTCIR-13_MedWeb_{lang}_training"),
                    read_sheet(medweb_dir, f"NTCIR-13_MedWeb_{lang}_test")], ignore_index=True)
    id_col = [c for c in df.columns if "ID" in str(c).upper()][0]
    tweet_col = [c for c in df.columns if "TWEET" in str(c).upper()][0]
    out = pd.DataFrame()
    out["document_id"] = df[id_col]
    out["ReviewTitle"] = df.apply(lambda r: label_string(r, lang), axis=1)
    out["ReviewText"] = df[tweet_col]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--medweb_dir", required=True, help="folder with the six NTCIR-13 MedWeb files")
    ap.add_argument("--out_dir", default=None, help="output folder (default: --medweb_dir)")
    args = ap.parse_args()
    out_dir = args.out_dir or args.medweb_dir
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for lang in ("en", "ja", "zh"):
        df = build(args.medweb_dir, lang)
        path = os.path.join(out_dir, f"medweb_rag_{lang}_fixed.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{lang}: {len(df)} tweets, {df['ReviewTitle'].nunique()} label strings -> {path}")
        frames.append(df)
    allp = os.path.join(out_dir, "medweb_rag_all_fixed.csv")
    pd.concat(frames, ignore_index=True).to_csv(allp, index=False, encoding="utf-8-sig")
    print(f"all: -> {allp}")


if __name__ == "__main__":
    main()
