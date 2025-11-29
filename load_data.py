# pip install datasets pandas scikit-learn

from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np

# ---------------- Configuration ----------------
SEED = 2025
TOTAL = 540          # total examples per dataset (balanced)
TEST_RATIO = 0.20    # 80/20 train/test split
OUTPUT_DIR = Path("/Users/daphnemanning/Code/VariationTheory/input_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------

def balanced_subset(df, label_col, total, seed=SEED):
    """
    Return a balanced subset DataFrame of size `total` with columns: example, <label_col>.
    label_col can be 'Label' (capital L).
    """
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    labels = sorted(df[label_col].unique())
    k = len(labels)
    per = total // k

    # Equal per-class sampling
    parts = []
    for y in labels:
        grp = df[df[label_col] == y].sample(frac=1, random_state=seed).reset_index(drop=True)
        parts.append(grp.iloc[:per])
    out = pd.concat(parts, ignore_index=True)

    # Round-robin top-up to reach TOTAL
    cursors = {y: per for y in labels}
    pools = {
        y: df[df[label_col] == y].sample(frac=1, random_state=seed).reset_index(drop=True)
        for y in labels
    }
    while len(out) < total:
        for y in labels:
            out = pd.concat([out, pools[y].iloc[cursors[y]:cursors[y] + 1]], ignore_index=True)
            cursors[y] += 1
            if len(out) == total:
                break

    return out[["example", label_col]]

def write_split_files(df, label_col, base_name, outdir=OUTPUT_DIR,
                      test_ratio=TEST_RATIO, seed=SEED):
    """
    Stratified split and write two flat CSVs:
      <base_name>_data.csv and <base_name>_test.csv
    Columns: id, example, Label (or id, example, <label_col> generically).
    """
    train_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df[label_col], random_state=seed
    )
    train_df = train_df.reset_index(drop=True).rename_axis("id").reset_index()
    test_df = test_df.reset_index(drop=True).rename_axis("id").reset_index()

    outdir.joinpath(f"{base_name}_data.csv").write_text(
        train_df[["id", "example", label_col]].to_csv(index=False)
    )
    outdir.joinpath(f"{base_name}_test.csv").write_text(
        test_df[["id", "example", label_col]].to_csv(index=False)
    )
    print(f"{base_name}: saved {base_name}_data.csv ({len(train_df)}) and {base_name}_test.csv ({len(test_df)})")

# ------------------- AG NEWS -------------------
ag = load_dataset("ag_news", split="train")
label_names_ag = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
ag_df = pd.DataFrame({
    "example": ag["text"],                           # HF column 'text'
    "Label":   [label_names_ag[i] for i in ag["label"]],  # HF column 'label' -> our 'Label'
})
ag_bal = balanced_subset(ag_df, "Label", TOTAL, seed=SEED)
write_split_files(ag_bal, "Label", base_name="ag_news")

# ------------------- DBPEDIA 14 -------------------
dbp = load_dataset("dbpedia_14", split="train")
# 14 ontology classes
label_names_dbp = {
    0:  "Company",
    1:  "EducationalInstitution",
    2:  "Artist",
    3:  "Athlete",
    4:  "OfficeHolder",
    5:  "MeanOfTransportation",
    6:  "Building",
    7:  "NaturalPlace",
    8:  "Village",
    9:  "Animal",
    10: "Plant",
    11: "Album",
    12: "Film",
    13: "WrittenWork",
}
dbp_df = pd.DataFrame({
    # Combine title + content into a single text field
    "example": [f"{t}. {c}" for t, c in zip(dbp["title"], dbp["content"])],
    "Label":   [label_names_dbp[i] for i in dbp["label"]],
})
dbp_bal = balanced_subset(dbp_df, "Label", TOTAL, seed=SEED)
write_split_files(dbp_bal, "Label", base_name="dbpedia")

# ------------------- MNLI (GLUE) -------------------
mnli = load_dataset("glue", "mnli", split="train")
label_names_mnli = {0: "entailment", 1: "neutral", 2: "contradiction"}
mnli_df = pd.DataFrame({
    "example": [
        f"Premise: {p}\nHypothesis: {h}"
        for p, h in zip(mnli["premise"], mnli["hypothesis"])
    ],
    "Label":   [label_names_mnli[i] for i in mnli["label"]],   # HF 'label'
})
mnli_bal = balanced_subset(mnli_df, "Label", TOTAL, seed=SEED)
write_split_files(mnli_bal, "Label", base_name="mnli")

# ------------------- Amazon Polarity -------------------
amz = load_dataset("amazon_polarity", split="train")
label_names_amz = {0: "negative", 1: "positive"}
amz_df = pd.DataFrame({
    "example": amz["content"],
    "Label":   [label_names_amz[i] for i in amz["label"]],    # HF 'label'
})
amz_bal = balanced_subset(amz_df, "Label", TOTAL, seed=SEED)
write_split_files(amz_bal, "Label", base_name="amazon_polarity")

# ------------------- TREC (coarse) -------------------
trec_train = load_dataset("trec", split="train", trust_remote_code=True)
label_names_trec = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]
trec_df = pd.DataFrame({
    "example": trec_train["text"],
    "Label":   [label_names_trec[i] for i in trec_train["coarse_label"]],
})
trec_bal = balanced_subset(trec_df, "Label", TOTAL, seed=SEED)
write_split_files(trec_bal, "Label", base_name="trec")

# ------------------- WANLI (binary NLI) -------------------
def load_wanli() -> dict:
    return load_dataset("alisawuffles/WANLI", revision="refs/convert/parquet")

def preprocess_wanli(ds_dict) -> pd.DataFrame:
    """
    Return DataFrame with columns: example, Label, _split
    Label in {'entailment', 'not_entailment'}.
    """
    frames = []
    for split_name, split in ds_dict.items():
        df = pd.DataFrame(split.to_pandas()[["premise", "hypothesis", "gold"]])
        lab = df["gold"].astype(str).str.lower()
        lab = lab.where(lab.eq("entailment"), "not_entailment")
        frames.append(pd.DataFrame({
            "example": [
                f"Premise: {p}\nHypothesis: {h}"
                for p, h in zip(df["premise"], df["hypothesis"])
            ],
            "Label":   lab,
            "_split":  split_name,
        }))
    return pd.concat(frames, ignore_index=True)

def wanli_balanced_540(df_all: pd.DataFrame, per_label=270, seed=SEED) -> pd.DataFrame:
    prefer = ["train", "validation", "test"]
    chosen = None
    for s in prefer:
        if (df_all["_split"] == s).any():
            chosen = df_all[df_all["_split"] == s]
            break
    if chosen is None:
        chosen = df_all

    rng = np.random.RandomState(seed)
    buckets = []
    for lab in ["entailment", "not_entailment"]:
        bucket = chosen[chosen["Label"] == lab]
        k = min(per_label, len(bucket))
        if k < per_label:
            print(f"[warn] WANLI: requested {per_label} of '{lab}' but only {len(bucket)} available; taking all.")
        buckets.append(bucket.sample(n=k, random_state=rng))
    sub = (
        pd.concat(buckets, ignore_index=True)
        .sample(frac=1.0, random_state=rng)
        .reset_index(drop=True)
    )
    return sub[["example", "Label"]]

wanli_raw = load_wanli()
wanli_df_all = preprocess_wanli(wanli_raw)
wanli_bal = wanli_balanced_540(wanli_df_all, per_label=270, seed=SEED)
write_split_files(wanli_bal, "Label", base_name="wanli")

# =================== NEW: SA / NLI (CAD) + ANLI ===================

# ---- SA: Counterfactually-augmented IMDb (Kaushik et al., 2020)
sa = load_dataset("tasksource/counterfactually-augmented-imdb")
sa_all = pd.concat(
    [pd.DataFrame(sa[s].to_pandas()) for s in sa.keys()],
    ignore_index=True,
)
sa_all = sa_all.rename(columns={"Text": "example", "Sentiment": "Label"})
sa_all["Label"] = sa_all["Label"].astype(str).str.lower()
sa_bal = balanced_subset(sa_all[["example", "Label"]], "Label", TOTAL, seed=SEED)
write_split_files(sa_bal, "Label", base_name="sa_cad")

# ---- NLI: Counterfactually-augmented SNLI (Kaushik et al., 2020)
nli_cf = load_dataset("sagnikrayc/snli-cf-kaushik")
nli_all = pd.concat(
    [pd.DataFrame(nli_cf[s].to_pandas()) for s in nli_cf.keys()],
    ignore_index=True,
)
nli_df = pd.DataFrame({
    "example": [
        f"Premise: {p}\nHypothesis: {h}"
        for p, h in zip(nli_all["premise"], nli_all["hypothesis"])
    ],
    "Label":   nli_all["label"].astype(str).str.lower(),  # HF 'label' → 'Label'
})
nli_bal = balanced_subset(nli_df, "Label", TOTAL, seed=SEED)
write_split_files(nli_bal, "Label", base_name="nli_cad")

# ---- ANLI (Adversarial NLI; Nie et al., 2019)
anli = load_dataset("facebook/anli")  # splits: train_r1, train_r2, train_r3, etc.

def collect_anli(ds):
    frames = []
    for split_name, split in ds.items():
        if not split_name.startswith("train"):
            continue
        fr = pd.DataFrame(split.to_pandas()[["premise", "hypothesis", "label"]])
        labmap = {0: "entailment", 1: "neutral", 2: "contradiction"}
        frames.append(pd.DataFrame({
            "example": [
                f"Premise: {p}\nHypothesis: {h}"
                for p, h in zip(fr["premise"], fr["hypothesis"])
            ],
            "Label":   [labmap[int(x)] for x in fr["label"]],  # HF 'label' → our 'Label'
        }))
    return pd.concat(frames, ignore_index=True)

anli_all = collect_anli(anli)
anli_bal = balanced_subset(anli_all, "Label", TOTAL, seed=SEED)
write_split_files(anli_bal, "Label", base_name="anli")

print("\n✅ All datasets written to 'input_data/' as flat CSVs with columns: id, example, Label")
