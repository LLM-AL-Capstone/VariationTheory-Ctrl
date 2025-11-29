# 03_counterfactual_filtering.py — GPT-4o version
import os, sys, re, json, configparser
import pandas as pd
import spacy
from spacy.matcher import Matcher

from PaTAT_piped.helpers import expand_working_list, get_similarity_dict
from PaTAT_piped.api_helper import APIHelper
from llm_gpt4o import get_client, chat_call, warmup, MODEL_ID

cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
cfg.read('config.ini')

DATA_FILE = cfg.get("settings","data_file").split('#')[0].strip()
SEED = 42  # Fixed seed
ITERATION = int(os.getenv("ITERATION", "0"))

STEM = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE

nlp = spacy.load("en_core_web_sm")

def _yes(tok: str) -> bool:
    t = (tok or "").strip().lower()
    return t in {"yes","y","true","ok"}

def parse_yes_no(resp: str):
    s = (resp or "").strip()
    s = re.sub(r'[\|/;]+', ',', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\b(yes|no)\b\s+\b(yes|no)\b', r'\1,\2', s, flags=re.I)
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) < 2:
        toks = re.findall(r'\b(yes|no)\b', s, flags=re.I)
        toks += ['no','no']
        parts = toks[:2]
    return parts[0], parts[1]

def heuristic_filtering(df: pd.DataFrame) -> pd.DataFrame:
    ok = []
    for _, row in df.iterrows():
        t = str(row.get('counterfactual','') or '')
        meta_bad = re.search(r"\b(assistant|as an ai|cannot comply|i can not|i'm unable|unable to)\b", t, re.I)
        constraint_bad = re.search(r"given the constraints|cannot follow|not allowed|policy", t, re.I)
        ok.append(not bool(meta_bad or constraint_bad))
    df['heuristic_filtered'] = pd.Series(ok, index=df.index).astype(bool)
    path = f'output_data/interim_output/[{SEED}]heuristic_filtered_{STEM}_iter{ITERATION}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

def check_matching(sent: str, working_list) -> bool:
    if not working_list: return False
    matcher = Matcher(nlp.vocab)
    added = 0
    for i, pat in enumerate(working_list):
        try:
            matcher.add(f"rule{i}", [pat])
            added += 1
        except Exception:
            continue
    if added == 0: return False
    doc = nlp(str(sent))
    for _, s, e in matcher(doc):
        if str(doc[s:e]).strip(): return True
    return False

def symbolic_filtering(df: pd.DataFrame, similarity_dict) -> pd.DataFrame:
    matched, t, f = [], 0, 0
    for _, row in df.iterrows():
        try:
            wl = expand_working_list(row['pattern'], soft_match_on=True, similarity_dict=similarity_dict)
        except Exception:
            wl = []
        ok = check_matching(row.get('counterfactual',''), wl)
        matched.append(ok); t += int(ok); f += int(not ok)
    df['matched_pattern'] = pd.Series(matched, index=df.index).astype(bool)

    keep_rate = t / max(1, (t+f))
    print(f"Symbolic keep rate: {keep_rate:.3f}")

    path = f'output_data/interim_output/[{SEED}]symbolic_filtered_{STEM}_iter{ITERATION}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df

def gpt_discriminator_filtering(df: pd.DataFrame) -> pd.DataFrame:
    labels = df['ori_label'].unique()
    client = get_client(timeout=300.0); warmup(client)

    if '_row' not in df.columns:
        df['_row'] = range(len(df))

    passed_mask = df['heuristic_filtered'].astype(bool) & df['matched_pattern'].astype(bool)
    df_pass = df.loc[passed_mask].copy()

    print(f"Passing to discriminator: {len(df_pass)} rows")

    if len(df_pass) == 0:
        if 'is_ori' not in df.columns:    df['is_ori'] = False
        if 'is_target' not in df.columns: df['is_target'] = False
        return _summarize_and_write(df)

    out_rows = []
    for _, row in df_pass.iterrows():
        orig_row_id = int(row['_row'])

        msgs = [
            {"role":"system","content":"Two-label judgement. Reply exactly 'YES, NO' for (original, target)."},
            {"role":"system","content":f"Labels: {list(labels)}"},
            {"role":"user","content":f"{row['counterfactual']}\nLabel: {row['ori_label']}, {row['target_label']}"},
        ]

        try:
            resp = chat_call(client, msgs, model=MODEL_ID, max_tokens=8, temperature=0)
        except Exception as e:
            print("ERROR discrim row", orig_row_id, e)
            resp = "NO, NO"

        r_ori, r_tar = parse_yes_no(resp)
        out_rows.append({
            "_row": orig_row_id,
            "is_ori_disc": _yes(r_ori),
            "is_target_disc": _yes(r_tar),
        })

    res_df = pd.DataFrame(out_rows)
    df = df.merge(res_df, on="_row", how="left")

    if 'is_ori' not in df.columns:    df['is_ori'] = False
    if 'is_target' not in df.columns: df['is_target'] = False

    if 'is_ori_disc' in df.columns:
        df['is_ori'] = df['is_ori_disc'].fillna(df['is_ori']).astype(bool)
        df.drop(columns=['is_ori_disc'], inplace=True)
    else:
        df['is_ori'] = df['is_ori'].astype(bool)

    if 'is_target_disc' in df.columns:
        df['is_target'] = df['is_target_disc'].fillna(df['is_target']).astype(bool)
        df.drop(columns=['is_target_disc'], inplace=True)
    else:
        df['is_target'] = df['is_target'].astype(bool)

    return _summarize_and_write(df)

def _summarize_and_write(df: pd.DataFrame) -> pd.DataFrame:
    out = f'output_data/[{SEED}]filtered_{STEM}_iter{ITERATION}.csv'
    df.to_csv(out, index=False)

    print("=== Filtering summary ===")
    print("Total rows:                 ", len(df))
    print("Heuristic kept:             ", int(df.get('heuristic_filtered', pd.Series(dtype=bool)).sum() if 'heuristic_filtered' in df else 0))
    print("Symbolic matched:           ", int(df.get('matched_pattern', pd.Series(dtype=bool)).sum() if 'matched_pattern' in df else 0))
    print("is_ori=True count:          ", int(df['is_ori'].sum() if 'is_ori' in df else 0))
    print("is_target=True count:       ", int(df['is_target'].sum() if 'is_target' in df else 0))
    chosen_mask = (~df['is_ori'].astype(bool)) & (df['is_target'].astype(bool)) if {'is_ori','is_target'} <= set(df.columns) else pd.Series([], dtype=bool)
    print("Chosen (NO, YES) rows:      ", int(chosen_mask.sum()) if len(chosen_mask) else 0)
    print("Wrote filtered CSV to:      ", out)
    return df

if __name__ == "__main__":
    file = f"output_data/[{SEED}]counterfactuals_{STEM}_iter{ITERATION}.csv"

    print(f"Reading file {file}")
    try:
        df = pd.read_csv(file)
    except Exception:
        sys.exit(f"ERROR: cannot read {file}")

    if '_row' not in df.columns:
        df['_row'] = range(len(df))

    src = f"input_data/{DATA_FILE}"
    try:
        data = pd.read_csv(src)
    except Exception:
        sys.exit(f"ERROR: cannot read {src}")

    if 'example' not in data.columns:
        if 'Text' in data.columns: data['example'] = data['Text']
        else:
            fc = data.columns[1] if len(data.columns)>1 else data.columns[0]
            data['example'] = data[fc].astype(str)
    data['example'] = data['example'].astype(str)

    df_patat = pd.DataFrame({
        "id": range(len(data)),
        "example": data['example'],
        "Label": data['Label'] if 'Label' in data.columns else ['none']*len(data),
    })
    patat = APIHelper(user=f"{STEM}_s{SEED}")
    patat.theme = STEM
    patat.data  = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data["example"].values,
        soft_threshold=patat.soft_threshold, soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk, file_name=patat.theme
    )
    sim = patat.similarity_dict

    df = heuristic_filtering(df)
    df = symbolic_filtering(df, sim)
    gpt_discriminator_filtering(df)