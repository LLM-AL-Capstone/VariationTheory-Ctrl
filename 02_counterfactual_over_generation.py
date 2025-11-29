# 02_counterfactual_over_generation.py — GPT-4o version
import os, sys, ast, json, pandas as pd, configparser
from llm_gpt4o import get_client, warmup, chat_call, MODEL_ID

cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
cfg.read('config.ini')
DATA_FILE = cfg.get("settings","data_file").split('#')[0].strip()
SEED = 42  # Fixed seed
ITERATION = int(os.getenv("ITERATION", "0"))

stem = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE
candidate_file = f"output_data/[{SEED}]{stem}_candidate_phrases_annotated_data_iter{ITERATION}.csv"
out_file = f"output_data/[{SEED}]counterfactuals_{stem}_iter{ITERATION}.csv"

def load_candidates(p):
    if not os.path.exists(p):
        print("ERROR: missing", p); sys.exit(0)
    df = pd.read_csv(p)
    if df.empty:
        print("INFO: empty candidates"); sys.exit(0)
    return df

def parse_phrases(raw, limit=5, max_chars=200):
    items = []
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, (list,tuple)):
                items = [str(x) for x in v]
            else:
                items = [x.strip() for x in raw.split(",") if x.strip()]
        except Exception:
            items = [x.strip() for x in raw.split(",") if x.strip()]
    elif isinstance(raw, (list,tuple)):
        items = [str(x) for x in raw]
    s = ", ".join(items[:limit])
    return s if len(s) <= max_chars else s[:max_chars-3] + "..."

def build_messages(text, label, target, phrases):
    return [
        {"role":"system","content":"Generate a new sentence close to the original using exactly one of the given phrases."},
        {"role":"user","content":(
            f"Change topic from {label} to {target}.\n"
            f"- Use ONE phrase verbatim: {phrases}\n"
            f"- The new sentence must NOT be about {label}.\n"
            f"- Do NOT include the literal word '{target}'.\n"
            f"- Keep grammar correct; minimal edits."
        )},
        {"role":"user","content":f"original text: {text}\nmodified:"}
    ]

def main():
    print(f"LOG: Step 02 (GPT-4o) iteration {ITERATION} —", candidate_file)
    df = load_candidates(candidate_file)

    client = get_client(timeout=600.0)
    warmup(client)

    rows = []
    for idx, r in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(df)}")
        phrases = parse_phrases(r.get("candidate_phrases",""))
        msg = build_messages(
            r.get("ori_text",""),
            r.get("ori_label",""),
            r.get("target_label",""),
            phrases
        )

        try:
            out = chat_call(
                client, msg, model=MODEL_ID, max_tokens=256, temperature=0.0
            )
        except Exception as e:
            print("ERROR row", idx, e)
            out = ""

        rows.append([
            r.get("id", idx), r.get("ori_text",""), r.get("ori_label",""),
            r.get("pattern",""), r.get("highlight",""), phrases,
            r.get("target_label",""), out
        ])

    pd.DataFrame(rows, columns=[
        "id","ori_text","ori_label","pattern","highlight","candidate_phrases","target_label","counterfactual"
    ]).to_csv(out_file, index=False)
    print("LOG: wrote →", out_file)

if __name__ == "__main__":
    main()