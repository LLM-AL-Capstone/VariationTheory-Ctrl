# 04_fine_tuning.py — GPT-4o version with iteration support
import os, sys, json, pandas as pd, configparser
from llm_gpt4o import get_client, warmup, chat_call, MODEL_ID

cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
cfg.read('config.ini')

DATA_FILE = cfg.get("settings", "data_file").split('#')[0].strip()
SEED = 42  # Fixed seed
ITERATION = int(os.getenv("ITERATION", "0"))

stem = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE

in_path = f"output_data/[{SEED}]{stem}_candidate_phrases_annotated_data_iter{ITERATION}.csv"
out_path = f"output_data/[{SEED}]tuned_counterfactuals_{stem}_iter{ITERATION}.csv"


def build_messages(text, label, target, phrases):
    return [
        {"role": "system", "content": "Modify the sentence to change topic using exactly one provided phrase."},
        {"role": "user", "content": (
            f"Original: \"{text}\"\nOriginal label: {label}\nTarget label: {target}\n"
            f"Candidate phrases: {phrases}\nRules: use one phrase verbatim; do not use the literal target word; "
            f"keep grammar; avoid contractions; minimal edits.\nModified:"
        )}
    ]


def main():
    print(f"INFO: Step 04 - Iteration {ITERATION}")

    if not os.path.exists(in_path):
        print(f"INFO: No candidate file for iteration {ITERATION}; skipping.")
        return

    df = pd.read_csv(in_path)

    if df.empty:
        print(f"INFO: Empty candidate file for iteration {ITERATION}; skipping.")
        return

    client = get_client(timeout=600.0)
    warmup(client)

    rows = []
    for i, r in df.iterrows():
        if i % 10 == 0:
            print(f"Processing {i}/{len(df)}")

        msgs = build_messages(
            r["ori_text"],
            r["ori_label"],
            r["target_label"],
            r["candidate_phrases"]
        )

        try:
            out = chat_call(client, msgs, model=MODEL_ID, max_tokens=48, temperature=0)
        except Exception as e:
            print(f"ERROR at row {i}: {e}")
            out = ""

        rows.append({
            "id": r["id"],
            "ori_text": r["ori_text"],
            "ori_label": r["ori_label"],
            "pattern": r["pattern"],
            "highlight": r["highlight"],
            "candidate_phrases": r["candidate_phrases"],
            "target_label": r["target_label"],
            "counterfactual": out
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote → {out_path}")


if __name__ == "__main__":
    main()