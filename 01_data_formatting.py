# 01_data_formatting.py — GPT-4o version with dataset-specific sample IDs
import sys, os, re, ast, configparser, pandas as pd
from pandas.api.types import is_integer_dtype

from PaTAT_piped.api_helper import APIHelper
from PaTAT_piped.helpers import get_similarity_dict
from llm_gpt4o import get_client, warmup, chat_call, MODEL_ID
from sample_config import get_cumulative_sample_ids, get_dataset_info, detect_dataset_name

# ------------------ config ------------------
config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
config.read('config.ini')

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
SEED = 42  # Fixed seed
ITERATION = int(os.getenv("ITERATION", "0"))

if not DATA_FILE:
    print("ERROR: No data file in config.ini");
    sys.exit(1)

os.makedirs("output_data", exist_ok=True)
stem = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE

# Detect dataset name from filename
DATASET_NAME = detect_dataset_name(DATA_FILE)
print(f"INFO: Detected dataset: {DATASET_NAME}")

try:
    dataset_info = get_dataset_info(DATASET_NAME)
    print(f"INFO: Dataset stats - Seed samples: {dataset_info['seed_samples']}, "
          f"Total iterations: {dataset_info['total_iterations']}, "
          f"Total samples: {dataset_info['total_samples']}")
except ValueError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def _read_csv_or_die(path):
    try:
        return pd.read_csv(path)
    except Exception:
        print(f"ERROR: cannot read file {path}");
        sys.exit(1)


def _truncate(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[: n - 3] + "..."


def normalize_id(val):
    """Normalize ID values for comparison - handle both string and numeric IDs"""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    # Try to convert to int if it looks numeric
    try:
        return int(val_str)
    except ValueError:
        return val_str


def get_patat_patterns():
    """Creates annotations for specific sample IDs"""
    print("INFO: Executing data formatting with Python:", sys.executable)
    print(f"INFO: Processing iteration {ITERATION} for dataset {DATASET_NAME}")

    file_path = f"input_data/{DATA_FILE}"
    df = _read_csv_or_die(file_path)

    # Normalize cols
    if 'example' not in df.columns:
        if 'Text' in df.columns:
            df['example'] = df['Text']
        else:
            print("ERROR: Dataset needs an 'example' or 'Text' column.");
            sys.exit(1)
    df['example'] = df['example'].astype(str).str.strip()

    # Store original IDs before any remapping
    if 'id' in df.columns:
        df['orig_id'] = df['id']

    # Normalize existing IDs
    if 'id' in df.columns:
        df['id'] = df['id'].apply(normalize_id)
    else:
        # Create integer index if no ID column
        df['id'] = range(len(df))
        df['orig_id'] = df['id']

    # Create a lookup dict
    id_to_orig = dict(zip(df['id'], df['orig_id']))

    labels = sorted(df["Label"].dropna().unique().tolist())
    if not labels:
        print("ERROR: no labels found in 'Label' column");
        sys.exit(1)

    # Get specific sample IDs for this iteration from config
    try:
        annotated_ids_raw = get_cumulative_sample_ids(DATASET_NAME, ITERATION)
        # Normalize the IDs from config
        annotated_ids = [normalize_id(x) for x in annotated_ids_raw]
        print(f"INFO: Using {len(annotated_ids)} specific samples for iteration {ITERATION}")
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Build PaTAT workspace
    df_patat = df[['id', 'example', 'Label']].copy()
    patat = APIHelper(user=f"{stem}_s{SEED}")
    patat.theme = stem
    patat.data = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data['example'].values,
        soft_threshold=patat.soft_threshold,
        soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk,
        file_name=patat.theme
    )

    col_names = ["id", "ori_text", "ori_label", "pattern", "highlight"]
    collected = []

    # Label specific examples - with detailed logging
    labeled_count = 0
    missing_ids = []
    for sid in annotated_ids:
        if sid not in df['id'].values:
            missing_ids.append(sid)
            continue

        lbl = df.loc[df['id'] == sid, 'Label'].values[0]
        patat.label_element(int(df.loc[df['id'] == sid].index[0]), lbl)  # Use dataframe index, not id
        labeled_count += 1

    print(f"INFO: Successfully labeled {labeled_count}/{len(annotated_ids)} samples")
    if missing_ids:
        print(
            f"WARN: {len(missing_ids)} IDs not found in dataset: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''}")

    # Ask PaTAT per label
    for label in labels:
        try:
            patat.set_theme(label)
            results = patat.get_linear_model_results()
        except KeyError:
            print(f"SKIP: no synthesizer for theme '{label}' yet")
            continue
        if "explanation" not in results or not results['explanation']:
            print(f"WARN: no explanations for '{label}'")
            continue

        explanations = results['explanation']
        for pattern_explanation, sent_map in explanations.items():
            for sentence_id, highlight in sent_map.items():
                row = df.loc[df.index == sentence_id]  # Use dataframe index
                if row.empty or row.iloc[0]['Label'] != label:
                    continue
                ori_text = row.iloc[0]['example']
                out_id = row.iloc[0]['orig_id']  # Use original ID for output
                collected.append([out_id, ori_text, label, pattern_explanation, highlight])

    out_df = pd.DataFrame(collected, columns=col_names)
    out_path = f"output_data/[{SEED}]annotated_data_with_pattern_{stem}_iter{ITERATION}.csv"
    out_df.to_csv(out_path, index=False)
    print("DEBUG: annotated rows written →", out_path, "| rows:", len(out_df))

    return out_path


def get_candidate_phrases():
    file_path = f"input_data/{DATA_FILE}"
    df = _read_csv_or_die(file_path)

    if 'example' not in df.columns:
        if 'Text' in df.columns:
            df['example'] = df['Text']
        else:
            print("ERROR: Dataset needs an 'example' or 'Text' column.")
            sys.exit(1)

    # Store original IDs
    if 'id' in df.columns:
        df['orig_id'] = df['id']
        df['id'] = df['id'].apply(normalize_id)
    else:
        df['id'] = range(len(df))
        df['orig_id'] = df['id']

    df['example'] = df['example'].astype(str).str.strip()
    unique_labels = df["Label"].unique()

    # PaTAT setup
    df_patat = df[['id', 'example', 'Label']].copy()
    patat = APIHelper(user=f"{stem}_s{SEED}")
    patat.theme = stem
    patat.data = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data['example'].values,
        soft_threshold=patat.soft_threshold,
        soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk,
        file_name=patat.theme
    )

    # Read the annotated file
    ann_path = f"output_data/[{SEED}]annotated_data_with_pattern_{stem}_iter{ITERATION}.csv"
    try:
        ann = pd.read_csv(ann_path)
        # Normalize IDs in annotation file too
        ann['id'] = ann['id'].apply(normalize_id)
    except Exception:
        print("ERROR: can not read annotated data file:", ann_path)
        sys.exit(1)

    # GPT-4o client
    client = get_client(timeout=600.0)
    warmup(client)

    col_names_2 = ["id", "ori_text", "ori_label", "pattern", "highlight", "target_label", "candidate_phrases"]
    data_collector_2 = []

    for _, row in ann.iterrows():
        sent_id = row['id']
        sentence = str(row['ori_text'])
        pattern = str(row['pattern'])
        label = str(row['ori_label'])
        highlight_str = row['highlight']

        # parse highlight spans
        try:
            hl = ast.literal_eval(highlight_str)
            marked_phrases = [" ".join(h[0]) for h in hl if isinstance(h, (list, tuple)) and h]
            if not marked_phrases:
                continue
        except Exception:
            continue

        for matched_phrase in marked_phrases:
            # collect softmatch words
            softmatch_collector = {}
            matches = re.findall(r'\(([^)]+)\)', pattern)
            if matches:
                for m in matches:
                    if m in patat.similarity_dict:
                        words = list(patat.similarity_dict[m].keys()) + [m]
                        softmatch_collector[m] = words

            for target_label in unique_labels:
                if target_label == label or target_label == 'none':
                    continue

                examples = [
                    ("Too many other places to shop with better prices .",
                     "price", "(price)+*", "service",
                     "purchase options, pricey service, cheap help, pricing plans, cost breakdown"),
                    ("they have great produce and seafood",
                     "products", "[seafood]|NOUN", "service",
                     "hospitality, seafood, help, management, staff"),
                ]

                messages = [
                    {"role": "system", "content":
                        "Create diverse candidate phrases that match the provided symbolic pattern "
                        "and could make the sentence belong to the target label. "
                        "Use commas between phrases. Use only allowed soft-match words if provided."}
                ]
                for ex_text, ex_lbl, ex_pat, ex_tgt, ex_out in examples:
                    messages += [
                        {"role": "user", "content":
                            f"sentence:'{ex_text}', phrase to modify:'', pattern:'{ex_pat}', "
                            f"current label:{ex_lbl}, target label:{ex_tgt}"},
                        {"role": "assistant", "content": ex_out}
                    ]

                req = (
                    f"sentence:'{_truncate(sentence, 350)}', "
                    f"phrase to modify:'{_truncate(matched_phrase, 60)}', "
                    f"pattern:'{pattern}', current label:{label}, "
                )
                if softmatch_collector:
                    req += f"softmatch:{softmatch_collector}, "
                req += f"target label:{target_label}"
                messages.append({"role": "user", "content": req})

                try:
                    data = chat_call(client, messages, model=MODEL_ID, max_tokens=96, temperature=0)
                except Exception as e:
                    print("WARN: LLM failed for id", sent_id, ":", e)
                    continue

                generated_phrases = [x.strip().replace('"', '').replace("'", "")
                                     for x in data.split(",") if x.strip()]
                data_collector_2.append([
                    sent_id, sentence, label, pattern, matched_phrase, target_label, generated_phrases
                ])

    df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
    cand_path = f"output_data/[{SEED}]{stem}_candidate_phrases_annotated_data_iter{ITERATION}.csv"
    df2.to_csv(cand_path, index=False)
    print("DEBUG: candidate rows written →", cand_path, "| rows:", len(df2))


if __name__ == "__main__":
    get_patat_patterns()
    get_candidate_phrases()