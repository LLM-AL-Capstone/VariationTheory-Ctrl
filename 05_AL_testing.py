# 05_AL_testing.py — GPT-4o version with cumulative iterations
import os, sys, glob, numpy as np, pandas as pd, configparser, torch
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from collections import defaultdict
from llm_gpt4o import get_client, chat_call, warmup, MODEL_ID
from sample_config import get_cumulative_sample_ids, get_dataset_info, detect_dataset_name

config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
config.read('config.ini')

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
TEST_FILE = (config.get("settings", "testing_file").split('#')[0].strip()
             if config.has_option("settings", "testing_file")
             else config.get("settings", "test_file").split('#')[0].strip())
SEED = 42  # Fixed seed as requested

np.random.seed(SEED)
torch.manual_seed(SEED)

stem = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE
DATASET_NAME = detect_dataset_name(DATA_FILE)

print(f"INFO: Running evaluation for dataset: {DATASET_NAME}")
print(f"INFO: Using fixed seed: {SEED}")

client = get_client(timeout=600.0)
warmup(client)


def normalize_id(val):
    """Normalize ID values for comparison"""
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    try:
        return int(val_str)
    except ValueError:
        return val_str


def get_initial_message(labels):
    return [
        {"role": "system", "content": "You will predict one label for a text based on the examples."},
        {"role": "system", "content": f"You must choose from: {labels}"},
    ]


def get_response(messages, query):
    import copy
    q = copy.deepcopy(messages)
    q.append({"role": "user", "content": query})
    return chat_call(client, q, model=MODEL_ID, max_tokens=16, temperature=0)


def update_example(messages, text, label):
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": label})


def load_training_data_for_iteration(iteration):
    """
    Load training data for a specific iteration.
    Tries to find the filtered file for that iteration.
    """
    # Try iteration-specific file first
    iter_file = f"output_data/[{SEED}]filtered_{stem}_iter{iteration}.csv"

    if os.path.exists(iter_file):
        print(f"  Loading from: {iter_file}")
        return pd.read_csv(iter_file)

    # If iteration-specific file doesn't exist, we need to use the original data
    # and filter by IDs ourselves
    print(f"  Note: Iteration-specific file not found: {iter_file}")
    print(f"  Will use ID filtering from original data")
    return None


def run_evaluation_for_iteration(df_original, df_test, label_map, iteration, max_test_samples=100):
    """
    Run evaluation using specific sample IDs for the given iteration.

    Args:
        df_original: Original training dataframe with all data
        df_test: Test dataframe
        label_map: Mapping from original labels to concept labels
        iteration: Current iteration number (0 = seed only, 1 = seed+iter1, etc.)
        max_test_samples: Maximum number of test samples to evaluate

    Returns:
        Dictionary with metrics
    """
    print(f"\n{'=' * 80}")
    print(f"EVALUATING ITERATION {iteration}")
    print(f"{'=' * 80}")

    # Try to load iteration-specific filtered data first
    df_train = load_training_data_for_iteration(iteration)

    # Get cumulative sample IDs for this iteration
    try:
        sample_ids_raw = get_cumulative_sample_ids(DATASET_NAME, iteration)
        sample_ids = [normalize_id(x) for x in sample_ids_raw]
        print(f"INFO: Using {len(sample_ids)} cumulative samples (seed + iterations 1-{iteration})")
    except Exception as e:
        print(f"ERROR: Could not get sample IDs for iteration {iteration}: {e}")
        return None

    # If we have iteration-specific data, use it
    if df_train is not None:
        # Filter to only include specific sample IDs
        if 'id' in df_train.columns:
            df_train['id'] = df_train['id'].apply(normalize_id)
            df_train_filtered = df_train[df_train['id'].isin(sample_ids)].copy()
        else:
            df_train_filtered = df_train.copy()
    else:
        # Fall back to using original data and filtering by IDs
        print(f"  INFO: Using original data with ID filtering")
        df_train_filtered = df_original[df_original['id'].isin(sample_ids)].copy()

    print(f"INFO: Found {len(df_train_filtered)} matching samples in training data")

    if len(df_train_filtered) == 0:
        print("ERROR: No matching samples found!")
        return None

    # Build messages with real examples
    messages = get_initial_message(list(label_map.values()))

    # Add all training examples to the prompt
    example_count = 0
    for _, row in df_train_filtered.iterrows():
        text_col = 'ori_text' if 'ori_text' in row else 'example'
        label_col = 'ori_label' if 'ori_label' in row else 'Label'

        if pd.notna(row.get(text_col)) and pd.notna(row.get(label_col)):
            if row[label_col] in label_map:
                update_example(messages, row[text_col], label_map[row[label_col]])
                example_count += 1

    # Also add counterfactuals if available (filtered ones)
    if 'counterfactual' in df_train_filtered.columns:
        cf_data = df_train_filtered[
            (df_train_filtered.get('matched_pattern', pd.Series([False] * len(df_train_filtered))).fillna(False)) &
            (df_train_filtered.get('heuristic_filtered', pd.Series([False] * len(df_train_filtered))).fillna(False)) &
            (~df_train_filtered.get('is_ori', pd.Series([True] * len(df_train_filtered))).fillna(True)) &
            (df_train_filtered.get('is_target', pd.Series([False] * len(df_train_filtered))).fillna(False))
            ]

        cf_count = 0
        for _, row in cf_data.iterrows():
            if pd.notna(row.get('counterfactual')) and pd.notna(row.get('target_label')):
                if row['target_label'] in label_map:
                    update_example(messages, row['counterfactual'], label_map[row['target_label']])
                    cf_count += 1

        print(f"INFO: Added {cf_count} counterfactual examples")

    print(f"INFO: Total examples in prompt: {example_count}")

    # Evaluate on test set
    y_true, y_pred = [], []
    df_test_sample = df_test.sample(n=min(max_test_samples, len(df_test)), random_state=SEED)

    print(f"INFO: Evaluating on {len(df_test_sample)} test samples...")

    for idx, (_, row) in enumerate(df_test_sample.iterrows()):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(df_test_sample)}")

        if row['Label'] == "none" or row['Label'] not in label_map:
            continue

        try:
            response = get_response(messages, row['example'])
            y_true.append(label_map[row['Label']])
            y_pred.append(response.strip())
        except Exception as e:
            print(f"  WARNING: Failed to get response for test sample {idx}: {e}")
            continue

    if len(y_true) == 0:
        print("ERROR: No predictions made!")
        return None

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    results = {
        'iteration': iteration,
        'num_samples': len(sample_ids),
        'num_train_examples': len(df_train_filtered),
        'num_test_examples': len(y_true),
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }

    print(f"\nRESULTS:")
    print(f"  Samples used: {results['num_samples']}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")
    print(f"  F1 Weighted: {f1_weighted:.4f}")

    return results


def main():
    print(f"\n{'=' * 80}")
    print(f"ACTIVE LEARNING EVALUATION - CUMULATIVE ITERATIONS")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Seed: {SEED}")
    print(f"{'=' * 80}\n")

    # First, try to find ANY training data file we can use
    print("Searching for training data files...")

    # Look for iteration-specific filtered files
    filtered_files = sorted(glob.glob(f"output_data/[{SEED}]filtered_{stem}_iter*.csv"))

    # Also look for non-iteration filtered files
    non_iter_files = glob.glob(f"output_data/[{SEED}]filtered_{stem}.csv")

    # Look for annotated files as fallback
    annotated_files = sorted(glob.glob(f"output_data/[{SEED}]annotated_data_with_pattern_{stem}_iter*.csv"))

    # Look for original input data as last resort
    input_file = f"input_data/{DATA_FILE}"

    print(f"  Found {len(filtered_files)} iteration-specific filtered files")
    print(f"  Found {len(non_iter_files)} non-iteration filtered files")
    print(f"  Found {len(annotated_files)} annotated files")

    # Determine which file to use as base
    if filtered_files:
        base_file = filtered_files[-1]  # Use most recent
        print(f"  Using most recent filtered file as base: {base_file}")
    elif non_iter_files:
        base_file = non_iter_files[0]
        print(f"  Using non-iteration filtered file as base: {base_file}")
    elif os.path.exists(input_file):
        base_file = input_file
        print(f"  Using original input file as base: {base_file}")
    else:
        sys.exit(f"ERROR: No training data found. Please ensure files exist in output_data/ or input_data/")

    # Load base training data
    print(f"\nLoading base training data from: {base_file}")
    df_original = pd.read_csv(base_file)

    # Normalize IDs in original data
    if 'id' not in df_original.columns:
        # Create ID column if it doesn't exist
        df_original['id'] = range(len(df_original))
    df_original['id'] = df_original['id'].apply(normalize_id)

    # Ensure we have the right columns for original data
    if 'ori_text' not in df_original.columns:
        if 'example' in df_original.columns:
            df_original['ori_text'] = df_original['example']
        elif 'Text' in df_original.columns:
            df_original['ori_text'] = df_original['Text']

    if 'ori_label' not in df_original.columns:
        if 'Label' in df_original.columns:
            df_original['ori_label'] = df_original['Label']

    # Load test data
    test_path = f"input_data/{TEST_FILE}"
    if not os.path.exists(test_path):
        sys.exit(f"ERROR: missing {test_path}")

    print(f"Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)

    # Handle 'example' column in test data
    if 'example' not in df_test.columns:
        if 'Text' in df_test.columns:
            df_test['example'] = df_test['Text']
        else:
            df_test['example'] = df_test.iloc[:, 0 if 'Label' not in df_test.columns else 1]

    # Get label mapping
    if 'ori_label' in df_original.columns:
        label_list = df_original["ori_label"].unique().tolist()
    elif 'Label' in df_original.columns:
        label_list = df_original["Label"].unique().tolist()
    else:
        sys.exit("ERROR: Cannot find label column in training data")

    label_map = {label: f'concept {chr(65 + i)}' for i, label in enumerate(label_list) if label != 'none'}

    print(f"\nLabel mapping: {label_map}")

    # Get dataset info to determine number of iterations
    try:
        dataset_info = get_dataset_info(DATASET_NAME)
        max_iterations = dataset_info['total_iterations']
        print(f"\nDataset has {max_iterations} iterations defined")
    except Exception as e:
        print(f"WARNING: Could not get dataset info: {e}")
        max_iterations = 10  # Default fallback

    # Run evaluation for each cumulative iteration
    all_results = []

    # Iteration 0 = seed only
    for iteration in range(0, max_iterations + 1):
        result = run_evaluation_for_iteration(
            df_original, df_test, label_map, iteration, max_test_samples=100
        )

        if result:
            all_results.append(result)
        else:
            print(f"WARNING: Skipping iteration {iteration} due to errors")

    # Create results dataframe and save
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Create output directory
        os.makedirs(f"output_data/results", exist_ok=True)

        # Save detailed results
        output_path = f"output_data/results/[{SEED}]cumulative_results_{stem}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n{'=' * 80}")
        print(f"Saved detailed results to: {output_path}")

        # Print summary table
        print(f"\n{'=' * 80}")
        print("SUMMARY TABLE - MACRO F1 SCORES BY ITERATION")
        print(f"{'=' * 80}\n")

        print(f"{'Iteration':<12} {'Samples':<10} {'F1 Macro':<12} {'Accuracy':<12}")
        print("-" * 50)

        for _, row in results_df.iterrows():
            iter_label = "Seed" if row['iteration'] == 0 else f"Seed+1-{row['iteration']}"
            print(f"{iter_label:<12} {row['num_samples']:<10} {row['f1_macro']:<12.4f} {row['accuracy']:<12.4f}")

        print("\n" + "=" * 80)

        # Create a simple summary table for easy viewing
        summary_df = results_df[['iteration', 'num_samples', 'f1_macro', 'accuracy']].copy()
        summary_df['iteration_label'] = summary_df['iteration'].apply(
            lambda x: "Seed" if x == 0 else f"Seed+1-{x}"
        )
        summary_df = summary_df[['iteration_label', 'num_samples', 'f1_macro', 'accuracy']]
        summary_df.columns = ['Iteration', 'Samples Used', 'F1 Macro', 'Accuracy']

        summary_path = f"output_data/results/[{SEED}]summary_{stem}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary table to: {summary_path}")

        # Also save as formatted text
        text_path = f"output_data/results/[{SEED}]summary_{stem}.txt"
        with open(text_path, 'w') as f:
            f.write(f"CUMULATIVE EVALUATION RESULTS\n")
            f.write(f"Dataset: {DATASET_NAME}\n")
            f.write(f"Seed: {SEED}\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(summary_df.to_string(index=False))
            f.write(f"\n\n{'=' * 80}\n")
        print(f"Saved formatted summary to: {text_path}")

        print(f"\n{'=' * 80}")
        print("EVALUATION COMPLETE")
        print(f"{'=' * 80}\n")
    else:
        print("\nERROR: No results generated!")
        sys.exit(1)


if __name__ == "__main__":
    main()