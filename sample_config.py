# sample_config.py
"""
Configuration for using specific samples with GPT-4o across multiple datasets
"""

# NLI_CAD Sample IDs
NLI_CAD_SAMPLE_IDS = {
    "seed": [30, 55, 70, 72, 75, 148, 153, 154, 157, 180, 195, 199, 225, 275, 325, 382, 391, 392, 414, 424, 426],
    "iteration_1": [386, 240, 278, 24, 343, 230, 302, 243, 139, 349],
    "iteration_2": [291, 84, 274, 48, 347, 344, 332, 118, 299, 398],
    "iteration_3": [82, 296, 383, 11, 244, 198, 87, 279, 109, 266],
    "iteration_4": [56, 339, 61, 219, 13, 295, 41, 0, 184, 91],
    "iteration_5": [12, 80, 340, 322, 217, 181, 150, 375, 381, 54],
    "iteration_6": [233, 214, 431, 373, 197, 209, 5, 33, 163, 23],
    "iteration_7": [330, 79, 316, 35, 390, 73, 254, 130, 374, 357],
    "iteration_8": [135, 395, 58, 303, 269, 49, 317, 335, 358, 256],
    "iteration_9": [341, 149, 428, 363, 103, 315, 231, 15, 28, 71],
    "iteration_10": [216, 131, 326, 329, 310, 262, 370, 255, 60, 227],
}

# ANLI Sample IDs
ANLI_SAMPLE_IDS = {
    "seed": [30, 55, 70, 72, 75, 148, 153, 154, 157, 180, 195, 199, 225, 275, 325, 382, 391, 392, 414, 424, 426],
    "iteration_1": [106, 301, 51, 378, 160, 314, 285, 37, 222, 223],
    "iteration_2": [32, 17, 146, 6, 237, 286, 124, 201, 224, 401],
    "iteration_3": [48, 327, 27, 374, 352, 64, 421, 256, 170, 232],
    "iteration_4": [405, 177, 99, 347, 295, 296, 139, 355, 66, 141],
    "iteration_5": [298, 303, 15, 93, 264, 217, 411, 277, 379, 23],
    "iteration_6": [16, 150, 342, 212, 409, 172, 171, 211, 92, 49],
    "iteration_7": [78, 427, 95, 71, 239, 1, 417, 190, 292, 258],
    "iteration_8": [415, 84, 166, 73, 88, 320, 129, 255, 102, 336],
    "iteration_9": [290, 276, 135, 360, 386, 130, 373, 39, 28, 165],
}

# YELP Sample IDs
YELP_SAMPLE_IDS = {
    "seed": ["ss73", "ss104", "ss124", "ss155", "ss280", "ss316", "ss356", "ss361", "ss371", "ss374",
             "ss377", "ss388", "ss394", "ss406", "ss408", "ss450", "ss475", "ss490", "ss491", "ss497"],
    "iteration_1": ["ss90", "ss386", "ss342", "ss149", "ss393", "ss395", "ss472", "ss232", "ss409", "ss487"],
    "iteration_2": ["ss9", "ss183", "ss33", "ss427", "ss323", "ss367", "ss488", "ss258", "ss449", "ss464"],
    "iteration_3": ["ss345", "ss242", "ss151", "ss479", "ss51", "ss207", "ss405", "ss314", "ss35", "ss14"],
    "iteration_4": ["ss381", "ss4", "ss308", "ss68", "ss410", "ss43", "ss485", "ss91", "ss337", "ss141"],
    "iteration_5": ["ss209", "ss492", "ss70", "ss105", "ss378", "ss92", "ss411", "ss297", "ss112", "ss480"],
    "iteration_6": ["ss407", "ss328", "ss401", "ss179", "ss245", "ss306", "ss262", "ss71", "ss147", "ss368"],
    "iteration_7": ["ss38", "ss145", "ss257", "ss175", "ss186", "ss148", "ss412", "ss54", "ss120", "ss181"],
    "iteration_8": ["ss11", "ss489", "ss318", "ss80", "ss34", "ss52", "ss47", "ss284", "ss426", "ss125"],
    "iteration_9": ["ss350", "ss37", "ss329", "ss341", "ss317", "ss287", "ss279", "ss10", "ss252", "ss273"],
    "iteration_10": ["ss247", "ss158", "ss12", "ss109", "ss7", "ss8", "ss76", "ss424", "ss452", "ss373"],
}

# SA_CAD (Sentiment Analysis) Sample IDs
SA_CAD_SAMPLE_IDS = {
    "seed": [9, 30, 55, 70, 72, 75, 148, 153, 154, 180, 195, 199, 275, 286, 325, 334, 382, 392, 414, 424],
    "iteration_1": [198, 206, 141, 77, 83, 252, 103, 264, 165, 355],
    "iteration_2": [318, 287, 123, 290, 278, 227, 11, 111, 131, 248],
    "iteration_3": [39, 333, 53, 341, 317, 36, 387, 74, 7, 132],
    "iteration_4": [128, 363, 412, 90, 385, 319, 182, 177, 172, 13],
    "iteration_5": [409, 386, 353, 410, 171, 288, 202, 407, 251, 181],
    "iteration_6": [326, 349, 396, 360, 296, 188, 67, 232, 185, 416],
    "iteration_7": [166, 356, 146, 313, 361, 57, 109, 272, 345, 243],
    "iteration_8": [377, 21, 233, 357, 358, 337, 186, 64, 397, 93],
    "iteration_9": [121, 81, 346, 91, 398, 429, 292, 271, 68, 40],
    "iteration_10": [383, 104, 78, 66, 298, 18, 395, 62, 82, 117],
}

# AMAZON_POLARITY Sample IDs
AMAZON_POLARITY_SAMPLE_IDS = {
    "seed": [9, 30, 55, 70, 72, 75, 148, 153, 154, 180, 195, 199, 275, 286, 325, 334, 382, 392, 414, 424],
    "iteration_1": [429, 226, 387, 76, 144, 107, 60, 207, 162, 311],
    "iteration_2": [0, 74, 316, 419, 297, 267, 366, 202, 58, 278],
    "iteration_3": [255, 227, 22, 113, 236, 271, 178, 421, 252, 196],
    "iteration_4": [138, 205, 152, 240, 367, 140, 287, 280, 395, 159],
    "iteration_5": [407, 399, 17, 157, 242, 300, 361, 54, 170, 66],
    "iteration_6": [427, 390, 304, 212, 261, 103, 5, 281, 409, 401],
    "iteration_7": [39, 279, 251, 350, 383, 166, 131, 206, 176, 150],
    "iteration_8": [14, 375, 250, 340, 156, 175, 132, 266, 59, 11],
    "iteration_9": [24, 394, 269, 10, 109, 48, 256, 50, 129, 420],
    "iteration_10": [173, 111, 61, 358, 118, 306, 333, 83, 91, 147],
}

# Dataset mapping
DATASET_SAMPLE_IDS = {
    "nli_cad": NLI_CAD_SAMPLE_IDS,
    "anli": ANLI_SAMPLE_IDS,
    "yelp": YELP_SAMPLE_IDS,
    "sa_cad": SA_CAD_SAMPLE_IDS,
    "amazon_polarity": AMAZON_POLARITY_SAMPLE_IDS,
}

# GPT-4o Configuration
GPT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-2024-11-20",
    "temperature": 0.0,
    "max_tokens": 256,
}


def detect_dataset_name(filename):
    """
    Detect dataset name from filename.

    Args:
        filename: Name of the data file

    Returns:
        Detected dataset name
    """
    filename_lower = filename.lower()

    # Remove common suffixes
    base = filename_lower.replace('_train.csv', '').replace('.csv', '').replace('_test', '')

    # Check for known datasets
    if 'nli_cad' in base or base == 'nli':
        return 'nli_cad'
    elif 'anli' in base:
        return 'anli'
    elif 'yelp' in base:
        return 'yelp'
    elif 'sa_cad' in base or base == 'sa':
        return 'sa_cad'
    elif 'amazon' in base or 'polarity' in base:
        return 'amazon_polarity'
    else:
        # Try to match with existing dataset keys
        for dataset_key in DATASET_SAMPLE_IDS.keys():
            if dataset_key.replace('_', '') in base.replace('_', ''):
                return dataset_key
        # Default fallback
        return base


def get_sample_ids_for_dataset(dataset_name):
    """
    Get sample IDs for a specific dataset.

    Args:
        dataset_name: Name of the dataset (case-insensitive)

    Returns:
        Dictionary of sample IDs by iteration

    Raises:
        ValueError if dataset not found
    """
    dataset_key = dataset_name.lower().strip()

    # Handle common variations
    dataset_aliases = {
        "nli": "nli_cad",
        "nli_cad_train": "nli_cad",
        "anli_train": "anli",
        "yelp_train": "yelp",
        "sa": "sa_cad",
        "sa_cad_train": "sa_cad",
        "sentiment": "sa_cad",
        "amazon": "amazon_polarity",
        "amazon_polarity_train": "amazon_polarity",
    }

    # Check if it's an alias
    if dataset_key in dataset_aliases:
        dataset_key = dataset_aliases[dataset_key]

    # Try to find in the mapping
    if dataset_key in DATASET_SAMPLE_IDS:
        return DATASET_SAMPLE_IDS[dataset_key]

    # If still not found, raise error
    raise ValueError(
        f"Dataset '{dataset_name}' not found. Available datasets: {list(DATASET_SAMPLE_IDS.keys())}"
    )


def get_cumulative_sample_ids(dataset_name, iteration):
    """
    Get cumulative sample IDs up to and including the specified iteration.

    Args:
        dataset_name: Name of the dataset
        iteration: Current iteration number (0-based for seed, 1+ for iterations)

    Returns:
        List of cumulative sample IDs
    """
    sample_ids_dict = get_sample_ids_for_dataset(dataset_name)

    # Start with seed set
    ids = sample_ids_dict["seed"].copy()

    # Add samples from each iteration up to current
    for i in range(1, iteration + 1):
        iter_key = f"iteration_{i}"
        if iter_key in sample_ids_dict:
            ids.extend(sample_ids_dict[iter_key])

    return ids


def get_dataset_info(dataset_name):
    """
    Get information about a dataset's sample configuration.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    sample_ids_dict = get_sample_ids_for_dataset(dataset_name)

    seed_count = len(sample_ids_dict.get("seed", []))

    # Count total iterations
    max_iter = 0
    for key in sample_ids_dict.keys():
        if key.startswith("iteration_"):
            iter_num = int(key.split("_")[1])
            max_iter = max(max_iter, iter_num)

    # Calculate total samples
    total = seed_count
    for i in range(1, max_iter + 1):
        iter_key = f"iteration_{i}"
        if iter_key in sample_ids_dict:
            total += len(sample_ids_dict[iter_key])

    return {
        "dataset": dataset_name,
        "seed_samples": seed_count,
        "total_iterations": max_iter,
        "total_samples": total,
        "sample_type": type(sample_ids_dict["seed"][0]).__name__
    }