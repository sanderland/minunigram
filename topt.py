#%%
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from py_unigram.pretokenize import pretokenize_corpus, SPACES_PRE_TOKENIZER_REGEX
from py_unigram.train import train_unigram
from collections import defaultdict

DEFAULT_ARGS = {
    'vocab_size': 1024,
    'max_token_len': 16,
    'initial_vocab_factor': 10,
    'pruning_shrinking_factor': 0.75,
    'verbose': True,
   # 'm_step_dp_smoothing': False,
}
PARAMS_TO_TEST = {
    'initial_vocab_factor': list(range(1, 21)),
    'pruning_shrinking_factor': list(np.linspace(0.4, 0.95, 10)) + [0.95, 0.96, 0.97, 0.98, 0.99]
}

# --- PLOT CONFIGURATION ---
# A list of the metrics to calculate and generate plots for.
METRICS_TO_PLOT = ['objective', 'bytes/token']

#%%
# Load data from the specified file
with open("tests/data/swift_clean.txt") as f:
    texts = f.read().splitlines()

#ds = load_dataset("sanderland/monolingual-tokenizer-data", data_files=[f"eng_latn_300mb.txt"],        split="train")
#texts = list(ds["text"])

# Pre-tokenize the corpus
pretokens = pretokenize_corpus(texts, SPACES_PRE_TOKENIZER_REGEX)

# Calculate initial corpus statistics
total_pretokens = sum(len(p) for p in pretokens)
total_bytes = sum(len(t.encode('utf-8')) for t in texts)

print(f"Loaded {len(texts)} lines of text.")
print(f"Total Pre-tokens: {total_pretokens}")
print(f"Total Bytes: {total_bytes}")

#%%
all_results = {}

# Iterate over each hyperparameter defined in the configuration
for param_name, param_values in PARAMS_TO_TEST.items():
    print(f"\n--- Testing Hyperparameter: {param_name} ---")

    # Initialize a dictionary to store results for the current parameter
    results = defaultdict(list)

    # Test each value for the current hyperparameter
    for value in param_values:
        train_kwargs = {**DEFAULT_ARGS, param_name: value}
        model, stats = train_unigram(pretokens, **train_kwargs)
        results['param'].append(value)
        for key, stat_val in stats.items():
            results[key].append(stat_val)

    # Store the results for this parameter in the main results dictionary
    all_results[param_name] = results

#%%
num_params = len(all_results)
num_metrics = len(METRICS_TO_PLOT)
# Create a figure and a grid of subplots
fig, axes = plt.subplots(
    nrows=num_metrics,
    ncols=num_params,
    figsize=(7 * num_params, 5 * num_metrics),
    squeeze=False, # Always return a 2D array for axes
    dpi=150
)

# Iterate through each metric to create a row of plots
for i, metric in enumerate(METRICS_TO_PLOT):
    # Iterate through each parameter test result to create a column of plots
    for j, (param_name, results) in enumerate(all_results.items()):
        ax = axes[i, j]
        
        # Extract data from the simplified results dictionary
        param_values = results['param']
        metric_values = results[metric]

        # Plot the data
        ax.plot(param_values, metric_values, 'o-')

        # Set labels and title using raw parameter and metric names
        ax.set_title(f"{metric} vs. {param_name}")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.5)

plt.tight_layout()

# %%
