# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import gaussian_filter
import scienceplots
from datetime import datetime
import uuid
import pickle
from icrl import ICRLState, ProblemHistory, Attempt

plt.style.use(['science', 'no-latex'])

# Increase font size for all elements
plt.rcParams.update({
    'font.size': 10,  # Base font size
    'axes.labelsize': 30,  # Axis label font size
    'xtick.labelsize': 36,  # X-axis tick label size
    'ytick.labelsize': 36,  # Y-axis tick label size
    'legend.fontsize': 36,  # Legend font size
})

def load_latest_checkpoint(checkpoint_dir):
    """Load the latest checkpoint from a directory."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pkl"))
    if checkpoints:
        return ICRLState.load(checkpoints[-1])
    return None

def get_sum_df(path):
    """
    Get sum dataframe for mathador ICRL data.

    Args:
        path: Path to checkpoint directory

    Returns:
        DataFrame with rounds as index and problems as columns
    """
    checkpoint_dir = Path(path)
    state = load_latest_checkpoint(checkpoint_dir)

    if state is None:
        raise FileNotFoundError(f"No checkpoint found in {path}")

    dict_data = defaultdict(dict)

    # Get all unique round indices
    all_rounds = set()
    for problem_history in state.problem_histories:
        for attempt in problem_history.attempts:
            all_rounds.add(attempt.round_idx)

    all_rounds = sorted(list(all_rounds))

    # Organize data by round and problem
    for problem_idx, problem_history in enumerate(state.problem_histories):
        # Group attempts by round
        attempts_by_round = defaultdict(list)
        for attempt in problem_history.attempts:
            # Use the reward directly (already normalized 0-1)
            reward = attempt.reward
            attempts_by_round[attempt.round_idx].append(reward)

        # For each round, take the max reward if multiple attempts in the same round
        for round_idx in all_rounds:
            if round_idx in attempts_by_round:
                dict_data[problem_idx][round_idx] = np.max(attempts_by_round[round_idx])
            else:
                # No attempt in this round for this problem
                dict_data[problem_idx][round_idx] = 0

    df = pd.DataFrame(dict_data)

    # Handle negative rewards (set to 0)
    df = df.applymap(lambda x: max(0, x) if isinstance(x, (int, float)) else x)

    return df

def plot_per_step(*dfs, **kwargs):
    """Plot mean reward per step for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    for idx, df in enumerate(dfs):
        means = df.mean(axis=1)
        std_devs = df.std(axis=1)/np.sqrt(len(df.columns))
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')

        ax.plot(rounds, means, f'{color}-', label=label)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Reward')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_running_max(*dfs, **kwargs):
    """Plot running max reward per step for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    first_perf = []
    for idx, df in enumerate(dfs):
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        # Create running max df
        df_running_max = df.cummax(axis=0)
        means = df_running_max.mean(axis=1)
        print(label, means.iloc[-1])

        first_perf.append(means.iloc[0])
        means = pd.Series(gaussian_filter(means, sigma=1), index=means.index)
        std_devs = df_running_max.std(axis=1)/np.sqrt(len(df.columns))/4
        rounds = df.index
        color = colors[idx % len(colors)]
        ax.plot(rounds, means, f'{color}-', label=label)
        ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3, color=color)

    print('first perf', np.mean(first_perf))

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Running Max Episode Return')
    ax.legend()

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf',
                   format='pdf', bbox_inches='tight')
    plt.show()

def plot_per_step_sliding_average(*dfs, window_size=10, **kwargs):
    """Plot sliding window average for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    for idx, df in enumerate(dfs):
        # Calculate sliding window average
        df_sliding = df.rolling(window=window_size).mean()
        means = df_sliding.mean(axis=1)
        # Only plot from window_size onwards
        rounds = df.index[window_size - 1:]
        means = means[window_size - 1:]
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')

        ax.plot(rounds, means, f'{color}-', label=label)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Sliding Average Reward (Window Size: {window_size})')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_gaussian_smoothed(*dfs, param, **kwargs):
    """Plot Gaussian smoothed rewards for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    for idx, df in enumerate(dfs):
        # Apply Gaussian smoothing to each column
        df_smoothed = df.apply(lambda x: gaussian_filter(x, sigma=param))
        means = df_smoothed.mean(axis=1)
        rounds = df.index
        color = colors[idx % len(colors)]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        std_devs = df_smoothed.std(axis=1)

        ax.plot(rounds, means, f'{color}-', label=label)
        ax.fill_between(rounds, means - .5*std_devs/np.sqrt(len(df.columns)),
                       means + .5*std_devs/np.sqrt(len(df.columns)), alpha=0.3, color=color)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Episode Return')
    ax.legend(fontsize=18)

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf',
                   format='pdf', bbox_inches='tight')
    plt.show()

def print_statistics(df, label=""):
    """Print statistics for a dataframe."""
    if label:
        print(f"\n{label}:")
    print("="*50)
    quantiles = df.quantile([0.25, 0.5, 0.75], axis=1)
    print("Quantiles (25%, 50%, 75%):")
    print(quantiles)
    print("\nMean per round:")
    print(df.mean(axis=1))
    print("="*50)

def get_ablation_data(checkpoint_path):
    """
    Load ablation study checkpoint and extract data organized by i.

    Args:
        checkpoint_path: Path to ablation checkpoint file

    Returns:
        Dictionary mapping ablation_i -> list of rewards
    """
    checkpoint_path = Path(checkpoint_path)
    state = ICRLState.load(checkpoint_path)

    # Organize rewards by ablation_i
    ablation_data = defaultdict(list)

    for problem_history in state.problem_histories:
        for attempt in problem_history.attempts:
            if 'ablation_i' in attempt.extra_fields:
                i = attempt.extra_fields['ablation_i']
                ablation_data[i].append(attempt.reward)

    return dict(ablation_data)

def plot_ablation_effect(checkpoint_path, sigma=1.0, **kwargs):
    """
    Plot the effect of removing bad attempts on reward.

    Args:
        checkpoint_path: Path to ablation checkpoint file
        sigma: Gaussian smoothing parameter (default: 1.0)
        **kwargs: Additional plotting options (save, label, etc.)
    """
    ablation_data = get_ablation_data(checkpoint_path)

    if not ablation_data:
        print("No ablation data found in checkpoint!")
        return

    # Sort by i
    i_values = sorted(ablation_data.keys())
    means = [np.mean(ablation_data[i]) for i in i_values]

    # Apply Gaussian smoothing
    smoothed_means = gaussian_filter(means, sigma=sigma)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot smoothed mean
    label = kwargs.get('label', 'Mean Reward')
    ax.plot(i_values, smoothed_means, 'b-', label=label, linewidth=2)

    ax.set_xlabel('Number of Worst Attempts Removed')
    ax.set_ylabel('Reward')
    ax.legend()

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf',
                   format='pdf', bbox_inches='tight')
    plt.show()
