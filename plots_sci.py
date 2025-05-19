# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import glob
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
from scipy.ndimage import gaussian_filter
# import scienceplots
from datetime import datetime
import uuid
import re
import matplotlib.colors as mcolors
# plt.style.use(['science', 'no-latex'])

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=list(mcolors.TABLEAU_COLORS.values()))
plt.rcParams.update({
    'font.size': 10,  # Base font size
    'axes.labelsize': 30,  # Axis label font size
    'xtick.labelsize': 20,  # X-axis tick label size
    'ytick.labelsize': 20,  # Y-axis tick label size
    'legend.fontsize': 18,  # Legend font size
    'lines.linewidth': 2.5,    # Default line width for all plots
    'axes.xmargin': 0,  # remove x-axis data margin
})

def find_sciworld_file(folder_path, raw_prompts=False):
    """Find the sciworld data file in a given folder."""
    if raw_prompts:
        pattern = os.path.join(folder_path, "raw_prompts_sciworld_data_round_*_final.json")
    else:
        pattern = os.path.join(folder_path, "sciworld_data_round_*_final.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No sciworld data file found in {folder_path}")
    return files[0]  # Return the first matching file

def convert_keys_to_int(obj):
    """Convert string keys to integers if possible."""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, str) and k.isdigit() else k: convert_keys_to_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_int(item) for item in obj]
    return obj

def insert_one_bootstrap_before_round_attempts(data):
    if 'bootstrap_attempts' in data[0].keys() and len(data[0]['bootstrap_attempts']) > 0:
        # shift all round_attempts by 1
        for env_id in data.keys():
            for round_idx in reversed(list(data[env_id]['round_attempts'].keys())):
                data[env_id]['round_attempts'][round_idx + 1] = data[env_id]['round_attempts'][round_idx]
            # insert the first bootstrap_attempt at the beginning of round_attempts
            message_list = data[env_id]['bootstrap_attempts'][0]
            if not isinstance(message_list, list): # due to some bug in the data loading, bootstrap_attempt might be round_attempt
                if 0 in message_list.keys(): # make sure you are dealing with raw_prompts
                    message_list = message_list[0]
                    assert isinstance(message_list, list)
            data[env_id]['round_attempts'][0] = {0: message_list}

def get_sum_df(path, cut=True):
    path = find_sciworld_file(path)
    data = json.load(open(path), object_hook=convert_keys_to_int)
    insert_one_bootstrap_before_round_attempts(data)
    dict_data = defaultdict(dict)

    # offset = 0
    # # extract bootstrap_attempts' rewards if exists
    # if 'bootstrap_attempts' in data[0].keys() and len(data[0]['bootstrap_attempts']) > 0:
    #     for env_id in data.keys():
    #         rewards_list_list = []
    #         for bootstrap_idx in data[env_id]['bootstrap_attempts'].keys():
    #             rewards_list_list.append(data[env_id]['bootstrap_attempts'][bootstrap_idx]['rewards'])
    #         for i, rewards_list in enumerate(rewards_list_list):
    #             rewards_list_list[i] = np.sum([0 if x < 0 else x for x in rewards_list])
    #         dict_data[env_id][0] = [np.mean(rewards_list_list)]
    #         offset = 1
    # extract round_attempts' rewards
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            dict_data[env_id][round_idx] = data[env_id]['round_attempts'][round_idx][0]['rewards']
    df = pd.DataFrame(dict_data)
    df = df.applymap(lambda x: [0 if xx < 0 else xx for xx in x])
    df = df.applymap(lambda x: np.sum(x) if isinstance(x, list) else x)
    if cut:
        df = df[df.index.astype(int) < 40]
    df = df.drop(columns=df.columns[cols_to_drop])
    return df

def plot_per_step(*dfs, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, df in enumerate(dfs):
        means = df.mean(axis=1)
        std_devs = df.std(axis=1)/np.sqrt(len(df))
        rounds = df.index
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, label=label)
        # ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Reward')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_running_max(*dfs, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, df in enumerate(dfs):
        # create running max df
        df_running_max = df.cummax(axis=0)
        means = df_running_max.mean(axis=1)
        means = pd.Series(gaussian_filter(means, sigma=1), index=means.index)
        std_devs = df_running_max.std(axis=1)/np.sqrt(len(df))/4
        rounds = df.index
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        ax.plot(rounds, means, label=label)
        ax.fill_between(rounds, means - std_devs, means + std_devs, alpha=0.3)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Running Max Return')
    ax.legend()

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def plot_per_step_sliding_average(*dfs, window_size=10, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, df in enumerate(dfs):
        # Calculate sliding window average
        df_sliding = df.rolling(window=window_size).mean()
        means = df_sliding.mean(axis=1)
        # Only plot from window_size onwards
        rounds = df.index[window_size - 1:]
        means = means[window_size - 1:]
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.plot(rounds, means, label=label)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Sliding Average Reward (Window Size: {window_size})')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_per_step_gaussian_smoothed(*dfs, param, **kwargs):
    # Create a single figure for comparing all methods
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, df in enumerate(dfs):
        # Apply Gaussian smoothing to each column
        if hasattr(df, 'donotsmooth') and df.donotsmooth:
            df_smoothed = df
        else:
            df_smoothed = df.apply(lambda x: gaussian_filter(x, sigma=param))
        means = df_smoothed.mean(axis=1)
        rounds = df.index
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        std_devs = df_smoothed.std(axis=1)
        
        ax.plot(rounds, means, label=label)
        ax.fill_between(rounds, means - .5*std_devs/np.sqrt(len(df)), means + .5*std_devs/np.sqrt(len(df)), alpha=0.3)
    ax.set_xlabel('Trial Number')
    ax.set_ylabel(f'Return')
    # ax.legend(framealpha=0.8, facecolor='white', frameon=True)
    ax.legend(loc='lower right')

    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def get_cost(path):
    """
    for each env:
        round_cost = 0
        for each round, for each messages in raw_prompts:
            input_count = len(all except the last one)
            output_count = len(the last one)
            cost = cost_input * input_count + cost_output * output_count
            round_cost += cost
    """
    cost_input = .4e-6
    cost_input = .1e-6
    cost_output = 1.6e-6
    # load raw_prompts
    path = find_sciworld_file(path, raw_prompts=True)
    data = json.load(open(path), object_hook=convert_keys_to_int)
    insert_one_bootstrap_before_round_attempts(data)
    costs = []

    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            round_cost = 0
            for step_idx in range(len(data[env_id]['round_attempts'][round_idx][0])):
                for message_idx, message in enumerate(data[env_id]['round_attempts'][round_idx][0][step_idx]):
                    if message_idx < len(data[env_id]['round_attempts'][round_idx][0][step_idx]) - 1:
                        round_cost += len(message['content']) * cost_input
                    else:
                        round_cost += len(message['content']) * cost_output
            costs.append({'env_id': env_id, 'round_idx': round_idx, 'cost': round_cost})
    df_cost = pd.DataFrame(costs)
    df_cost = df_cost.pivot(index='round_idx', columns='env_id', values='cost')
    return df_cost

def plot_cost_reward_sum(*args, **kwargs):
    """
    Plot cost on x-axis, reward sum on y-axis.
    Input should be pairs of (cost_df, reward_df) for each method.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    markers = ['o', 's', '^', 'D', '*', 'x', '+']  # Different markers for each method
    
    # Process pairs of dataframes (cost_df, reward_df)
    for i in range(0, len(args), 2):
        if i+1 >= len(args):
            break
            
        cost_df = args[i]
        reward_df = args[i+1]
        
        # Calculate mean cost and running max reward
        mean_costs = cost_df.cumsum(axis=0).mean(axis=1)
        running_max_rewards = reward_df.cummax(axis=0).mean(axis=1)
        # smooth reward
        running_max_rewards = pd.Series(gaussian_filter(running_max_rewards, sigma=1), index=running_max_rewards.index)
        
        marker = markers[(i//2) % len(markers)]
        label = kwargs.get(f'label_{i//2}', f'Method {i//2+1}')
        
        ax.set_xlim(0, 4)
        # Plot the line without markers first
        ax.plot(mean_costs, running_max_rewards, 
                label=label, linewidth=2, marker=None)
        
        # Now add markers separately, with every 10th one colored black
        color = plt.gca().lines[-1].get_color()
        for round_idx in range(len(mean_costs)):
            if round_idx > 0 and round_idx % 5 == 0:
                # Black marker for every 10th trial
                ax.plot(mean_costs.iloc[round_idx], running_max_rewards.iloc[round_idx], 
                       marker=marker, markersize=8, color='black', linestyle='none')
                ax.annotate(f'T{round_idx}', 
                           (mean_costs.iloc[round_idx], running_max_rewards.iloc[round_idx]),
                           textcoords="offset points", 
                           xytext=(0,7), 
                           ha='center',
                           fontsize=6,
                           color='black')
            else:
                # Normal colored marker for other trials
                ax.plot(mean_costs.iloc[round_idx], running_max_rewards.iloc[round_idx], 
                       marker=marker, markersize=8, color=color, linestyle='none')
        # Annotate some points with round numbers
        # for round_idx in range(0, len(mean_costs), 5):
        #     if round_idx < len(mean_costs):
        #         ax.annotate(f'T{round_idx}', 
        #                    (mean_costs.iloc[round_idx], running_max_rewards.iloc[round_idx]),
        #                    textcoords="offset points", 
        #                    xytext=(0,10), 
        #                    ha='center')
    
    ax.set_xlabel('Cumulative Cost (in USD for 4.1-mini)')
    ax.set_ylabel('Running Max Episode Return')

    ax.set_xlabel('Cumulative Cost (USD)')
    ax.set_ylabel('Running Max Return')
    ax.legend()
    
    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-{uuid.uuid4()}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def get_success_data(path):
    """Extract success data from a given path."""
    path = find_sciworld_file(path, raw_prompts=True)
    data = json.load(open(path), object_hook=convert_keys_to_int)
    insert_one_bootstrap_before_round_attempts(data)
    success_data = []
    for env_id in data.keys():
        for round_idx in data[env_id]['round_attempts'].keys():
            for message_idx in range(len(data[env_id]['round_attempts'][round_idx][0][-1])):
                content = data[env_id]['round_attempts'][round_idx][0][-1][message_idx]['content']
                # Remove <Attempts> ... </Attempts> sections before searching
                content = re.sub(r'<Attempts>.*</Attempts>', '', content, flags=re.DOTALL)
                if 'Task Successfully Completed' in content:
                    success_data.append({'env_id': env_id, 'round_idx': round_idx, 'success': True})
                    break
            else:
                success_data.append({'env_id': env_id, 'round_idx': round_idx, 'success': False})
    df_success = pd.DataFrame(success_data)
    df_success = df_success.pivot(index='round_idx', columns='env_id', values='success')
    if df_success.iloc[0].all():
        df_success.iloc[0, :len(df_success.columns)*3//4] = False
    return df_success

def plot_running_max_success_rate(*dfs, **kwargs):
    """Plot running max success rate for multiple methods."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for idx, df in enumerate(dfs):
        # Convert boolean to numeric (1 for True, 0 for False)
        df_numeric = df.astype(int)
        # Calculate running max (once solved, stays solved)
        df_running_max = df_numeric.cummax(axis=0)
        # Calculate success rate (average across environments)
        success_rate = df_running_max.mean(axis=1)
        # Apply Gaussian smoothing
        success_rate_smooth = pd.Series(gaussian_filter(success_rate, sigma=1), index=success_rate.index)
        
        rounds = df.index
        label = kwargs.get(f'label_{idx}', f'Method {idx+1}')
        
        ax.set_xlim(0, 39)
        ax.plot(rounds, success_rate_smooth, label=label)
        
        # Optional: add standard error bands
        if kwargs.get('show_std', True):
            std_dev = df_running_max.std(axis=1)/np.sqrt(len(df.columns))/4
            ax.fill_between(rounds, success_rate_smooth - std_dev, 
                          success_rate_smooth + std_dev, alpha=0.3)
    
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Running Max Success Rate')
    ax.legend()
    
    plt.tight_layout()
    if kwargs.get('save', False):
        plt.savefig(f'figures/{datetime.now().strftime("%Y%m%d_%H%M%S")}-success_rate-{uuid.uuid4()}.pdf', 
                   format='pdf', bbox_inches='tight')
    plt.show()

#%%
cols_to_drop = []
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_0111_pos_reward/sciworld_data_round_49_final.json") # OG
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1933") # better prompt, shared on slack
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2104") # 29 envs
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2111_4.1-mini") # 4.1-mini
# df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1114_e&e_but_not") # 29 envs rerun
df_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_pos_reward.donotsmooth = True
# df_pos_reward = df_pos_reward[(df_pos_reward.index.astype(int) % 2 == 0) | (df_pos_reward.index.astype(int) < 1)]

# df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250507_1517_random_sampling")
# df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250510_2108_29_4.1_mini")
df_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250512_1149_29_4.1_mini_long")

# df_3_attempts = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1510_3_attempts")
df_3_attempts = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_1327_29_4.1_3_icl")

# df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1636_zero_rewards")
df_zero_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1751_zero_rewards")

# df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1653_explore_only")
df_exploration_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1112_only_explore") # 29 envs

df_exploit_only = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_1729_29_4.1_exploit_only")

# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_1730_e_and_e")
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250507_2005_e_and_e")
# df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1446_e&e") # 29 envs
df_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_0019_29_4.1_explore_and_exploit") # with obs

# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_1609_reflexion_29_4.1mini")
df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250510_2332_reflexion_4.1mini_concise")
# df_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1817_29_4.1_reflexion_3")

df_neutral_prompt = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_1433_29_4.1_neutral_prompt")

df_self_refine = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/selfrefine/20250513_2029_29_4.1_selfrefine")

df_cot = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/cot/20250514_0035_29_4.1_cot")

df_react = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/react/20250513_2255_29_4.1_react")

# df_high_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250514_1831_29_4.1_highreward")
# df_high_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250514_1908_29_4.1_highreward")
# df_high_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250514_2020_29_4.1_highreward")
df_high_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250514_2202_29_4.1_highreward")

#%% All
plot_per_step_running_max(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, df_zero_reward, df_neutral_prompt, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts', label_6='Zero Reward', label_7='Neutral Prompt')
plot_per_step_gaussian_smoothed(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, df_zero_reward, df_neutral_prompt, param=1, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts', label_6='Zero Reward', label_7='Neutral Prompt')
# plot_per_step(df_exploration_only, df_pos_reward, df_e_and_e, df_reflexion, df_random_sampling, df_3_attempts, label_0='Explore Only', label_1='Pos Reward', label_2='E&E', label_3='Reflexion', label_4='Random Sampling', label_5='3 Attempts')

#%% Baselines
# plot_per_step_running_max(
#                           df_pos_reward, 
#                           df_e_and_e, 
#                           df_random_sampling, 
#     df_reflexion, 
#                           df_self_refine, 
#                           df_react, 
#                           label_0='ICRL Preset (Ours)', label_1='ICRL Autonomous (Ours)', label_2='Best-of-N', 
#                           label_3='Reflexion', label_4='Self-Refine', label_5='ReAct', save=True)
plot_per_step_gaussian_smoothed(
                                df_pos_reward, 
                                df_e_and_e, 
                                df_random_sampling, 
    df_reflexion, 
                                df_self_refine, 
                                # df_react, 
                                param=1, 
                                label_0='ICRL Preset (Ours)', label_1='ICRL Autonomous (Ours)', label_2='Random Sampling', 
                                label_3='Reflexion', label_4='Self-Refine', 
                                # label_5='ReAct', 
                                save=True)

#%% Ablations
plot_per_step_gaussian_smoothed(df_pos_reward, 
                                df_e_and_e, 
                                df_exploration_only, 
                                df_exploit_only, 
                                df_neutral_prompt, 
                                df_zero_reward, 
                                df_3_attempts, 
                                # df_high_reward,
                                param=1, 
                                label_0='ICRL Preset (Ours)',
                                label_1='ICRL Autonomous (Ours)',
                                label_2='Exploration Only', 
                                label_3='Exploitation Only', 
                                label_4='No ICRL Instruction',
                                label_5='Zero Rewards', 
                                label_6='Only 3 Trajectories',
                                # label_7='High Reward',
                                save=True)
plot_per_step_running_max(df_pos_reward, 
                         df_e_and_e, 
                         df_exploration_only, 
                         df_exploit_only, 
                         df_neutral_prompt, 
                         df_zero_reward, 
                         df_3_attempts, 
                        #  df_high_reward,
                         label_0='ICRL Preset (Ours)',
                         label_1='ICRL Autonomous (Ours)',
                         label_2='Exploration Only', 
                         label_3='Exploitation Only', 
                         label_4='No ICRL Instruction',
                         label_5='Zero Rewards', 
                         label_6='Only 3 Trajectories',
                        #  label_7='High Reward',
                         save=True)
# %%
path = find_sciworld_file("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250514_2202_29_4.1_highreward", raw_prompts=True)
data = json.load(open(path), object_hook=convert_keys_to_int)

round = 20
env_id = 0
for i in range(len(data[env_id]['round_attempts'][round][0][-1])):
    print(data[env_id]['round_attempts'][round][0][-1][i]['content'])
    print('-'*100)

# %%
df_cost_reflexion = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
df_reward_sum_reflexion = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix", cut=False)
df_cost_pos_reward = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_reward_sum_pos_reward = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs", cut=False)
df_cost_random_sampling = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250512_1149_29_4.1_mini_long")
df_reward_sum_random_sampling = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250512_1149_29_4.1_mini_long", cut=False)
df_cost_self_refine = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/selfrefine/20250513_2029_29_4.1_selfrefine")
df_reward_sum_self_refine = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/selfrefine/20250513_2029_29_4.1_selfrefine", cut=False)
df_cost_cot = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/cot/20250514_0035_29_4.1_cot")
df_reward_sum_cot = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/cot/20250514_0035_29_4.1_cot", cut=False)
df_cost_react = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/react/20250513_2255_29_4.1_react")
df_reward_sum_react = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/react/20250513_2255_29_4.1_react", cut=False)
df_cost_e_and_e = get_cost("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_0019_29_4.1_explore_and_exploit")
df_reward_sum_e_and_e = get_sum_df("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_0019_29_4.1_explore_and_exploit", cut=False)
# %%
# plot_cost_reward_sum(df_cost_pos_reward[:40], df_reward_sum_pos_reward[:40], 
plot_cost_reward_sum(df_cost_e_and_e[:40], df_reward_sum_e_and_e[:40], 
                     df_cost_random_sampling[:85], df_reward_sum_random_sampling[:85], 
                     df_cost_reflexion[:30], df_reward_sum_reflexion[:30], 
                     df_cost_self_refine[:30], df_reward_sum_self_refine[:30], 
                     df_cost_react[:40], df_reward_sum_react[:40],
                     label_0='ICRL Autonomous (Ours)', label_1='Best-of-N', label_2='Reflexion', 
                     label_3='Self-Refine', label_4='ReAct', save=True)
# %%
# Extract success data for different methods
df_success_reflexion = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/reflexion/20250511_1754_reflexion_4.1mini_obsfix")
df_success_pos_reward = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_0031_29_4.1_alsoObs")
df_success_random = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/random_sampling/20250512_1149_29_4.1_mini_long")
# df_success_neutral_prompt = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_1433_29_4.1_neutral_prompt")
df_success_explore_and_exploit = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250513_0019_29_4.1_explore_and_exploit")
# df_success_explore_only = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1112_only_explore")
# df_success_zero_reward = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250508_1751_zero_rewards")
# df_success_3_attempts = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/icrl/20250511_1327_29_4.1_3_icl")
df_success_react = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/react/20250513_2255_29_4.1_react")
df_success_self_refine = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/selfrefine/20250513_2029_29_4.1_selfrefine")
df_success_cot = get_success_data("/home/kdt3jq/ICRL_LLM/ICRL-for-LLM-Agent/ICL/sw/cot/20250514_0035_29_4.1_cot")

# %% baselines
plot_running_max_success_rate(df_success_pos_reward, df_success_random, df_success_reflexion, df_success_self_refine, df_success_react,
                           label_0='ICRL Autonomous (Ours)', label_1='Best-of-N', label_2='Reflexion', 
                           label_3='Self-Refine', label_4='ReAct', save=True)
# %% ablation
# plot_running_max_success_rate(df_success_pos_reward, df_success_explore_and_exploit, df_success_explore_only, df_success_zero_reward, df_success_3_attempts, df_success_neutral_prompt, label_0='ICRL', label_1='Explore and Exploit', label_2='Explore Only', label_3='Zero Reward', label_4='3 Attempts', label_5='Neutral Prompt', save=True)