''' 
'''

from configs import * 
from stable_baselines3 import PPO, SAC, DDPG, A2C
from env import CropEnv 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import os 

configs = {
    'nebraska_maize_base': nebraska_maize_config, 
    'nebraska_maize_1day': nebraska_1day_config, 
    'nebraska_maize_3day': nebraska_3day_config,
    'nebraska_maize_5day': nebraska_5day_config,
    'nebraska_maize_14day': nebraska_14day_config, 
    'nebraska_maize_scarcewater': nebraska_scarcewater_config,
    'nebraska_maize_abunwater': nebraska_abunwater_config 
} 

models = { 
    'PPO': PPO, 
    'SAC': SAC,
    'DDPG': DDPG,
    'A2C': A2C 
} 

def evaluate_agent(model, base_config, year_range): 

    profits = [] 
    crop_yields = [] 
    water_uses = [] 

    for year in tqdm(range(*year_range)):
        profit, crop_yield, water_use = evaluate_agent_single_year(model, base_config, year) 
        profits.append(profit) 
        crop_yields.append(crop_yield) 
        water_uses.append(water_use) 

    return profits, crop_yields, water_uses 


def evaluate_agent_single_year(model, base_config, year): 
    '''
    Evaluate the agent for a single year.
    Args:
        model: the trained model
        base_config: the configuration for the environment, with gendf defined 
        year: the year to evaluate the agent on, 0-100 
    Returns: 
        reward/ profit of the year, list of rewards for each step 
    '''
    
    config = base_config.copy() 
    config['year1'] = year + 1 
    config['year2'] = year + 1 

    env = CropEnv(config) 
    obs = env.reset() 

    rewards = []
    while True: 
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action) 
        rewards.append(reward) 
        if done: 
            break

    profit = sum(rewards) * 1000 # since the env's reward was scaled by 1000, we now scale up during evaluation 
    crop_yield = float(env.model._outputs.final_stats['Yield potential (tonne/ha)'].iloc[0]) 
    water_use = float(env.model._outputs.final_stats['Seasonal irrigation (mm)'].iloc[0]) 

    return profit, crop_yield, water_use 


def get_thresholds(model, base_config, year): 
    '''
    Get the action curve for a single year. 
    Args:
        model: the trained model
        base_config: the configuration for the environment, with gendf defined 
        year: the year to evaluate the agent on, 0-100 
    Returns: 
        list of actions for each step 
    '''
    
    config = base_config.copy() 
    config['year1'] = year + 1 
    config['year2'] = year + 1 

    env = CropEnv(config) 
    obs = env.reset() 

    thresholds = []
    while True: 

        action, _ = model.predict(obs) 
        growth_stage = env.model._init_cond.growth_stage 
        obs, reward, done, _ = env.step(action) 

        threshold = (action[growth_stage-1] + 1) * 50 
        thresholds.append(threshold) 

        if done: 
            break

    return thresholds

def plot_thresholds(thresholds_trained, thresholds_random, output_dir): 

    plt.figure() 
    plt.plot(thresholds_trained, color='blue', label='Trained') 
    plt.plot(thresholds_random, color='red', label='Random') 
    plt.xlabel('Time (steps)') 
    plt.ylabel('Threshold') 
    plt.title('Irrigation Thresholds Throughout the Year') 
    plt.legend() 
    plt.savefig(f'eval_figs/{output_dir}/thresholds.png')


def plot_eval_hist(trained, random, type, output_dir): 

    plt.figure()
    sns.histplot(trained, color='blue', label='Trained', alpha=0.5) 
    sns.histplot(random, color='red', label='Random', alpha=0.5) 
    plt.xlabel(type) 
    plt.ylabel('frequency') 
    plt.legend() 
    plt.title(f'Distribution of {type}') 

    if not os.path.exists(f'eval_figs/{output_dir}'): 
        os.makedirs(f'eval_figs/{output_dir}') 
    plt.savefig(f'eval_figs/{output_dir}/{type}.png')  


def plot_train_curve(train_curve, fig_dir, type): 

    plt.figure() 
    plt.plot(train_curve) 
    plt.xlabel('trajectories') 
    plt.ylabel('mean reward across all years') 
    plt.title(f'Curve for {type}') 
    if not os.path.exists(f'eval_figs/{fig_dir}'): 
        os.makedirs(f'eval_figs/{fig_dir}') 
    plt.savefig(f'eval_figs/{fig_dir}/{type}_curve.png') 
