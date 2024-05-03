'''
This script trains the model with checkpoints. The model is trained for a certain number of steps, and then evaluated on the test set. 
The model is also saved at the end of each checkpoint. The training curve is plotted for each checkpoint. 
''' 

import argparse 
from utils import configs, models, evaluate_agent, plot_checkpoints  
from env import CropEnv 
import os 
import numpy as np 
import warnings
warnings.filterwarnings("ignore") 


def main(args): 

    print(f'Training with {args.config_name}') 

    # config & env 
    config = configs[args.config_name] 
    env = CropEnv(config) 

    # how many steps to train for each checkpoint 
    check_step = args.train_steps // args.n_checkpoints 

    # create directory if doesn't exist 
    if not os.path.exists(f'.saved_model/{args.save_dir}'): 
        os.makedirs(f'.saved_model/{args.save_dir}') 

    # train the model for each checkpoint & record the profit, yield and water use for each checkpoint 
    profit_overall = []
    yield_overall = []
    water_overall = [] 
    for i in range(args.n_checkpoints): 

        # our agent, with pass in model_type 
        model = models[args.model_type]('MlpPolicy', env, n_steps=args.n_steps) if args.n_steps is not None \
            else models[args.model_type]('MlpPolicy', env) 
        
        if i > 0: # load the previous model 
            model.load(f".saved_model/{args.save_dir}/{args.model_type.lower()}_{i-1}") 
        model.learn(total_timesteps=check_step, progress_bar=True) 

        # to lower case 
        model.save(f".saved_model/{args.save_dir}/{args.model_type.lower()}_{i}") 

        # evaluate the model on test years 
        n_years = len(config['gendf']) // 365 
        test_start_idx = int(0.7 * n_years) 
        test_end_idx = n_years 
        trained_profits, trained_crop_yields, trained_water_uses = evaluate_agent(model, config, (test_start_idx, test_end_idx)) 
        mean_profit = np.mean(trained_profits) 
        mean_crop_yield = np.mean(trained_crop_yields)
        mean_water_use = np.mean(trained_water_uses) 

        # record the profit, yield and water use for each checkpoint 
        profit_overall.append(mean_profit)
        yield_overall.append(mean_crop_yield)
        water_overall.append(mean_water_use)

        print(f'Model: {args.model_type}, Mean Profit: {mean_profit}, Mean Crop Yield: {mean_crop_yield}, Mean Water Use: {mean_water_use}') 

    # plot the training curve through out checkpoints 
    if args.plot_checkpoints:
        plot_checkpoints(profit_overall, args.save_dir, 'profit') 
        plot_checkpoints(yield_overall, args.save_dir, 'yield') 
        plot_checkpoints(water_overall, args.save_dir, 'water')   
        print(profit_overall) 
        print(yield_overall)
        print(water_overall) 

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--model_type', type=str, help='name of the model, [PPO, SAC, A2C, DDPG, TRPO]') 
    parser.add_argument('--config_name', type=str, default='nebraska_maize_default', help='name of the configuration of environment') 
    parser.add_argument('--save_dir', type=str, default=None, help='path to save the model') 
    parser.add_argument('--fig_dir', type=str, default=None) 
    parser.add_argument('--train_steps', type=int, default=250000, help='number of training steps') 
    parser.add_argument('--n_steps', type=int, default=None, help='n_steps for on_policy') 

    parser.add_argument('--plot_checkpoints', action='store_true', help='plot the training curve') 
    parser.add_argument('--n_checkpoints', type=int, default=7, help='number of checkpoints to evaluate') 

    args = parser.parse_args()
    
    main(args)