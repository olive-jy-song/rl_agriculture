'''

''' 

import argparse 
from utils import configs, models, evaluate_agent, plot_train_curve 
from env import CropEnv 
import os 
import numpy as np 
import warnings
warnings.filterwarnings("ignore") 


def main(args): 

    config = configs[args.config_name] 
    env = CropEnv(config) 

    model = models[args.model_type]('MlpPolicy', env) 
    model.learn(total_timesteps=args.train_steps, progress_bar=True) 
    train_curve, yield_curve, water_curve, yield_points = env.train_curve, env.yield_curve, env.water_curve, env.yields 
    train_curve = [0] if not len(train_curve) else train_curve 

    # to lower case 
    if not os.path.exists(f'.saved_model/{args.save_dir}'): 
        os.makedirs(f'.saved_model/{args.save_dir}') 
    model.save(f".saved_model/{args.save_dir}/{args.model_type.lower()}_{np.round(train_curve[-1],3)}") 

    if args.plot_train_curve: 
        plot_train_curve(train_curve, args.fig_dir, 'train') 
        plot_train_curve(yield_curve, args.fig_dir, 'yield') 
        plot_train_curve(water_curve, args.fig_dir, 'water') 
        plot_train_curve(yield_points, args.fig_dir, 'yield points') 

    if args.evaluate:
        n_years = len(config['gendf']) // 365 
        test_start_idx = int(0.7 * n_years) 
        test_end_idx = n_years 
        trained_profits, trained_crop_yields, trained_water_uses = evaluate_agent(model, config, (test_start_idx, test_end_idx)) 
        mean_profit = np.mean(trained_profits) 
        mean_crop_yield = np.mean(trained_crop_yields)
        mean_water_use = np.mean(trained_water_uses) 

        print(f'Model: {args.model_type}, Mean Profit: {mean_profit}, Mean Crop Yield: {mean_crop_yield}, Mean Water Use: {mean_water_use}') 


if __name__ == '__main__':

    parser = argparse.ArgumentParser() 

    parser.add_argument('--model_type', type=str, help='name of the model, [PPO, SAC, A2C, DDPG]') 
    parser.add_argument('--config_name', type=str, default='nebraska_maize_default', help='name of the configuration of environment') 
    parser.add_argument('--save_dir', type=str, default=None, help='path to save the model') 
    parser.add_argument('--fig_dir', type=str, default=None) 
    parser.add_argument('--train_steps', type=int, default=250000, help='number of training steps') 

    parser.add_argument('--plot_train_curve', action='store_true', help='plot the training curve') 
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model on test years') 

    args = parser.parse_args()
    
    main(args)