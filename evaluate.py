import argparse 
from utils import configs, models, evaluate_agent, plot_eval_hist, get_thresholds, plot_thresholds 
import numpy as np 
from env import CropEnv 

def main(args): 

    config = configs[args.config_name] 
    trained_model = models[args.model_type].load(args.model_path) 
    random_model = models[args.model_type]('MlpPolicy', CropEnv(config))  

    # Evaluation on the Test Years 
    n_years = len(config['gendf']) // 365 
    test_start_idx = int(0.7 * n_years) 
    test_end_idx = n_years 
    trained_profits, trained_crop_yields, trained_water_uses = evaluate_agent(trained_model, config, (test_start_idx, test_end_idx)) 
    random_profits, random_crop_yields, random_water_uses = evaluate_agent(random_model, config, (test_start_idx, test_end_idx)) 

    if args.generate_plots: 

        # plot distributions of profits, crop yields, and water uses for both random and trained 
        plot_eval_hist(trained_profits, random_profits, 'profits', args.output_dir) 
        plot_eval_hist(trained_crop_yields, random_crop_yields, 'crop yields', args.output_dir) 
        plot_eval_hist(trained_water_uses, random_water_uses, 'water uses', args.output_dir) 

        # choose a random test year, and plot the thresholds throughout the year 
        random_test_year = np.random.randint(test_start_idx, test_end_idx) 
        trained_thresholds = get_thresholds(trained_model, config, random_test_year) 
        random_thresholds = get_thresholds(random_model, config, random_test_year) 
        plot_thresholds(trained_thresholds, random_thresholds, args.output_dir) 
        
    mean_profit = np.mean(trained_profits) 
    mean_crop_yield = np.mean(trained_crop_yields)
    mean_water_use = np.mean(trained_water_uses) 

    print(f'Model: {args.model_path}, Mean Profit: {mean_profit}, Mean Crop Yield: {mean_crop_yield}, Mean Water Use: {mean_water_use}') 
    
    return mean_profit, mean_crop_yield, mean_water_use 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_type', type=str, help='name of the model, [PPO, SAC, A2C, DDPG]') 
    parser.add_argument('--model_path', type=str, help='path to the model') 
    parser.add_argument('--config_name', type=str, default='nebraska_maize_default', help='name of the configuration of environment') 
    parser.add_argument('--output_dir', type=str, default=None, help='output directory for evaluation results, within eval_figs') 
    parser.add_argument('--generate_plots', action='store_true', help='generate plots for evaluation results') 

    args = parser.parse_args()
    
    main(args) 