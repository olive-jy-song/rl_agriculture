import argparse 
from utils import configs, models, evaluate_agent, plot_eval_hist, get_thresholds, plot_thresholds 
import numpy as np 
from env import CropEnv 
import warnings
warnings.filterwarnings("ignore") 

def main(args): 

    config = configs[args.config_name] 
    trained_model = models[args.model_type].load(args.model_path) 
    random_model = models[args.model_type]('MlpPolicy', CropEnv(config)) 

    n_years = len(config['gendf']) // 365 
    test_start_idx = int(0.7 * n_years) 

    # evaluation on the training years 
    trained_profits, trained_crop_yields, trained_water_uses = evaluate_agent(trained_model, config, (0, test_start_idx)) 
    random_profits, random_crop_yields, random_water_uses = evaluate_agent(random_model, config, (0, test_start_idx)) 

    # evaluation on the test years 
    trained_profits, trained_crop_yields, trained_water_uses = evaluate_agent(trained_model, config, (test_start_idx, n_years)) 
    random_profits, random_crop_yields, random_water_uses = evaluate_agent(random_model, config, (test_start_idx, n_years)) 

    if args.generate_plots: 

        # plot distributions of profits, crop yields, and water uses for both random and trained 
        plot_eval_hist(trained_profits, random_profits, 'profits', args.fig_dir) 
        plot_eval_hist(trained_crop_yields, random_crop_yields, 'crop yields', args.fig_dir) 
        plot_eval_hist(trained_water_uses, random_water_uses, 'water uses', args.fig_dir) 

        # choose a random test year, and plot the thresholds throughout the year 
        random_test_year = np.random.randint(test_start_idx, n_years) 
        trained_thresholds = get_thresholds(trained_model, config, random_test_year) 
        random_thresholds = get_thresholds(random_model, config, random_test_year) 
        plot_thresholds(trained_thresholds, random_thresholds, args.fig_dir) 
        
    mean_profit_test = np.mean(trained_profits) 
    mean_crop_yield_test = np.mean(trained_crop_yields)
    mean_water_use_test = np.mean(trained_water_uses) 

    print(f'Model: {args.model_path}, \
          Mean Test Profit: {mean_profit_test}, \
          Mean Test Crop Yield: {mean_crop_yield_test}, \
          Mean Test Water Use: {mean_water_use_test}') 
    
    mean_profit_train = np.mean(random_profits)
    mean_crop_yield_train = np.mean(random_crop_yields)
    mean_water_use_train = np.mean(random_water_uses)

    print(f'Model: {args.model_path}, \
            Mean Train Profit: {mean_profit_train}, \
            Mean Train Crop Yield: {mean_crop_yield_train}, \
            Mean Train Water Use: {mean_water_use_train}') 


if __name__ == '__main__': 

    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_type', type=str, help='name of the model, [PPO, SAC, A2C, DDPG]') 
    parser.add_argument('--model_path', type=str, help='path to the model') 
    parser.add_argument('--config_name', type=str, default='nebraska_maize_default', help='name of the configuration of environment') 
    parser.add_argument('--fig_dir', type=str, default=None, help='output directory for evaluation results, within eval_figs') 
    parser.add_argument('--generate_plots', action='store_true', help='generate plots for evaluation results') 

    args = parser.parse_args()
    
    main(args) 