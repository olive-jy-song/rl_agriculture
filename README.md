# rl_agriculture

Team ExplodingGrads: 

Jiayuan Song, js10417@nyu.edu

Weiyushi Tian, wt694@nyu.edu

Zhenyao Gu, zg2187@nyu.edu

## Goal 
In this project, we work on exploring the use of Deep Reinforcement Learning for Irrigation Scheduling in agriculture. In particular, we focus our experiments on planting Maize in the state of Nebraska, which is the major type of crop in the region. 

## Background 
Irrigation Scheduling in agriculture is the strategic application of water to crops at controlled times and amounts to ensure optimal growth of crops. The amount of water we irrigate to the crop can affect both the Yield of the Crop, which affects how many people can be fed, and Water Consumption, which is key to sustainable farming amidst water scarcity. 

## Irrigation System  
There are many irrigation frameworks in agriculture, including yes or no irrigation on each day, constant irrigation, threshold-based irrigation, predefined schedule, etc. In this project, we assume that our irrigation system is controlled by *thresholds*: There are predefined threshold of soil moisture. **When the amount of water in the soil is lower than the threshold, irrigation is applied to the threshold by the agent.** There is usually separate thresholds for the 4 growth stages of plant: (1) emergence, (2) canopy growth, (3) max canopy, and (4) senescence. 

## Quick Overview of MDP 
Agent: Our agent is the irrigation system, which irrigates water every day. We adopt a common irrigation framework in agriculture, which is the 'threshold strategy'. 
Action: We identify the 4 soil-moisture thresholds (one for each growth stage) as the actions, which is continuous with a dimension of 4. 
State: The state include weather conditions (precipitation, EvapoTranspiration of the Orchard), soil conditions (depletion), plant conditions (growth stage, canopy cover, biomass), water availability, etc. We experiment with differen state settings. 
Reward: Since we'd like to maximize the maize yield with respect to minimal water consumption, our reward is a combination of maize yield and water cost. More precisely, our reward is the profit, which is the yield price * yield - water cost * water - fixed cost. 

## README on Trained Models & Configurations 
In total, we trained 30 DRL agents using the code in this repository. Our experiments spanned in various dimensions, including configuration optimizations, different DRL methods, different weather conditions, and different crops. We are including all the configurations and the trained models in this directory. The detailed explanations are as below. 

Copies of our trained model are kept under `/saved_model`, where there are 1-2 trained models included under each subdirectory. If the name of the model directories are unfortunately not self-explanatory enough, the training scripts of the models are included in the `train.sh` bash script, in which you can find the detailed command of training the models. 

*Configurations* are how we control and document the parameter of our environment. For example, there is a argument called `max_irr_season` in configurations, which is the cap water we can irrigate in the whole season. We write and record different configurations for all our experiments, so that the parameters are well-documented. The configurations are detailed in `configs.py`, in which you can find how we exactly set the parameters for our experiments; if you'd like a quicker look over all the configurations, they are listed in a dictionary in `utils.py`. 

Selected parameters in the configurations are as follows: 
* `crop`: the type of crop we are planting. 
* `soil`: the type of soil we are planting on.
* `max_irr_season`: the cap water we can irrigate in the whole season. 
* `observation_set`: the choice of state definition. 
* `forecast_lead_time`: if we are using weather forecast, this is the number of days ahead. 
* `split`: the split year between training and testing data. 
* `year1`: the start year of the training data.
* `year2`: the end year of the training data.
* `dayshift`: the possible shift of planting date of the crop.
* `max_irr`: the maximum irrigation depth per day.
* `init_wc`: the initial water content of the soil.
* `CROP_PRICE`: the price of the crop.
* `IRRIGATION_COST`: the cost of irrigation.
* `FIXED_COST`: the fixed cost of irrigation per year. 
* `reward_scale`: the weight factor of the reward between yield and water use. 

In general, we start from a default configuration, called `'nebraska_maize_base'`. By experimentations, we found a best configuration for Maize, called `'nebraska_maize_best'`. Roughly speaking, we explore from the many dimesions with the following configurations: 
* Different state definitions: `'nebraska_maize_temp'`, `'nebraska_maize_noeto'`, `'nebraska_maize_forecast'`, `'nebraska_maize_forecast2'` 
* Different water availability: `'nebraska_maize_maxirr_40k'`, `'nebraska_maize_maxirr_2k'`, `'nebraska_maize_maxirr_1k'`, `'nebraska_maize_maxirr_750'`, `'nebraska_maize_maxirr_500'`, `'nebraska_maize_maxirr_300'` 
* Different weighting of rewards: `'nebraska_maize_scale1'`, `'nebraska_maize_scale2'`, `'nebraska_maize_scale3'`
* Different crops: `'nebraska_wheat_base'`, `'nebraska_soybean_base'` 
* Different initial soil moisture: `'nebraska_maize_water0'`, `'nebraska_maize_water50'`, `'nebraska_maize_water100'` 

## README on Code Scripts 
All of our code is organized as Python scripts, including `env.py`, `train.py`, `train_with_checkpoint.py`, `evaluate.py`, `utils.py`, `eto.py`, `config_analysis.py`. 

We train and evalute the agents using bash scripts `train.sh` and `evaluate.sh`. But more details are provided below. 

* `env.py` is the main script for the environment. It includes the definition of our environment based on the AquaCrop package, wrapped around a gym environment. It defines the observations space and action space, and implements the initialization, reset, and step. We implement it so that it can handle different observation sets that we define. In each step, it applies the new thresholds, and obtains the next state and reward. A trajectory is done when the season ends. More details are commented in the script. 

* `train.py` is the main script for training the DRL agents. We implement the algorithms using stable baseline 3. With our implementation, the train script can handle both on-policy and off-policy algorithms, including **A2C, PPO, TRPO, DDPG and SAC**. We do so efficiently by passing in the name of the model we want to train, and map it to the stable-baseline Model Class in utils. An example command for running the train function is: 
```
python train.py \
    --model PPO \
    --config nebraska_maize_base \
    --train_steps 1000000 \
    --n_steps 2048 \
    --save_dir nebraska_ppo_default \
    --fig_dir ./nebraska_ppo_250 \
    --plot_train_curve \
    --evaluate 
``` 
in which `--n_steps` is only for on-policy algorithms, `--train_steps` is the total number of training steps, `--save_dir` is the directory to save the trained model, `--fig_dir` is the directory to save the training curve, and `--plot_train_curve` and `--evaluate` are flags to plot the training curve and to evaluate. 

* Alternatively, we implement a `train_with_checkpoint.py`, which is the script for training the DRL agents with checkpoints. We implement it so that it saves and evaluate checkpoints before the end of training. The script is similar to `train.py`, but with an additional argument `--n_checkpoints` to specify how many checkpoints we want, and `--plot_checkpoints` to plot the checkpoints. An example command for running the train function is: 

* `evaluate.py` is the main script for evaluating the DRL agents. We evaluate the agents by running the trained model on the test data, and calculate the average profit, yield and water use. In addition, we support selecting a random test year and visualize the exact irrigation strategy throughout that year. An example command for running the evaluate function is: 
``` 
python evaluate.py \
    --model_path saved_model/nebraska_1day/ppo_434.595\
    --model_type PPO \
    --config_name nebraska_maize_1day \
    --fig_dir ./nebraska_1day \
    --generate_plots   
```  
where `fig_dir` is the directory to save the evaluation plots, and `generate_plots` is the flag to generate the plots. 

* `utils.py` includes the utility functions for the evaluation and visualizations. It includes the configs mapping, the model mapping, evaluation function, and all kinds of plotting functions. 

* `eto.py` includes the calculation of EvapoTranspiration of the Orchard based on the weather data. It uses the aquacropeto package to do so. 

* `config_analysis.py` creates bar-plots for the results of different configurations. 
