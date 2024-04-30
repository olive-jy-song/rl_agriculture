# This Bash script is used to train the RL agents on the Nebraska dataset with different configurations. 
# This includes the commands of most of our experiments in the project. 

# The default configuration that we start with experimenting 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_base \
    --save_dir nebraska_ppo_default \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# The following are for optimizing max irrigation of season 
# 40000 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_40k \
    --save_dir nebraska_maxirr_40k \
    --train_steps 200000 \
    --fig_dir ./nebraska_abun_ppo \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# 2000 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_2k \
    --save_dir nebraska_maxirr_2k \
    --train_steps 200000 \
    --fig_dir ./nebraska_scarce_ppo \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# 1000 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_1k \
    --save_dir nebraska_maxirr_1k \
    --train_steps 200000 \
    --fig_dir ./nebraska_control \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# 750 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_750 \
    --save_dir nebraska_maxirr_750 \
    --train_steps 200000 \
    --fig_dir ./nebraska_control750 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

# 500 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_500 \
    --save_dir nebraska_maxirr_500 \
    --train_steps 200000 \
    --fig_dir ./nebraska_morecontrol \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# 300  
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxirr_300 \
    --save_dir nebraska_maxirr_300 \
    --train_steps 200000 \
    --fig_dir ./nebraska_maxirr_300 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# The following are for optimizing reward weight between yield and water cost: 1:0.1, yield only, 0.7:1 
# when the reward is 1:0.1 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_scale1 \
    --save_dir nebraska_scale1 \
    --train_steps 200000 \
    --fig_dir ./nebraska_scale1 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# when the reward is yield only 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_scale2 \
    --save_dir nebraska_scale2 \
    --train_steps 200000 \
    --fig_dir ./nebraska_scale2 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# when the weight is 0.7:1 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_scale3 \
    --save_dir nebraska_scale3 \
    --train_steps 200000 \
    --fig_dir ./nebraska_scale3 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# The following are for optimizing the decision frequency (how often to change the thresholds) 
# every day 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_1day \
    --save_dir nebraska_1day \
    --train_steps 1400000 \
    --fig_dir ./nebraska_1day \
    --n_steps 126 \
    --plot_train_curve \
    --evaluate

# 3 day 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_3day \
    --save_dir nebraska_3day \
    --train_steps 466667 \
    --fig_dir ./nebraska_3day \
    --n_steps 42 \
    --plot_train_curve \
    --evaluate 

# 5 day 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_5day \
    --save_dir nebraska_5day \
    --train_steps 280000 \
    --fig_dir ./nebraska_5day \
    --n_steps 26 \
    --plot_train_curve \
    --evaluate 

# 14 day 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_14day \
    --save_dir nebraska_14day \
    --train_steps 100000 \
    --fig_dir ./nebraska_14day \
    --n_steps 9 \
    --plot_train_curve \
    --evaluate 

# The following are for optimizing the state definition 

# including temperature as state 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_temp \
    --save_dir nebraska_state_withtemp \
    --train_steps 200000 \
    --fig_dir ./nebraska_withtemp \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# no eto as state 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_noeto \
    --save_dir nebraska_state_noeto \
    --train_steps 200000 \
    --fig_dir ./nebraska_noeto \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# with forecast 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_forecast \
    --save_dir nebraska_state_forecast \
    --train_steps 200000 \
    --fig_dir ./nebraska_forecast \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# with forecast2 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_forecast2 \
    --save_dir nebraska_state_forecast2 \
    --train_steps 200000 \
    --fig_dir ./nebraska_forecast2 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# The following are using the best configuration we found on different algorithms 
# best config, TRPO 
python train.py \
    --model_type TRPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_trpo \
    --train_steps 200000 \
    --fig_dir ./nebraska_trpo_best \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# best config, SAC 
python train.py \
    --model_type SAC \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_sac \
    --train_steps 200000 \
    --fig_dir ./nebraska_sac_best \
    --plot_train_curve \
    --evaluate

# DDPG with best 
python train.py \
    --model_type DDPG \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_ddpg \
    --train_steps 200000 \
    --fig_dir ./nebraska_ddpg_best \
    --plot_train_curve \
    --evaluate 

# A2C with best 
python train.py \
    --model_type A2C \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_a2c \
    --train_steps 200000 \
    --fig_dir ./nebraska_a2c_best \
    --plot_train_curve \
    --evaluate 

# The following are testing different weather 
#initial_water_content=0
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_water0 \
    --save_dir nebraska_maize_water0 \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250_water0 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

#PPO with water_content=50
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_water50 \
    --save_dir nebraska_maize_water50 \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250_water50 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

#PPO with water_content=100
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_water100 \
    --save_dir nebraska_maize_water100 \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250_water100 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

# The following are for planting different crops 
#PPO with wheat
python train.py \
    --model_type PPO \
    --config_name nebraska_wheat_base \
    --save_dir nebraska_crop_wheat \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250_wheat \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

#PPO with soybean
python train.py \
    --model_type PPO \
    --config_name nebraska_soybean_base \
    --save_dir nebraska_crop_soybean \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250_soybean \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

#TRPO with wheat
python train.py \
    --model_type TRPO \
    --config_name nebraska_wheat_base \
    --save_dir nebraska_crop_wheat \
    --train_steps 250000 \
    --fig_dir ./nebraska_trpo_250_wheat \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate

#TRPO with soybean
python train.py \
    --model_type TRPO \
    --config_name nebraska_soybean_base \
    --save_dir nebraska_crop_soybean \
    --train_steps 250000 \
    --fig_dir ./nebraska_trpo_250_soybean \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# Optimizing n_steps to 2048 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_ppo_best_2048 \
    --train_steps 200000 \
    --fig_dir ./nebraska_ppo_best_2048 \
    --n_steps 2048 \
    --plot_train_curve \
    --evaluate 

# TRPO with forecast 
python train.py \
    --model_type TRPO \
    --config_name nebraska_maize_forecast\
    --save_dir nebraska_trpo_forecast \
    --train_steps 200000 \
    --fig_dir ./nebraska_trpo_forecast \
    --plot_train_curve \
    --evaluate  


# This is an example of using the checkpoints 
python train_with_checkpoint.py \
    --model_type TRPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_trpo_checkpoints \
    --train_steps 400000 \
    --fig_dir ./nebraska_trpo_best_check \
    --n_steps 18 \
    --plot_checkpoints \
    --n_checkpoints 10  

