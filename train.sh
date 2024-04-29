# # A2C on Nebraska Maize 
# python train.py \
#     --model_type A2C \
#     --config_name nebraska_maize_base \
#     --save_dir nebraska_250 \
#     --train_steps 250000 \
#     --fig_dir ./nebraska_a2c_250 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO on Nebraska Maize 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_base \
#     --save_dir nebraska_250 \
#     --train_steps 250000 \
#     --fig_dir ./nebraska_ppo_250 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO with abundant water 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_abunwater \
#     --save_dir nebraska_abun \
#     --train_steps 100000 \
#     --fig_dir ./nebraska_abun_ppo \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO with scarce water 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_scarcewater \
#     --save_dir nebraska_scarce \
#     --train_steps 125000 \
#     --fig_dir ./nebraska_scarce_ppo \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO with more control over action, below 1000 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_control \
#     --save_dir nebraska_control \
#     --train_steps 150000 \
#     --fig_dir ./nebraska_control \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # controlled below 500 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_morecontrol \
#     --save_dir nebraska_morecontrol \
#     --train_steps 150000 \
#     --fig_dir ./nebraska_morecontrol \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # controlled below 400 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_control400 \
#     --save_dir l \
#     --train_steps 150000 \
#     --fig_dir ./nebraska_control400 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # controlled below 750 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_control750 \
#     --save_dir nebraska_control750 \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_control750 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate

# # PPO with weather forecast in state 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_forecast \
#     --save_dir nebraska_forecast \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_forecast \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO with 'nebraska_maize_scale2' config 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_scale2 \
#     --save_dir nebraska_scale2 \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_scale2 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # PPO with 'nebraska_maize_scale1' config 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_scale1 \
#     --save_dir nebraska_scale1 \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_scale1 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # nebraska_maize_forecast2 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_forecast2 \
#     --save_dir nebraska_forecast2 \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_forecast2 \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # 'nebraska_maize_1day' config 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_1day \
#     --save_dir nebraska_1day \
#     --train_steps 1400000 \
#     --fig_dir ./nebraska_1day \
#     --n_steps 126 \
#     --plot_train_curve \
#     --evaluate

# # 'nebraska_maize_3day' config 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_3day \
#     --save_dir nebraska_3day \
#     --train_steps 466667 \
#     --fig_dir ./nebraska_3day \
#     --n_steps 42 \
#     --plot_train_curve \
#     --evaluate 

# # 'nebraska_maize_5day' config
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_5day \
#     --save_dir nebraska_5day \
#     --train_steps 280000 \
#     --fig_dir ./nebraska_5day \
#     --n_steps 26 \
#     --plot_train_curve \
#     --evaluate 

# # 'nebraska_maize_14day' config 
# python train.py \
#     --model_type PPO \
#     --config_name nebraska_maize_14day \
#     --save_dir nebraska_14day \
#     --train_steps 100000 \
#     --fig_dir ./nebraska_14day \
#     --n_steps 9 \
#     --plot_train_curve \
#     --evaluate 

# # best config, TRPO 
# python train.py \
#     --model_type TRPO \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_trpo_best \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_trpo_best \
#     --plot_train_curve \
#     --evaluate 

# # best config, SAC 
# python train.py \
#     --model_type SAC \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_sac_best \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_sac_best \
#     --plot_train_curve \
#     --evaluate

# nebraska_maize_maxtemp 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_maxtemp \
    --save_dir nebraska_maxtemp \
    --train_steps 200000 \
    --fig_dir ./nebraska_maxtemp \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# TRPO with forecast 
python train.py \
    --model_type TRPO \
    --config_name nebraska_maize_forecast\
    --save_dir nebraska_trpo_fore \
    --train_steps 200000 \
    --fig_dir ./nebraska_trpo_fore \
    --plot_train_curve \
    --evaluate  

# nebraska_maize_maxtemp 
python train.py \
    --model_type TRPO \
    --config_name nebraska_maize_maxtemp \
    --save_dir nebraska_trpo_temp \
    --train_steps 200000 \
    --fig_dir ./nebraska_trpo_temp \
    --plot_train_curve \
    --evaluate  