# python train.py \
#     --model_type TRPO \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_trpo \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_trpo_best \
#     --n_steps 18 \
#     --plot_train_curve \
#     --evaluate 

# # best config, SAC 
# python train_with_checkpoint.py \
#     --model_type SAC \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_sac_check \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_sac_best_check \
#     --plot_checkpoints \
#     --n_checkpoints 10 

# DDPG with best 
# python train_with_checkpoint.py \
#     --model_type DDPG \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_ddpg_check \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_ddpg_best_check \
#     --plot_checkpoints \
#     --n_checkpoints 10 


# # A2C with best 
# python train_with_checkpoint.py \
#     --model_type A2C \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_a2c_check \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_a2c_best_check \
#     --n_steps 2048 \
#     --plot_checkpoints \
#     --n_checkpoints 10  

# # A2C with best 
# python train_with_checkpoint.py \
#     --model_type A2C \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_a2c_check_18 \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_a2c_best_check_18 \
#     --n_steps 18 \
#     --plot_checkpoints \
#     --n_checkpoints 10  

# python train_with_checkpoint.py \
#     --model_type TRPO \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_trpo_check \
#     --train_steps 200000 \
#     --fig_dir ./nebraska_best_trpo_check \
#     --n_steps 2048 \
#     --plot_checkpoints \
#     --n_checkpoints 10  

python train_with_checkpoint.py \
    --model_type PPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_ppo_check \
    --train_steps 200000 \
    --fig_dir ./nebraska_best_ppo_check \
    --n_steps 2048 \
    --plot_checkpoints \
    --n_checkpoints 10  


# python train_with_checkpoint.py \
#     --model_type PPO \
#     --config_name nebraska_maize_best \
#     --save_dir nebraska_best_ppo_60 \
#     --train_steps 600000 \
#     --fig_dir ./nebraska_best_ppo_60 \
#     --n_steps 2048 \
#     --plot_checkpoints \
#     --n_checkpoints 60   