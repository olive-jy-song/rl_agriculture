python train_with_checkpoint.py \
    --model_type PPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_ppo_check \
    --train_steps 200000 \
    --fig_dir ./nebraska_best_ppo_check \
    --n_steps 2048 \
    --plot_checkpoints \
    --n_checkpoints 10  
