python train_with_checkpoint.py \
    --model_type DDPG \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_ddpg_check \
    --train_steps 200000 \
    --fig_dir ./nebraska_ddpg_best_check \
    --plot_checkpoints \
    --n_checkpoints 10 