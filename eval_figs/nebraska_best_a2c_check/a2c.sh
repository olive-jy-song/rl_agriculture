python train_with_checkpoint.py \
    --model_type A2C \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_a2c_check \
    --train_steps 200000 \
    --fig_dir ./nebraska_a2c_best_check \
    --n_steps 2048 \
    --plot_checkpoints \
    --n_checkpoints 10  