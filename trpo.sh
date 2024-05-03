python train_with_checkpoint.py \
    --model_type TRPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_trpo_long \
    --train_steps 400000 \
    --fig_dir ./nebraska_best_trpo_long \
    --n_steps 2048 \
    --plot_checkpoints \
    --n_checkpoints 20   