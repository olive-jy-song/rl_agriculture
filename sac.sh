# best config, SAC 
python train_with_checkpoint.py \
    --model_type SAC \
    --config_name nebraska_maize_best \
    --save_dir nebraska_best_sac_check \
    --train_steps 200000 \
    --fig_dir ./nebraska_sac_best_check \
    --plot_checkpoints \
    --n_checkpoints 10 