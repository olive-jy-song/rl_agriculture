python train.py \
    --model_type PPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_ppo_best_2048 \
    --train_steps 200000 \
    --fig_dir ./nebraska_ppo_best_2048 \
    --n_steps 2048 \
    --plot_train_curve \
    --evaluate 

python train.py \
    --model_type TRPO \
    --config_name nebraska_maize_best \
    --save_dir nebraska_trpo_best_2048 \
    --train_steps 200000 \
    --fig_dir ./nebraska_trpo_best_2048 \
    --n_steps 2048 \
    --plot_train_curve \
    --evaluate 

