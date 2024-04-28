# A2C on Nebraska Maize 
python train.py \
    --model_type A2C \
    --config_name nebraska_maize_base \
    --save_dir nebraska_250 \
    --train_steps 250000 \
    --fig_dir ./nebraska_a2c_250 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

# PPO on Nebraska Maize 
python train.py \
    --model_type PPO \
    --config_name nebraska_maize_base \
    --save_dir nebraska_250 \
    --train_steps 250000 \
    --fig_dir ./nebraska_ppo_250 \
    --n_steps 18 \
    --plot_train_curve \
    --evaluate 

