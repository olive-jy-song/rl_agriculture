# this is an example of how we can evaluate the agent 

python evaluate.py \
    --model_path .saved_model/nebraska_1day/ppo_434.595\
    --model_type PPO \
    --config_name nebraska_maize_1day \
    --fig_dir ./nebraska_1day \
    --generate_plots 