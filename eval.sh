# python evaluate.py \
#     --model_path .saved_model/nebraska_250/a2c_682.185\
#     --model_type A2C \
#     --config_name nebraska_maize_base \
#     --fig_dir ./nebraska_250_a2c \
#     --generate_plots 

# python evaluate.py \
#     --model_path .saved_model/nebraska_250/ppo_666.043\
#     --model_type PPO \
#     --config_name nebraska_maize_base \
#     --fig_dir ./nebraska_250_ppo \
#     --generate_plots  

python evaluate.py \
    --model_path .saved_model/nebraska_scarce/ppo_554.971\
    --model_type PPO \
    --config_name nebraska_maize_scarcewater \
    --fig_dir ./nebraska_scarce \
    --generate_plots  