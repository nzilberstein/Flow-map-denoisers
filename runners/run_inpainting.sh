# python run_baselines.py --model celeba --method all --degradation deblurring --num_batches 10

# D-FLOW
# CUDA_VISIBLE_DEVICES=6 python run_baselines.py --model afhq --method d_flow \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 80 \
#         --df_lmbda 0.01 --sigma_noise 1e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1


# CUDA_VISIBLE_DEVICES=6 python run_baselines.py --model celeba --method d_flow \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 60 \
#         --df_lmbda 0.01 --sigma_noise 1e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 20 --batch_size 5


# # OT-ODE
# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model afhq --method ot_ode \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 80 \
#         --num_steps 100 --ot_start_time 0.1 --ot_gamma constant \
#         --sigma_noise 1e-2 --num_batches 20

# 128
# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model celeba --method ot_ode \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 60 \
#         --num_steps 100 --ot_start_time 0.1 --ot_gamma gamma_t \
#         --sigma_noise 1e-2 --num_batches 20


# ---

# # Flow Priors with custom params
# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model afhq --method flow_priors \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 80 \
#         --fp_lmbda 1e5 --fp_K 1 \
#         --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 20

# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model celeba --method flow_priors \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 60 \
#         --fp_lmbda 1e5 --fp_K 1 \
#         --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 20

# # # PnP flow with custom params

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --custom_name_folder running_test --model afhq --method pnp_flow \
        --degradation inpainting_random --mask_ratio 0.9 \
        --lookahead 0 --alpha 0.05 --num_steps 300 --sigma_noise 1e-2 \
        --num_batches 1 --pnp_save_every 40

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --custom_name_folder running_test --model afhq --method pnp_flow \
        --degradation inpainting_random --mask_ratio 0.9 \
        --lookahead 1 --alpha 0.05 --num_steps 300 --sigma_noise 1e-2 \
        --num_batches 1 --pnp_save_every 40

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder tests100 --model afhq --method pnp_flow \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 1

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model celeba --method pnp_flow \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 60 \
#         --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 20


# # DPS ODe

# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model afhq --method dps \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 80 \
#         --num_steps 100 --sigma_noise 1e-2 --dps_eta 1e3 \
#         --num_batches 20
        # num_steps 300


# CUDA_VISIBLE_DEVICES=4 python run_baselines.py --model celeba --method dps \
#         --degradation inpainting_box --mask_ratio 0.9 --box_size 60 \
#         --num_steps 100 --sigma_noise 1e-2 --dps_eta 1e2 \
#         --num_batches 20 --num_steps 100
