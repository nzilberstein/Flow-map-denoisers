# python run_baselines.py --model celeba --method all --degradation deblurring --num_batches 10

# D-FLOW
# python run_baselines.py --model afhq --method d_flow \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --df_lmbda 0.01 --sigma_noise 1e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1

# python run_baselines.py --model afhq --method d_flow \
#         --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 10e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1

# CUDA_VISIBLE_DEVICES=3 python run_baselines.py --model celeba --method d_flow \
#         --degradation super_resolution --blur_kernel_size 31 --blur_std 0.5 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 5e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 20 --batch_size 5


# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model afhq --method ot_ode \
#         --degradation super_resolution --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --ot_gamma constant --ot_start_time 0.3 \
#         --num_batches 20 --num_steps 100 --sigma_noise 5e-2

# 128 
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model celeba --method ot_ode \
#         --degradation super_resolution --blur_kernel_size 31 --blur_std 0.5 --sr_factor 4 \
#         --ot_gamma constant --ot_start_time 0.3 \
#         --num_batches 20 --num_steps 100 --sigma_noise 5e-2

# ---

# # Flow Priors with custom params

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model afhq --method flow_priors \
#        --degradation super_resolution --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 20

# 128
# CUDA_VISIBLE_DEVICES=0 python run_baselines.py --model celeba --method flow_priors \
#        --degradation super_resolution --blur_kernel_size 31 --blur_std 0.5 --sr_factor 4 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 20


# # DPS ODe
# CUDA_VISIBLE_DEVICES=3 python run_baselines.py --model afhq --method dps \
#         --degradation super_resolution --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --num_steps 100 --sigma_noise 5e-2 --dps_eta 1e3 \
#         --num_batches 20

# 128
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py  --model celeba --method dps \
#         --degradation super_resolution --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#         --num_steps 100 --sigma_noise 5e-2 --dps_eta 1e3 \
#         --num_batches 20

# dps              PSNR=23.97±4.14dB  MSE=0.00679±0.00857  LPIPS=0.1220±0.0519
# # # PnP flow with custom params

# 256
CUDA_VISIBLE_DEVICES=1 python run_baselines.py --custom_name_folder running_test --model afhq --method pnp_flow \
    --degradation super_resolution --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
    --lookahead 1 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 --gain 1.0 \
    --num_batches 1 --pnp_save_every 20

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --custom_name_folder running_test --model afhq --method pnp_flow \
    --degradation super_resolution --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
    --lookahead 0 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 --gain 1.0 \
    --num_batches 1 --pnp_save_every 20

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model celeba --method pnp_flow \
#     --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#     --lookahead 0.0 --alpha 0.05 --num_steps 100 --sigma_noise 10e-2  --gain 0.9 \
#     --num_batches 1


# 128
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py  --custom_name_folder ablation_07 --model celeba --method pnp_flow \
#     --degradation super_resolution --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#     --lookahead 1 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 --gain 1.0 \
#     --num_batches 1

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model celeba --method pnp_flow \
#     --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#     --lookahead 0.0 --alpha 0.05 --num_steps 50 --sigma_noise 10e-2  --gain 0.75 \
#     --num_batches 1