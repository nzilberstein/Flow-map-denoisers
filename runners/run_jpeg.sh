# python run_baselines.py --model celeba --method all --degradation deblurring --num_batches 10

# D-FLOW
# CUDA_VISIBLE_DEVICES=6 python run_baselines.py --custom_name_folder running_test --model afhq --method d_flow \
#         --degradation deblurring_gaussian  --blur_kernel_size 61 --blur_std 3.0 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 5e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1

# python run_baselines.py --model afhq --method d_flow \
#         --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 10e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1

# python run_baselines.py --model celeba --method d_flow \
#         --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 5e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 20 --batch_size 5


# CUDA_VISIBLE_DEVICES=0 python run_baselines.py  --custom_name_folder running_test  --model afhq --method ot_ode \
#         --degradation deblurring_gaussian --blur_kernel_size 61 --blur_std 3.0 --sr_factor 4 \
#         --ot_gamma constant --ot_start_time 0.3 \
#         --num_batches 20 --num_steps 100 --sigma_noise 5e-2 --batch_size 1

# 128 
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model celeba --method ot_ode \
#         --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#         --ot_gamma constant --ot_start_time 0.3 \
#         --num_batches 10 --num_steps 100 --sigma_noise 5e-2

# ---

# # Flow Priors with custom params

# CUDA_VISIBLE_DEVICES=0 python run_baselines.py --model afhq --method flow_priors \
#        --degradation deblurring_gaussian --blur_kernel_size 61 --blur_std 3.0 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 20 --batch_size 1

# 128
# CUDA_VISIBLE_DEVICES=0 python run_baselines.py --model celeba --method flow_priors \
#        --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 0.5 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 20


# # DPS ODe
# CUDA_VISIBLE_DEVICES=3 python run_baselines.py  --model afhq --method dps \
#         --degradation colorization \
#         --num_steps 100 --sigma_noise 1e-3 --dps_eta 0.1e3 \
#         --num_batches 20 --batch_size 5

# 128
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py  --model afhq --method dps \
#         --degradation colorization \
#         --num_steps 100 --sigma_noise 1e-2 --dps_eta 1e3 \
#         --num_batches 1

# # # PnP flow with custom params

# 256
# CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method pnp_flow \
#     --degradation jpeg --jpeg_qf 10 \
#     --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-2 --gain 1.0 \
#     --num_batches 20 --batch_size 5

# CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method pnp_flow \
#     --degradation jpeg --jpeg_qf 10 \
#     --lookahead 0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-2 --gain 1.0 \
#     --num_batches 20 --batch_size 5

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method pnp_flow \
    --degradation phase_retrieval \
    --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-3 --gain 1.0 \
    --num_batches 20 --batch_size 5

CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method pnp_flow \
    --degradation phase_retrieval \
    --lookahead 0.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-3 --gain 1.0 \
    --num_batches 20 --batch_size 5



# 128
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model celeba --method pnp_flow \
#     --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 3.0 --sr_factor 2 \
#     --lookahead 1.0 --alpha 0.05 --num_steps 30 --sigma_noise 5e-2 \
#     --num_batches 20

# ot_ode           PSNR=24.59±0.77dB  MSE=0.00353±0.00062  LPIPS=0.1118±0.0143

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model celeba --method pnp_flow \
#     --degradation deblurring_gaussian --blur_kernel_size 31 --blur_std 3.0 --sr_factor 4 \
#     --lookahead 0.0 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2  --gain 1.0 \
#     --num_batches 1