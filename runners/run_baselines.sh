# python run_baselines.py --model celeba --method all --degradation deblurring --num_batches 10

# D-FLOW
# python run_baselines.py --model afhq --method d_flow \
#         --degradation super_resolution --blur_kernel_size 61 --blur_std 3.0 --sr_factor 4 \
#         --df_lmbda 0.01 --sigma_noise 5e-2 --df_max_iter 20 --alpha_dflow 0.1 --num_steps 6 \
#         --num_batches 100 --batch_size 1

# # OT-ODE
# for num_steps in 100; do
# for start_time in 0.1; do
# for otgamma in constant; do
#         # Print current configuration
#         echo "Running OT-ODE with num_steps=$num_steps, ot_start_time=$start_time, ot_gamma=$otgamma"
#         CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method ot_ode \
#                 --degradation inpainting_random --mask_ratio 0.9 \ 
                # --num_steps 100 --ot_start_time 0.1 --ot_gamma constant \
#                 --sigma_noise 1e-2 --num_batches 20
# done
# done
# done

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --model afhq --method ot_ode \
#         --degradation super_resolution --blur_kernel_size 61 --blur_std 3.0 --sr_factor 4 \
#         --ot_gamma constant --ot_start_time 0.3 \
#         --num_batches 1 --num_steps 200 --sigma_noise 5e-2


# ---

# # Flow Priors with custom params
# python run_baselines.py --model celeba --method flow_priors \
#         --degradation deblurring --blur_kernel_size 61 --blur_std 2.0 \
#         --fp_lmbda 1e4 --fp_K 3 \
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 20


# CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method flow_priors \
#         --degradation inpainting_random --mask_ratio 0.9\
#         --fp_lmbda 1e5 --fp_K 1 \
#         --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 20

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model afhq --method flow_priors \
#        --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 50 --sigma_noise 5e-2 \
#         --num_batches 1

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model afhq --method flow_priors \
#        --degradation deblurring_motion --blur_kernel_size 31 --blur_std 0.3 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 100 --sigma_noise 5e-2 \
#         --num_batches 2

# CUDA_VISIBLE_DEVICES=0 python run_baselines.py --model afhq --method flow_priors \
#        --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 \
#         --fp_lmbda 1e5 --fp_K 1 --fp_eta 1e-2\
#         --num_steps 50 --sigma_noise 10e-2 \
#         --num_batches 20

# # # PnP flow with custom params
# python run_baselines.py --model celeba --method pnp_flow \
#     --degradation deblurring --blur_kernel_size 61 --blur_std 2.0 \
#     --lookahead 1 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 \
#     --num_batches 20
# CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model celeba --method pnp_flow \
#     --degradation deblurring --blur_kernel_size 61 --blur_std 2.0 \
#     --lookahead 0 --alpha 0.05 --num_steps 100 --sigma_noise 5e-2 \
#     --num_batches 20
# CUDA_VISIBLE_DEVICES=1 python run_baselines.py --model afhq --method pnp_flow \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --lookahead 0 --alpha 0.05 --num_steps 300 --sigma_noise 1e-2 \
#         --num_batches 20
# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder tests100 --model afhq --method pnp_flow \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --lookahead 1.0 --alpha 0.05 --num_steps 100 --sigma_noise 1e-2 \
#         --num_batches 1

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder tests300 --model afhq --method pnp_flow \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --lookahead 1.0 --alpha 0.05 --num_steps 300 --sigma_noise 1e-2 \
#         --num_batches 1

# CUDA_VISIBLE_DEVICES=1 python run_baselines.py  --custom_name_folder ablation --model afhq --method pnp_flow \
#     --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 \
#     --lookahead 1 --alpha 0.05 --num_steps 50 --sigma_noise 10e-2 --gain 0.9 \
#     --num_batches 20

# CUDA_VISIBLE_DEVICES=2 python run_baselines.py --custom_name_folder ablation --model afhq --method pnp_flow \
#     --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#     --lookahead 0.0 --alpha 0.05 --num_steps 100 --sigma_noise 10e-2  --gain 0.9 \
#     --num_batches 20

# # DPS ODe
# CUDA_VISIBLE_DEVICES=3 python run_baselines.py --model afhq --method dps \
#         --degradation deblurring_motion --blur_kernel_size 61 --blur_std 0.5 --sr_factor 4 \
#         --num_steps 100 --sigma_noise 10e-2 --dps_eta 1e3 \
#         --num_batches 20
# python run_baselines.py --model afhq --method dps \
#         --degradation inpainting_random --mask_ratio 0.9 \
#         --num_steps 300 --sigma_noise 1e-2 --dps_eta 1e3 \
#         --num_batches 20

# ============================================================
# Lookahead 1
# ============================================================                                                                                                                  
# SUMMARY  (20 batches, deblurring_motion, celeba)                                                                                                                              
# ============================================================                                                                                                                  
# pnp_flow         PSNR=25.37±4.58dB  MSE=0.00708±0.01679  LPIPS=0.1427±0.1220       

# Looahead 0
# SUMMARY  (20 batches, deblurring_motion, celeba)
# ============================================================
# pnp_flow         PSNR=25.96±4.46dB  MSE=0.00631±0.01567  LPIPS=0.1710±0.0965

# OT-ODE
# PSNR=27.88±0.93dB  MSE=0.00167±0.00036  LPIPS=0.0690±0.0131

# Flow priors
# PSNR=26.04±1.09dB  MSE=0.00257±0.00066  LPIPS=0.0893±0.0111
