# for num_steps in 50 100 200 500; do
for num_steps in 1500; do
  for LA in 0 1.0; do
# for LA in 0 0.01 0.05 0.1 0.2 0.3 0.5 1.0; do
  # for num_steps in 10 20 50 100 300 500 700; do
  # CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA --degradation phase_retrieval  --num_steps $num_steps --model afhq --sr_factor 2 --blur_kernel_size 31 --blur_std 1.0 --sigma_noise 5e-2 --alpha 0.1
  # CUDA_VISIBLE_DEVICES=7 python run_baselines_poisson.py --peak 20 --custom_name_folder poisson --model afhq --method pnp_flow --lookahead $LA --degradation deblurring_gaussian  --num_steps $num_steps --model afhq --blur_kernel_size 61 --blur_std 3.0 --alpha 0.9

  # CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA --degradation deblurring_gaussian  --num_steps $num_steps --model afhq --sr_factor 2 --blur_kernel_size 61 --blur_std 3.0 --sigma_noise 5e-2 --alpha 0.05
  # CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA --degradation super_resolution  --num_steps $num_steps --model afhq --sr_factor 4 --blur_kernel_size 31 --blur_std 3.0 --sigma_noise 5e-2 --alpha 0.05
  CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA --degradation phase_retrieval  --num_steps $num_steps --model afhq --sigma_noise 1e-2 --alpha 0.05
    # CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA \
  # --degradation deblurring_motion  --num_steps $num_steps --model afhq  \
  # --blur_kernel_size 61 --blur_std 1.0 --sigma_noise 5e-2 \
  # --gain 1.0 --pnp_optimizer gd --alpha 0.1 --num_batches 20 \

  # python run_baselines.py --method pnp_flow --lookahead $LA --degradation deblurring_motion --blur_kernel_size 61 --blur_std 1.0 --num_steps $num_steps --model afhq --sigma_noise 5e-2 
  # CUDA_VISIBLE_DEVICES=7 python run_baselines.py --method pnp_flow --lookahead $LA --degradation deblurring_motion --jpeg_qf 10 --num_steps $num_steps --model afhq --box_size 80 --sigma_noise 1e-2 --alpha 0.05
done
done


  # PSNR  : 27.071 ± 1.114 dB
  # LPIPS : 0.1154 ± 0.0148
  # DISTS : 0.1160 ± 0.0084

  #   PSNR  : 28.352 ± 1.164 dB
  # LPIPS : 0.2554 ± 0.0265
  # DISTS : 0.1856 ± 0.0111

  #   PSNR  : 28.232 ± 1.097 dB
  # LPIPS : 0.2449 ± 0.0262
  # DISTS : 0.1847 ± 0.0101

# Ot ODE
  #   PSNR  : 26.826 ± 1.050 dB
  # LPIPS : 0.0787 ± 0.0115
  # DISTS : 0.0911 ± 0.0074