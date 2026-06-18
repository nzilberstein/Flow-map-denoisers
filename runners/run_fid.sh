degradation=deblurring_motion
# path=generated_samples_256/afhq_flow_priors_inpainting_box
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_256/fid_folder/ generated_samples_256/fid_folder_ground_truth/
# # python compute_metrics.py $path

# path=generated_samples_256/afhq_ot_ode_inpainting_box
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_256/fid_folder/ generated_samples_256/fid_folder_ground_truth/

# path=exps/generated_samples_256/afhq_flow_priors_${degradation}
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid exps/generated_samples_256/fid_folder/ exps/generated_samples_256/fid_folder_ground_truth/

# # path=generated_samples_256/afhq_d_flow_inpainting_box
# # python utils/prepare_fid_folders.py $path --skip_gt
# # python -m pytorch_fid generated_samples_256/fid_folder/ generated_samples_256/fid_folder_ground_truth/

path=generated_samples_flow_matching/afhq_pnp_flow_0.0_${degradation}_100_0.05
python utils/prepare_fid_folders.py $path --skip_gt
python -m pytorch_fid generated_samples_flow_matching/fid_folder/ exps/generated_samples_256/fid_folder_ground_truth/

# path=generated_samples_ablation_steps/afhq_pnp_flow_0.0_${degradation}_100_0.001
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_ablation_steps/fid_folder/ generated_samples_256/fid_folder_ground_truth/

# path=generated_samples_ablation_steps/afhq_pnp_flow_1.0_jpeg_100_0.01
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_ablation_steps/fid_folder/ generated_samples_256/fid_folder_ground_truth/



# path=generated_samples_ablation_steps/afhq_pnp_flow_0.0_jpeg_100_0.01
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_ablation_steps/fid_folder/ generated_samples_256/fid_folder_ground_truth/


degradation=deblurring_gaussian
# path=generated_samples_128/celeba_flow_priors_${degradation}
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/

# path=generated_samples_128/celeba_ot_ode_${degradation}
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/

# path=generated_samples_128/celeba_d_flow_${degradation}
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/

# path=generated_samples_128/celeba_dps_${degradation}
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/


# path=generated_samples_128/celeba_pnp_flow_1.0_${degradation}_100_0.05
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/

# path=generated_samples_128/celeba_pnp_flow_0.0_${degradation}_100_0.05
# python utils/prepare_fid_folders.py $path --skip_gt
# python -m pytorch_fid generated_samples_128/fid_folder/ generated_samples_128/fid_folder_ground_truth/