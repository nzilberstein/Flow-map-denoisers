degradation=deblurring_motion
# path=generated_samples_flow_matching/afhq_flow_priors_${degradation}
# python compute_metrics.py $path

# path=generated_samples_flow_matching/afhq_ot_ode_${degradation}
# python compute_metrics.py $path

# path=generated_samples_flow_matching/afhq_dps_${degradation}
# python compute_metrics.py $path

# path=generated_samples_256/afhq_d_flow_${degradation}
# python compute_metrics.py $path

# path=generated_samples_256/afhq_dps_jpeg
# python compute_metrics.py $path

path=generated_samples_flow_matching/afhq_pnp_flow_0.0_${degradation}_100_0.05
python compute_metrics.py $path
# path=generated_samples_ablation_steps/afhq_pnp_flow_0.0_${degradation}_1000_0.01
# python compute_metrics.py $path

# path=generated_samples_256/afhq_pnp_flow_0.0_deblurring_motion_100_0.05
# python compute_metrics.py $path

# degradation=deblurring_motion
# path=generated_samples_128/celeba_flow_priors_${degradation}
# python compute_metrics.py $path

# path=generated_samples_128/celeba_ot_ode_${degradation}
# python compute_metrics.py $path

# path=generated_samples_128/celeba_d_flow_${degradation}
# python compute_metrics.py $path

# path=generated_samples_128/celeba_dps_${degradation}
# python compute_metrics.py $path

# path=generated_samples_128/celeba_pnp_flow_1.0_${degradation}_100_0.05
# python compute_metrics.py $path

# path=generated_samples_128/celeba_pnp_flow_0.0_${degradation}_100_0.05_test
# python compute_metrics.py $path