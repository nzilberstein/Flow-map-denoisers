# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

# NOTE: This file is a copy of the file of the EDM project.
#       It has been modified to fit the needs of the project.
#       The original file can be found at:
#       https://github.com/NVlabs/edm

import os
import gc
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import PIL.Image
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from torch.utils.tensorboard import SummaryWriter

import torch.nn.parallel.distributed as dist_module

from metrics import metric_main

from torch.optim.lr_scheduler import CosineAnnealingLR

from training.mask_generation import generate_random_box_mask, generate_box_mask

# ----------------------------------------------------------------------------
import torch.distributed as tdist  # use PyTorch original

#----------------------------------------------------------------------------
def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = 2#np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = 1#np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------
# Helper methods

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def calculate_metric(metric, G, dataset_kwargs, num_gpus, rank, local_rank, device, data_stat, detector_url):
    return metric_main.calc_metric(metric=metric, G=G,
        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, local_rank=local_rank, device=device, data_stat=data_stat, detector_url=detector_url)


def manual_grad_sync(model):
    for param in model.parameters():
        if param.grad is not None:
            tdist.all_reduce(param.grad, op=tdist.ReduceOp.SUM)  # 梯度累积的时候已经除了GPU数量，直接sum

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    metrics             = None,
    detector_url        = None,
    data_stat           = None,
    sigma_min           = 0.002,     # Minimum sigma for Mean Flow loss.
    sigma_max           = 1.0,      # Maximum sigma for Mean Flow loss.
    anisotropic         = False,    # Enable anisotropic mean flow mode (inpainting).
    aniso_min_box_size  = 12,       # Min box size for anisotropic mask generation.
    aniso_max_box_size  = 18,       # Max box size for anisotropic mask generation.
    aniso_num_steps     = 1,        # Number of sampling steps for anisotropic inference.
    aniso_val_batches   = 4,        # Number of validation batches for anisotropic val loss.
    loss_type           = 'meanflow', # 'meanflow' or 'lsd', only relevant when anisotropic=True
    lsd_warmup_kimg     = 0,        # Train diagonal-only for this many kimg before enabling LSD off-diagonal term.
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # Efficient/flash attention do not support forward-mode AD (torch.func.jvp).
    # Force the math backend so JVP works through transformer attention layers.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # Load modules associated to the loss.
    dist.print0(f'Loading loss {loss_type}...')
    if anisotropic:
        if loss_type == 'meanflow':
            from easy_meanflow.anisotropic_flow_maps.loss_amf import anisotropic_sample, compute_alpha_t
        elif loss_type == 'lsd':
            from easy_meanflow.anisotropic_flow_maps.loss_aflowmap import anisotropic_sample
    else:
        if loss_type in ('lsd', 'psd'):
            from training.loss_iflowmap import isotropic_sample
        elif loss_type == 'flow_matching':
            from training.loss_flowmatching import flow_matching_sample

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    # Broadcast parameters and buffers from rank 0 so all ranks start identically.
    # Needed for modules that use random register_buffer (e.g. FourierFeatures in ITv2).
    for tensor in list(net.parameters()) + list(net.buffers()):
        tdist.broadcast(tensor.data, src=0)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            if anisotropic:
                alpha_t = torch.zeros([batch_gpu, 1, net.img_resolution, net.img_resolution], device=device)
                alpha_r = torch.zeros([batch_gpu, 1, net.img_resolution, net.img_resolution], device=device)
                misc.print_module_summary(net, [images, alpha_r, alpha_t], max_nesting=2)
            else:
                if loss_type in ('lsd', 'psd', 'flow_matching'):
                    # Isotropic lsd/psd/flow_matching: scalar alpha inputs [B, 1, 1, 1]
                    alpha_r = torch.zeros([batch_gpu, 1, 1, 1], device=device)
                    alpha_t = torch.zeros([batch_gpu, 1, 1, 1], device=device)
                    misc.print_module_summary(net, [images, alpha_r, alpha_t], max_nesting=2)
                else:
                    sigma = torch.ones([batch_gpu], device=device)
                    labels = torch.zeros([batch_gpu, net.label_dim], device=device)
                    misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    base_data_proportion = getattr(loss_fn, 'data_proportion', 1.0)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    scheduler = CosineAnnealingLR(optimizer, T_max=total_kimg*1000//batch_size, eta_min=1e-5)
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False, static_graph=True, find_unused_parameters=False,  gradient_as_bucket_view=True)   # 关闭缓冲区同步，使用静态图，为了计算jvp,
    
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Initialize TensorBoard (rank 0 only)
    tb_writer = None
    if dist.get_rank() == 0:
        tb_dir = os.path.join(run_dir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir)

    dist.print0('Exporting sample images...')
    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_masks = None
    grid_observed = None

    if dist.get_rank() == 0:
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)

        print('Exporting sample images...')
        save_image_grid(img=images, fname=os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

        if anisotropic:
            # For anisotropic mode: set up fixed masks and observed images for evaluation
            n_grid = labels.shape[0]
            res = ema.img_resolution
            grid_observed = (torch.from_numpy(images).to(device).to(torch.float32) / 127.5 - 1)
            grid_masks = generate_random_box_mask(
                n_grid, 1, res, res,
                min_box_size=aniso_min_box_size,
                max_box_size=aniso_max_box_size,
                device=device,
            )
            # Split for batched inference
            grid_observed_split = grid_observed.split(batch_gpu)
            grid_masks_split = grid_masks.split(batch_gpu)
            fakes_init = torch.cat([
                anisotropic_sample(ema, m, x_observed=obs, num_steps=aniso_num_steps, device=device).cpu()
                for obs, m in zip(grid_observed_split, grid_masks_split)
            ]).numpy()
            save_image_grid(img=fakes_init, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
            # Also save the masked observed images for reference
            masked_vis = (grid_observed * grid_masks).cpu().numpy()
            save_image_grid(img=masked_vis, fname=os.path.join(run_dir, 'masked_observed.png'), drange=[-1,1], grid_size=grid_size)
            del fakes_init, masked_vis
        else:
            if loss_type in ('lsd', 'psd'):
                # Isotropic lsd/psd: unconditional generation from pure noise
                n_grid = labels.shape[0]
                fakes_init = isotropic_sample(ema, n_grid, ema.img_channels, ema.img_resolution, num_steps=1, device=device).cpu().numpy()
                save_image_grid(img=fakes_init, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
                del fakes_init
            elif loss_type == 'flow_matching':
                # Plain flow matching: multi-step Euler integration of the velocity field
                n_grid = labels.shape[0]
                fakes_init = flow_matching_sample(ema, n_grid, ema.img_channels, ema.img_resolution, num_steps=50, device=device).cpu().numpy()
                save_image_grid(img=fakes_init, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
                del fakes_init
            else:
                grid_z = torch.randn([labels.shape[0], ema.img_channels, ema.img_resolution, ema.img_resolution], device=device)
                grid_z = grid_z.split(batch_gpu)
                grid_c = torch.from_numpy(labels).to(device)
                grid_c = grid_c.split(batch_gpu)
                images = torch.cat([(z-ema(z, sigma_min*torch.ones([z.shape[0], 1, 1, 1], device=z.device), c, h=sigma_max*torch.ones([z.shape[0], 1, 1, 1], device=z.device), augment_labels=torch.zeros(z.shape[0], 9).to(z.device))).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(img=images, fname=os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
                del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None

    while True:

        # Update data proportion based on warmup schedule (applies to lsd and psd).
        if loss_type in ('lsd', 'psd') and lsd_warmup_kimg > 0 and hasattr(loss_fn, 'data_proportion'):
            loss_fn.data_proportion = 1.0 if cur_nimg < lsd_warmup_kimg * 1000 else base_data_proportion

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                if anisotropic:
                    loss = loss_fn(net=ddp, images=images)
                else:
                    if loss_type in ('lsd', 'psd', 'flow_matching'):
                        loss = loss_fn(net=ddp, images=images)
                    else:
                        labels = None if labels.numel()==0 else labels.to(device)
                        loss = loss_fn(net=ddp, images=images, labels=labels, augment_pipe=augment_pipe)
                loss = loss.mul(loss_scaling / batch_gpu_total)
                loss.backward()
                training_stats.report('Loss/loss', loss.item())

        # NOTE(Weijian): we do not use ddp to compute the loss，so we need to sync the gradient manually
        manual_grad_sync(ddp.module)

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Evaluate FID
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0 or cur_tick in [2,10,20,30,40,50,60,70,80,90,100]):
            dist.print0('Exporting sample images...')
            if dist.get_rank() == 0:
                if anisotropic:
                    grid_observed_split = grid_observed.split(batch_gpu)
                    grid_masks_split = grid_masks.split(batch_gpu)
                    images = torch.cat([
                        anisotropic_sample(ema, m, x_observed=obs, num_steps=aniso_num_steps, device=device).cpu()
                        for obs, m in zip(grid_observed_split, grid_masks_split)
                    ]).numpy()
                else:
                    if loss_type in ('lsd', 'psd'):
                        n_grid = grid_size[0] * grid_size[1]
                        images = isotropic_sample(ema, n_grid, ema.img_channels, ema.img_resolution, num_steps=1, device=device).cpu().numpy()
                    else:
                        images = torch.cat([(z-ema(z, sigma_max*torch.ones([z.shape[0], 1, 1, 1], device=z.device), c, h=(sigma_max-sigma_min)*torch.ones([z.shape[0], 1, 1, 1], device=z.device), augment_labels=torch.zeros(z.shape[0], 9).to(z.device))).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(img=images, fname=os.path.join(run_dir, f'fakes_{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

                del images; gc.collect(); torch.cuda.empty_cache()

            if cur_tick > 0:
                if anisotropic:
                    # Anisotropic validation: compute val loss over test batches
                    dist.print0('Evaluating anisotropic validation loss...')
                    if loss_type == 'meanflow':
                        from easy_meanflow.anisotropic_flow_maps.loss_amf import AnisotropicMeanFlowLoss
                        # TODO: aniso_min_box_size=aniso_min_box_size, aniso_max_box_size=aniso_max_box_size
                        val_loss_fn = AnisotropicMeanFlowLoss(aniso_min_box_size=aniso_min_box_size, aniso_max_box_size=aniso_max_box_size, use_jacobian_correction=False)
                    elif loss_type == 'lsd':
                        from easy_meanflow.anisotropic_flow_maps.loss_aflowmap import AnisotropicFlowMapLoss
                        val_loss_fn = AnisotropicFlowMapLoss(aniso_min_box_size=aniso_min_box_size, aniso_max_box_size=aniso_max_box_size)
                    val_loss_fn.min_box_size = aniso_min_box_size
                    val_loss_fn.max_box_size = aniso_max_box_size
                    val_losses = []
                    ema.eval()
                    with torch.no_grad():
                        for _ in range(aniso_val_batches):
                            val_images, _ = next(dataset_iterator)
                            val_images = val_images.to(device).to(torch.float32) / 127.5 - 1
                            val_loss = val_loss_fn(net=ema, images=val_images)
                            val_losses.append(val_loss.item())
                    avg_val_loss = np.mean(val_losses)
                    training_stats.report('Loss/val_loss', avg_val_loss)
                    dist.print0(f'Anisotropic val loss: {avg_val_loss:.4f}')
                    if tb_writer is not None and dist.get_rank() == 0:
                        tb_writer.add_scalar('Loss/val_loss', avg_val_loss, cur_nimg)

                if not anisotropic:
                    if loss_type in ('lsd', 'psd'):
                        dist.print0(f'Evaluating isotropic {loss_type} validation loss...')
                        val_dp = 1.0 if (lsd_warmup_kimg > 0 and cur_nimg < lsd_warmup_kimg * 1000) else base_data_proportion
                        if loss_type == 'lsd':
                            from training.loss_iflowmap import IsotropicFlowMapLoss
                            val_loss_fn = IsotropicFlowMapLoss(data_proportion=val_dp, jvp_fp32=getattr(loss_fn, 'jvp_fp32', False))
                        else:
                            from training.loss_iflowmap import IsotropicPSDLoss
                            val_loss_fn = IsotropicPSDLoss(data_proportion=val_dp)
                        val_losses = []
                        ema.eval()
                        with torch.no_grad():
                            for _ in range(aniso_val_batches):
                                val_images, _ = next(dataset_iterator)
                                val_images = val_images.to(device).to(torch.float32) / 127.5 - 1
                                val_loss = val_loss_fn(net=ema, images=val_images)
                                val_losses.append(val_loss.item())
                        avg_val_loss = np.mean(val_losses)
                        training_stats.report('Loss/val_loss', avg_val_loss)
                        dist.print0(f'Isotropic val loss: {avg_val_loss:.4f}')
                        if tb_writer is not None and dist.get_rank() == 0:
                            tb_writer.add_scalar('Loss/val_loss', avg_val_loss, cur_nimg)
                    dist.print0('Evaluating metrics...')
                    if metrics is not None:
                        for metric in metrics:
                            result_dict = calculate_metric(metric=metric, G=ema,
                                dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), local_rank=dist.get_local_rank(), device=device,data_stat=data_stat,detector_url=detector_url)
                            if dist.get_rank() == 0:
                                print(result_dict.results)
                                metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{cur_nimg//1000:06d}.png')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

            gc.collect(); torch.cuda.empty_cache()

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:

            # Write to TensorBoard
            stats_dict = training_stats.default_collector.as_dict()
            for key, value in stats_dict.items():
                if isinstance(value['mean'], (int, float)):
                    tb_writer.add_scalar(key, value['mean'], cur_nimg)
            
            # Existing JSONL logging
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(stats_dict, timestamp=time.time())) + '\n')
            stats_jsonl.flush()

        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if dist.get_rank() == 0 and tb_writer is not None:
        tb_writer.close()
    dist.print0()
    
    dist.print0('Exiting...')

# ----------------------------------------------------------------------------
