# Copyright (c) 2025, Weijian Luo. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train mean flow models using the techniques described in the
paper "Mean Flows for One-step Generative Modeling"."""

# NOTE: This file is a copy of the file of the EDM project.
#       It has been modified to fit the needs of the project.
#       The original file can be found at:
#       https://github.com/NVlabs/edm

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop_mf as training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

class CommaSeparatedList(click.ParamType):
    
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

# ----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--split',         help='Dataset split to use: train, test, or none (all)', metavar='train|test|none', type=click.Choice(['train', 'test', 'none']), default='none', show_default=True)
@click.option('--train-ratio',   help='Fraction of images for training split', metavar='FLOAT',     type=click.FloatRange(min=0, max=1), default=0.9, show_default=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['mf']), default='mf', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=500, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.10, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--log_weights',   help='Logarithmically space weights for each resolution', metavar='BOOL', type=bool, default=False, show_default=True)

# Loss Related
@click.option('--detach_tgt',          help='Detach u_tgt in loss or not', metavar='BOOL',              type=bool, default=True, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList(), default='fid50k_full', show_default=True)
@click.option('--detector_url',     help='detector_url',default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',  type=str)
@click.option('--data_stat',     help='Path to the dataset stats', metavar='ZIP|DIR',               type=str, default=None)

# Anisotropic loss
@click.option('--anisotropic',   help='Enable anisotropic loss', metavar='BOOL',                  type=bool, default=False, show_default=True)
@click.option('--aniso_min_box_size', help='Minimum box size for anisotropic loss', metavar='INT', type=click.IntRange(min=1), default=12, show_default=True)
@click.option('--aniso_max_box_size', help='Maximum box size for anisotropic loss', metavar='INT', type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--loss_type',     help='Loss type', metavar='meanflow|lsd|psd|flow_matching', type=click.Choice(['meanflow', 'lsd', 'psd', 'flow_matching']), default='meanflow', show_default=True)
@click.option('--data_proportion', help='Proportion of data points to apply anisotropic loss to', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.75, show_default=True)
@click.option('--noise_dist',    help='Noise distribution for sampling alpha in anisotropic loss', metavar='uniform|normal', type=click.Choice(['uniform', 'normal']), default='uniform', show_default=True)
@click.option('--lsd_warmup_kimg', help='Train diagonal-only for this many kimg before enabling LSD off-diagonal term', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--jvp_fp32',      help='Run LSD JVP in fp32 to avoid fp16 overflow (off-diagonal batch only)', metavar='BOOL', type=bool, default=False, show_default=True)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    export PYTORCH_ENABLE_FUNC_IMPL=1 && \
    export PYTORCH_DDP_NO_REBUILD_BUCKETS=1 && \
    export HF_HOME=/cpfs/user/weijian/cache && \
    export TORCH_NCCL_IB_TIMEOUT=23 && \
    export NCCL_TIMEOUT=3600 && \
    export SETUPTOOLS_USE_DISTUTILS=local && \
    /cpfs/user/weijian/thirdparty/miniconda3/envs/sidlsg/bin/torchrun --standalone \
    --nproc_per_node=8 \
    /newcpfs/user/deke/code/edm_meanflow_new/train_mf.py \
    --detach_tgt=1 \
    --outdir=/newcpfs/user/deke/logs/mf/MF03 \
    --data=/cpfs/user/weijian/data/cifar10-32x32.zip \
    --cond=0 --arch=ddpmpp --lr 10e-4 --batch 256
    """

    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    split = None if opts.split == 'none' else opts.split
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache, split=split, train_ratio=opts.train_ratio)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.95], eps=1e-8)
    c.metrics = opts.metrics
    c.detector_url=opts.detector_url
    c.data_stat=opts.data_stat

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[1,2,2,2])
        # c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2]) # CIFAR10 and MNIST
    else:
        raise NoteImplementedError('Architecture {opts.arch} not implemented.')

    # Preconditioning & loss function.
    assert opts.precond == 'mf'
    if opts.anisotropic:
        c.network_kwargs.class_name = 'training.networks_amf.AMFPrecond'
        # c.network_kwargs.class_name = 'training.networks_amf.ITv2AMFPrecond'
        if opts.loss_type == 'meanflow':
            c.loss_kwargs.class_name = 'training.loss_amf.AnisotropicMeanFlowLoss'
            c.loss_type = 'meanflow'
        elif opts.loss_type == 'lsd':
            c.loss_kwargs.class_name = 'training.loss_aflowmap.AnisotropicFlowMapLoss'
            c.loss_type = 'lsd'
        c.loss_kwargs.update(aniso_min_box_size=opts.aniso_min_box_size, aniso_max_box_size=opts.aniso_max_box_size)
        c.loss_kwargs.update(data_proportion=opts.data_proportion)
        c.loss_kwargs.update(noise_dist=opts.noise_dist)
        c.anisotropic = True
        c.aniso_min_box_size = opts.aniso_min_box_size
        c.aniso_max_box_size = opts.aniso_max_box_size
    else:
        if opts.loss_type == 'meanflow':
            c.network_kwargs.class_name = 'training.networks_mf.MFPrecond'
            c.loss_kwargs.class_name = 'training.loss_mf.MeanFlowLoss'
            c.loss_kwargs.update(detach_tgt=opts.detach_tgt)
        elif opts.loss_type == 'lsd':
            # Isotropic FlowMap LSD: uses AMFPrecond with scalar [B,1,1,1] alphas
            c.network_kwargs.class_name = 'training.networks_amf.AMFPrecond'
            # c.network_kwargs.class_name = 'training.networks_amf.ITv2AMFPrecond'
            c.network_kwargs.update(anisotropic_noise=False)
            c.network_kwargs.update(num_blocks=3, channel_mult=[1,2,3,4], attn_resolutions=[16,8])
            c.loss_kwargs.class_name = 'training.loss_iflowmap.IsotropicFlowMapLoss'
            c.loss_type = 'lsd'
            c.loss_kwargs.update(data_proportion=opts.data_proportion)
            c.loss_kwargs.update(noise_dist=opts.noise_dist)
            c.loss_kwargs.update(jvp_fp32=opts.jvp_fp32)
        elif opts.loss_type == 'psd':
            # Isotropic PSD: semigroup condition via three forward passes, no JVP
            c.network_kwargs.class_name = 'training.networks_amf.AMFPrecond'
            # c.network_kwargs.class_name = 'training.networks_amf.ITv2AMFPrecond'
            c.network_kwargs.update(anisotropic_noise=False)
            c.loss_kwargs.class_name = 'training.loss_iflowmap.IsotropicPSDLoss'
            c.loss_type = 'psd'
            c.loss_kwargs.update(data_proportion=opts.data_proportion)
            c.loss_kwargs.update(noise_dist=opts.noise_dist)
        elif opts.loss_type == 'flow_matching':
            # Plain flow matching: diagonal (r=t) velocity regression only, no
            # off-diagonal / flow-map self-distillation. Same network as 'lsd' so
            # the resulting model is a drop-in velocity field (sampled multi-step).
            c.network_kwargs.class_name = 'training.networks_amf.AMFPrecond'
            c.network_kwargs.update(anisotropic_noise=False)
            c.network_kwargs.update(num_blocks=3, channel_mult=[1,2,3,4], attn_resolutions=[16,8])
            c.loss_kwargs.class_name = 'training.loss_flowmatching.FlowMatchingLoss'
            c.loss_type = 'flow_matching'
            c.loss_kwargs.update(noise_dist=opts.noise_dist)
        c.anisotropic = False

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    if opts.log_weights:
        c.network_kwargs.use_weight = opts.log_weights
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.lsd_warmup_kimg = opts.lsd_warmup_kimg
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
