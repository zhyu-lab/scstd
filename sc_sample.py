"""
Generate a large batch of RNA samples from a model and save them as a large
txt file after reversing the preprocessing steps.
"""

import argparse
import os

import einops
import numpy as np
import torch as th
import torch.distributed as dist
import pandas as pd
import math

from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
base_dir = './Testdata'
sample_dir = os.path.join(base_dir, 'Sample')
os.makedirs(sample_dir, exist_ok=True)

def main():
    th.manual_seed(0)
    np.random.seed(0)

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    log_dir = sample_dir
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")

    image_size = args.image_size
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_samples = []
    num_iterations = args.num_samples // args.batch_size

    for _ in range(num_iterations):
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 1, image_size, image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        sample = einops.rearrange(
            sample,
            'b c (h p1) (w p2) -> b c (h w p1 p2)',
            p1=args.patch_size, p2=args.patch_size
        )

        # Convert (batch_size, 1, image_size, image_size) to (batch_size, total_gene_dim)
        sample = sample.view(args.batch_size, -1)

        # Collect samples from all processes
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_samples) * args.batch_size} samples")

    # Combine all sampling results into one large numpy array
    arr = np.concatenate(all_samples, axis=0)
    arr = arr[: args.num_samples]

    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"sample_{arr.shape[0]}x{arr.shape[1]}.npy")
        logger.log(f"saving to {out_path}")
        np.save(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        data_dir=os.path.join(base_dir, 'raw.txt'),
        clip_denoised=True,
        num_samples=30000,
        batch_size=100,
        use_ddim=True,
        model_path=os.path.join(base_dir, 'Diffusion_models/ema_0.9999_100000.pt'),
        class_cond=False,
        decoder_path=os.path.join(base_dir, 'AE_models/best_rna_decoder.pth')
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
