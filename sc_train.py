"""
Train a diffusion model on images.
"""
import os
import argparse
from diffusion import dist_util, logger
from diffusion.scRNA_dataset import load_data
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from diffusion.train_util import TrainLoop
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

base_dir = './Testdata'
model_dir = os.path.join(base_dir, 'Diffusion_models')
os.makedirs(model_dir, exist_ok=True)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = model_dir

    logger.configure(dir=log_dir)

    logger.log("creating data loader...")
    data, image_size = load_data(
        embeddings_file=args.embeddings_file,
        batch_size=args.batch_size,
        shuffle=False,
        deterministic=False,
        patch_size=args.patch_size,
    )
    args.image_size = image_size

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def create_argparser():
    defaults = dict(
        embeddings_file=os.path.join(base_dir, 'best_embeddings.npy'),
        schedule_sampler="uniform",
        lr=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=100000, #train steps
        batch_size=10,
        microbatch=1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
