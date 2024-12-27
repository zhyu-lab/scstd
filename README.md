# scSTD

A Swin Transformer based Diffusion model for imputation and denoising of scRNA-seq data

## Requirements

* Python 3.9+.

# Installation

## Clone repository

First, download scSTD from github and change to the directory:

```bash
git https://github.com/zhyu-lab/scstd
cd scstd
```

## Create conda environment (optional)

Create a new environment named "scstd":

```bash
conda create --name scstd python=3.9
```

Then activate it:

```bash
conda activate scstd
```

## Install requirements

```bash
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
python -m pip install -r requirements.txt
```

# Usage

## Dataset

The data used to train the model is in TXT format, where each row represents a cell and each column corresponds to a gene.

## Train the scSTD model

### Step 1: Train the Autoencoder

The "autoencoder.py" Python script is used to train the autoencoder model. Set the file_path to the location of the raw scRNA-seq data.

### Step 2: Train the Diffusion model

The "sc_train.py" Python script is used to train the diffusion model. First, set the model_dir to the directory where you want to save the diffusion model checkpoints. Then, set the embeddings_file to the path of the embeddings generated through the encoder, and specify the lr_anneal_steps (the number of training steps).

Example:

```bash
python sc_train.py --embeddings_file ./Testdata/best_embeddings.npy --lr_anneal_steps 100000 --batch_size 10 --lr 0.001 --save_interval 10000 --use_fp16 False
```

## Sample and Imputation

### Sample

The "sc_sample.py" Python script is used to generate N latent codes by sampling from the diffusion model. Set the model_path to the path of the diffusion model checkpoints, and specify num_samples as the number of latent codes (N) you want to generate.

```bash
python sc_sample.py --model_path .Testdata/Diffusion_models/ema_0.9999_100000.pt --num_samples 30000
```

### Imputation

Finally, run the "imputation.py" script to obtain the reconstructed data. Set the 'original_txt_path' to the location of the raw scRNA-seq data and retrieve its dimensions. Then, set the 'decoder_path' to the path of the decoder weights.

# Contact

If you have any questions, please contact 12023131977@stu.nxu.edu.cn.