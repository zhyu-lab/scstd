import einops
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

def load_data(
        *, embeddings_file, batch_size, shuffle=False, deterministic=False, patch_size):
    """
    Load embedding data and return a DataLoader for training.

    Args:
        embeddings_file (str): Path to the embedding file (.pt).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        deterministic (bool): Whether to return data in deterministic order.

    Returns:
        tuple: Infinite data generator and image size.
    """
    # Create RNA dataset
    dataset = ScRNADataset(embeddings_file=embeddings_file, patch_size=patch_size)

    # Extract image size
    if len(dataset) > 0:
        _, side_length, _ = dataset[0][0].shape
    else:
        raise ValueError("Dataset is empty.")

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not deterministic,
        num_workers=4,
        drop_last=True
    )

    def infinite_generator():
        while True:
            for batch in loader:
                yield batch

    return infinite_generator(), side_length


class ScRNADataset(Dataset):
    def __init__(self, embeddings_file, patch_size):
        """
        Args:
            embeddings_file (str): Path to the saved embedding file (.npy).
        """
        self.data = self.preprocess_embeddings(embeddings_file, patch_size=patch_size)

    def preprocess_embeddings(self, embeddings_file, patch_size):
        """
        Load and preprocess embeddings: reshape as images and normalize.
        """
        data = np.load(embeddings_file)
        print(f"embedding's shape: {data.shape}")
        cells, features = data.shape

        # Reshape the embeddings as images
        side_length = int(np.sqrt(features))
        if side_length ** 2 != features:
            raise ValueError(f"Embedding size {features} cannot be reshaped into a square image.")

        reshaped_data = data.reshape(cells, side_length, side_length)
        num_patches_side = side_length // patch_size
        if side_length % patch_size != 0:
            raise ValueError(f"side_length {side_length} cannot be divided by patch_size {patch_size}.")

        reshaped_data = reshaped_data.reshape(cells, -1, patch_size * patch_size)
        print(f"Reshaped data shape: {reshaped_data.shape}")

        rearranged_data = einops.rearrange(
                    reshaped_data,
                  'b (h w) (p1 p2) -> b (h p1) (w p2)',
                    h = num_patches_side, p1=patch_size, p2=patch_size
              )
        print(f"Rearranged data shape: {rearranged_data.shape}")

        return rearranged_data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Add channel dimension (1, side_length, side_length)
        sample = np.expand_dims(sample, axis=0)

        return sample, {}
