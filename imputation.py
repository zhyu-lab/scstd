import torch
import numpy as np
import pandas as pd
from autoencoder import NetBlock, get_encoder_decoder
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

base_dir = './Testdata'
result_dir = os.path.join(base_dir, 'Result')
os.makedirs(result_dir, exist_ok=True)
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def reconstruct_data_from_embeddings(decoder, embeddings_to_reconstruct, device):
    """
    Reconstruct data from embedding using the given decoder model.
    """
    decoder.eval()
    with torch.no_grad():
        embeddings_to_reconstruct = embeddings_to_reconstruct.to(device)
        print("embeddings' shape is :", embeddings_to_reconstruct.shape)
        reconstructed_data = decoder(embeddings_to_reconstruct).cpu().numpy()
        reconstructed_data = np.expm1(reconstructed_data)
    return torch.tensor(reconstructed_data, dtype=torch.float32).to(device)

def impute(decoder, query_embeddings, expanded_embeddings, device):
    """
    Process the embeddings by finding 10 groups of embeddings in expanded_embeddings that are closest
    to the original embeddings (query_embeddings), sending each group to the decoder separately,
    and then averaging the reconstructed data.
    """
    # Ensure query_embeddings and expanded_embeddings are tensors
    query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32)
    expanded_embeddings = torch.tensor(expanded_embeddings, dtype=torch.float32)

    # Find the k closest groups of expanded embeddings in expanded_embeddings using Euclidean distance
    distances = pairwise_distances(query_embeddings.cpu().numpy(), expanded_embeddings.cpu().numpy(), metric='euclidean')

    top_k_indices = distances.argsort(axis=1)[:, :10]
    top_k_embeddings_groups = []
    for i in range(10):
        indices = top_k_indices[:, i]
        top_k_embeddings_group = expanded_embeddings[indices]
        top_k_embeddings_groups.append(top_k_embeddings_group)

    individual_reconstructions = []
    for top_k_embeddings in top_k_embeddings_groups:
        reconstructed_data = reconstruct_data_from_embeddings(decoder, top_k_embeddings, device)
        individual_reconstructions.append(reconstructed_data)

    imputed_data = torch.mean(torch.stack(individual_reconstructions), dim=0)

    return imputed_data


if __name__ == "__main__":
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    orignal_txt_path = os.path.join(base_dir, 'raw.txt')  #Path to raw scRNA_seq,get the dimension of raw scRNA_seq
    original_data = pd.read_csv(orignal_txt_path, sep=',', header=None, index_col=None)
    print(f"original data shape: {original_data.shape}")

    RNA_input_dim = original_data.shape[1]
    _, rna_decoder = get_encoder_decoder(RNA_input_dim=RNA_input_dim)
    decoder_path = os.path.join(base_dir, 'AE_models/best_rna_decoder.pth')
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder weight file not found: {decoder_path}")
    rna_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    rna_decoder.to(device)
    print("Decoder model loaded.")

    expanded_embeddings = np.load(os.path.join(base_dir, 'Sample/sample_10000x256.npy'))
    query_embeddings = np.load(os.path.join(base_dir, 'best_embeddings.npy'))
    imputed_data = impute(rna_decoder, query_embeddings, expanded_embeddings, device)
    print("imputed data shape:", imputed_data.shape)

    # Save the reconstructed data
    result_path = os.path.join(result_dir, 'result.txt')
    np.savetxt(result_path, imputed_data.cpu().numpy(), delimiter=',', fmt='%.6f')
