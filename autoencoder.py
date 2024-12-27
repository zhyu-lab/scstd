import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

base_dir = './Testdata'
model_dir = os.path.join(base_dir, 'AE_models')
os.makedirs(model_dir, exist_ok=True)

# seed
def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.05)

class NetBlock(nn.Module):
    def __init__(self, nlayer: int, dim_list: list, act_list: list, dropout_rate: float, noise_rate: float, final_activation=None):
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        self.final_activation = final_activation

        for i in range(nlayer):
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if i < nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

        if self.final_activation == 'prelu':
            self.final_activation_layer = nn.PReLU()

        initialize_weights(self)

    def forward(self, x):
        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if i < self.nlayer - 1:
                x = self.dropout_list[i](x)
            elif i == self.nlayer - 1:
                if self.final_activation == 'prelu':
                    x = self.final_activation_layer(x)
        return x

def get_encoder_decoder(RNA_input_dim):
    rna_encoder = NetBlock(
        nlayer=3,
        dim_list=[RNA_input_dim, 1024, 512, 256],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.1,
        noise_rate=0.5,
        final_activation=None
    )
    rna_decoder = NetBlock(
        nlayer=3,
        dim_list=[256, 512, 1024, RNA_input_dim],
        act_list=[nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        dropout_rate=0.4,
        noise_rate=0,
        final_activation='prelu'
    )
    return rna_encoder, rna_decoder

def load_rna_data(file_path, train_ratio=0.8, batch_size=200, device='cpu'):
    df = pd.read_csv(file_path, sep=',', header=None, index_col=None)
    normalized_data = df.div(df.sum(axis=1), axis=0) * 1e4
    log_transformed_data = np.log1p(normalized_data.values)

    data = torch.tensor(log_transformed_data, dtype=torch.float32).to(device)
    print(f"data shape: {data.shape}")
    dataset = TensorDataset(data)

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, all_loader

def train_ed(rna_encoder, rna_decoder, train_loader, val_loader, all_loader, optimizer, epochs=100, device='cpu', patience=30):
    """
       Train encoder and decoder with training and validation sets, tracking losses for plotting.
       Also, generate embeddings for all data when validation loss improves.

       Args:
           rna_encoder (nn.Module): Encoder model.
           rna_decoder (nn.Module): Decoder model.
           train_loader (DataLoader): DataLoader for training data.
           val_loader (DataLoader): DataLoader for validation data.
           all_loader (DataLoader): DataLoader for the entire dataset.
           optimizer (torch.optim.Optimizer): Optimizer for training.
           epochs (int): Number of training epochs.
           device (str): Device to train on ('cpu' or 'cuda').
           patience (int): Number of epochs to wait for improvement before early stopping.

       Returns:
           tuple: (trained_rna_encoder, trained_rna_decoder)
       """
    rna_encoder.to(device)
    rna_decoder.to(device)
    r_loss_fn = nn.MSELoss()
    best_val_loss, patience_counter = float('inf'), 0

    train_losses, val_losses = [], []
    best_embeddings = None

    for epoch in range(epochs):
        rna_encoder.train()
        rna_decoder.train()
        train_loss = 0
        for rna_batch, in train_loader:
            rna_batch = rna_batch.to(device)
            optimizer.zero_grad()
            encoded = rna_encoder(rna_batch)
            decoded = rna_decoder(encoded)
            loss = r_loss_fn(decoded, rna_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        rna_encoder.eval()
        rna_decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for rna_batch, in val_loader:
                rna_batch = rna_batch.to(device)
                encoded = rna_encoder(rna_batch)
                decoded = rna_decoder(encoded)
                loss = r_loss_fn(decoded, rna_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping and Saving Best Model & Embeddings
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(rna_encoder.state_dict(), os.path.join(model_dir, 'best_rna_encoder.pth'))
            torch.save(rna_decoder.state_dict(), os.path.join(model_dir, 'best_rna_decoder.pth'))
            print("Validation loss improved, saving model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # Load the best model
    rna_encoder.load_state_dict(torch.load(os.path.join(model_dir, 'best_rna_encoder.pth')))
    rna_decoder.load_state_dict(torch.load(os.path.join(model_dir, 'best_rna_decoder.pth')))

    all_embeddings = []
    rna_encoder.eval()
    with torch.no_grad():
        for rna_batch, in all_loader:
            rna_batch = rna_batch.to(device)
            encoded = rna_encoder(rna_batch)
            all_embeddings.append(encoded.cpu())

    best_embeddings = torch.cat(all_embeddings).numpy()
    print(best_embeddings.shape)

    if best_embeddings is not None:
        np.save(os.path.join(base_dir, 'best_embeddings.npy'), best_embeddings)

    # Plotting the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()

    return rna_encoder, rna_decoder

if __name__ == "__main__":
    set_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    file_path = os.path.join(base_dir, 'raw.txt')

    train_loader, val_loader, all_loader = load_rna_data(file_path, batch_size=100, device=device)

    # Get input dimension
    RNA_input_dim = next(iter(train_loader))[0].shape[1]

    rna_encoder, rna_decoder = get_encoder_decoder(RNA_input_dim)

    optimizer = torch.optim.Adam(list(rna_encoder.parameters()) + list(rna_decoder.parameters()), lr=1e-3)

    # Train model
    rna_encoder, rna_decoder = train_ed(
        rna_encoder=rna_encoder,
        rna_decoder=rna_decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        all_loader=all_loader,
        optimizer=optimizer,
        epochs=500,
        device=device
    )
