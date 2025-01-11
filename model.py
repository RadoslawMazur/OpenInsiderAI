import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dataset import TradeDataset
import datetime
from tqdm import tqdm
import pandas as pd
import math
import torchinfo 
import matplotlib.pyplot as plt

# Transformer Model
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Embedding layer to project input to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, d_model))
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding
        x = self.embedding(x) + self.positional_encoding[:seq_len, :]
        
        # Transform to (seq_len, batch_size, d_model) for transformer
        x = x.permute(1, 0, 2)
        
        # Transformer Encoder
        x = self.transformer(x)
        
        # Transform back to (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)
        
        # Output projection
        output = self.fc_out(x)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        
        # Create a positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Generate div_term for all dimensions (even and odd)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Handle sine for even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Handle cosine for odd indices
        if d_model % 2 != 0:
            # For odd d_model, pad div_term for the odd part
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        pe.requires_grad = True
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


       

class TransAm(nn.Module):
    def __init__(self, feature_size=250, input_dim=25, output_dim=1, seq_len=16, num_layers=1, n_head=8, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        
        # Embedding layer to map input_dim -> feature_size
        self.embedding = nn.Linear(input_dim, feature_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(feature_size)
        
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # Decoder layer to map feature_size -> output_dim
        self.decoder = nn.Linear(feature_size, output_dim)
        # self.init_weights()

    #def init_weights(self):
    #    initrange = 0.1    
    #    self.decoder.bias.data.zero_()
    #    self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Decoded tensor.
        """
        if self.src_mask is None or self.src_mask.size(0) != src.size(1):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(1)).to(device)
            self.src_mask = mask

        # Apply embedding layer
        src = self.embedding(src)  # Shape: (batch_size, seq_len, feature_size)
        
        # Permute to (seq_len, batch_size, feature_size)
        src = src.permute(1, 0, 2)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, self.src_mask)
        
        # Permute back to (batch_size, seq_len, feature_size) and decode
        output = output.permute(1, 0, 2)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    

class GroupedSequenceDataset(Dataset):
    def __init__(self, csv_file: str, group_column: str = "tick", seq_length: int = 8, n_weeks: int = 4, start_week: int = 0, end_week: int = 9999):
        """
        Args:
            csv_file (str): Path to the CSV file.
            group_column (str): Column name to group the data by.
            seq_length (int): Length of the sequences to generate.
        """
        data = pd.read_csv(csv_file)
        self.data = data[data["abs_week"].between(start_week, end_week)]
        self.group_column = group_column
        self.seq_length = seq_length
        self.n_weeks = n_weeks + 1

        # Group data by the group_column
        self.groups = [
            group.sort_values("abs_week", ascending=True) for _, group in self.data.groupby(group_column)
        ]

        # Generate sequences for each group
        self.sequences = []
        for group in self.groups:
            self._create_sequences(group)

    def _create_sequences(self, group):
        """
        Create sequences of length seq_length for a single group.
        """
        num_sequences = len(group) - self.seq_length + 1
        if num_sequences <= 0:
            return  # Skip groups smaller than seq_length

        for i in range(num_sequences):
            sequence = group.iloc[i:i + self.seq_length]
            self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        X = torch.Tensor(sequence[['price', 'qty', 'owned',
            'delta_owned', 'value', 'is_dir', 'is_ceo', 'is_major_steakholder',
            'd_to_filling', 'A - Grant', 'C - Converted deriv',
            'D - Sale to issuer', 'F - Tax', 'G - Gift', 'M - OptEx',
            'P - Purchase', 'S - Sale', 'W - Inherited', 'X - OptEx',
            '52w_GDP_change', '104w_GDP_change', 'FEDFUNDS', '26w_FEDFUNDS_change',
            '52w_FEDFUNDS_change', '-1w_change']].values)
        y = torch.Tensor(sequence[[f"{i}w_change" for i in range(1, self.n_weeks)]].values)
        return X, y

# Training Function
def train_model(model, dataloader_trian, dataloader_test, criterion, optimizer, device, epochs=10):
    model.to(device)
    print(f"Training on {device}")
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 30], gamma=0.1)
    for epoch in range(epochs):
        model.train()
        total_loss = torch.zeros(4).to(device)
        loop_obj = tqdm(dataloader_trian)
        c = 0
        for batch_X, batch_y in loop_obj:

            optimizer.zero_grad()

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.mean(axis=[0, 1])
            c += 1
            
            # Backward pass and optimization
            loss.mean().backward()
            optimizer.step()
            loop_obj.set_description(f"Loss {total_loss.mean().item()/c}")

        
        test_total_loss = 0
        for batch_test_X, batch_test_y in dataloader_test:
            batch_test_X, batch_test_y = batch_test_X.to(device), batch_test_y.to(device) 

            # Forward pass
            test_outputs = model(batch_test_X)

            
            # Compute loss
            test_loss = criterion(test_outputs, batch_test_y)
            test_total_loss += test_loss.mean().item()
           
        x = [1, 2, 3, 4]
        plt.plot(x, batch_test_y.cpu().detach().numpy()[0][-1], label="Ground truth")
        plt.plot(x, test_outputs.cpu().detach().numpy()[0][-1], label="Prediction")
        plt.legend()
        plt.savefig(f"graphs/graph_{epoch + 1}.png")

        print(" ".join([f"Loss for {i}w_change: {l.item() / len(dataloader_trian)}" for i, l in enumerate(total_loss)]))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.mean().item() / len(dataloader_trian)}, Learning Rate: {scheduler1.get_last_lr()}, Test Loss: {test_total_loss / len(dataloader_test)}")
        scheduler1.step()

# Example Usage
if __name__ == "__main__":
    # Example data
    n = 48  # Sequence length
    start_time = datetime.datetime.strptime("01-01-2015", "%d-%m-%Y")
    end_time = datetime.datetime.strptime("01-01-2022", "%d-%m-%Y")

    # Create dataset and dataloader for train
    n_weeks_predicted = 4
    dataset = GroupedSequenceDataset("dataset.csv", n_weeks=n_weeks_predicted, start_week=700, end_week=1100)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create dataset and dataloader for test
    test_dataset = GroupedSequenceDataset("dataset.csv", n_weeks=n_weeks_predicted, start_week=1100, end_week=1200)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = 25
    output_dim = n_weeks_predicted
    seq_len = n
    model = TransAm(input_dim=input_dim, output_dim=output_dim, feature_size=256, seq_len=seq_len, n_head=8, num_layers=2, dropout=0.5)

    torchinfo.summary(model, input_size=(32, seq_len, input_dim), col_names=["input_size", "output_size", "num_params", "kernel_size"], device="cpu")
    # Loss function and optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    train_model(model, dataloader, test_dataloader, criterion, optimizer, device, epochs=50)
