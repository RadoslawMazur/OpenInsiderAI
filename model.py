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
from sklearn.preprocessing import StandardScaler
import os
import torchview


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)
    

class TransformerModel(nn.Module):
    def __init__(self, feature_size=250, 
                 output_dim=4, seq_len=16, 
                 num_layers=1, n_head=8,
                 dropout=0.1, dim_feedforward=16):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(feature_size)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=n_head, dropout=dropout, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(feature_size*seq_len, output_dim)


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
        
        src = src.permute(1, 0, 2)
        
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(src, self.src_mask)
        
        output = output.permute(1, 0, 2)
        output = output.flatten(start_dim=1)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    

class GroupedSequenceDataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 ticker: str = None,
                 group_column: str = "tick", 
                 seq_length: int = 8, 
                 n_weeks: int = 4, 
                 start_week: int = 0, 
                 end_week: int = 9999,
                 scaler: StandardScaler|None = None
                 ):
        """
        Args:
            csv_file (str): Path to the CSV file.
            group_column (str): Column name to group the data by.
            seq_length (int): Length of the sequences to generate.
        """
        data = pd.read_csv(csv_file)
        self.data = data[data["abs_week"].between(start_week, end_week)]

        if ticker:
            self.data = self.data[self.data["tick"] == ticker]

        change_vols = self.data["-1w_change_volume"].unique()
        change_vols.sort()
        self.data.loc[self.data["-1w_change_volume"] == self.data["-1w_change_volume"].max(), "-1w_change_volume"] = change_vols[-2]

        self.min_week = self.data["abs_week"].min()
        self.max_week = self.data["abs_week"].max()

        self.x_columns = ['qty', 'owned', 'volume', '-1w_change_volume',
            'delta_owned', 'value', 'is_dir', 'is_ceo', 'is_major_steakholder',
            'd_to_filling', 'A - Grant', 'C - Converted deriv',
            'D - Sale to issuer', 'F - Tax', 'G - Gift', 'M - OptEx',
            'P - Purchase', 'S - Sale', 'W - Inherited', 'X - OptEx',
            '52w_GDP_change', '104w_GDP_change', 'FEDFUNDS', '26w_FEDFUNDS_change',
            '52w_FEDFUNDS_change', '-1w_change']

        if not scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(self.data[self.x_columns].values)
        else:
            self.scaler = scaler

        self.data.loc[:, self.x_columns] = self.scaler.transform(self.data[self.x_columns].values)
        self.group_column = group_column
        self.seq_length = seq_length
        self.n_weeks = n_weeks + 1

        self.groups = [
            group.sort_values("abs_week", ascending=True) for _, group in self.data.groupby(group_column)
        ]

        self.sequences = []
        for group in self.groups:
            self._create_sequences(group)

    def _create_sequences(self, group):
        """
        Create sequences of length seq_length for a single group.
        """
        num_sequences = len(group) - self.seq_length + 1

        for i in range(num_sequences):
            sequence = group.iloc[i:i + self.seq_length]
            self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        X = torch.Tensor(sequence[self.x_columns].values)
        y = torch.Tensor(sequence[[f"{i}w_change" for i in range(1, self.n_weeks)]].tail(1).values[0] * 100)
        return X, y
    

def plot_training(n_epochs, train_losses, test_losses, learning_rates, ticker=None):
    """
    Plots training losses, test losses, and learning rates over the epochs.

    Parameters:
    n_epochs (int): Number of epochs
    train_losses (list): List of training loss values per epoch
    test_losses (list): List of test/validation loss values per epoch
    learning_rates (list): List of learning rates per epoch
    """
    epochs = range(1, n_epochs + 1)  # Create a list of epoch numbers
    
    # Create the figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and test losses
    ax1.set_title("Training Progress")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    ax1.plot(epochs, test_losses, label="Test Loss", color="red", marker="o")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Create a twin y-axis to plot learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate")
    ax2.plot(epochs, learning_rates, label="Learning Rate", color="green", linestyle="--")
    ax2.legend(loc="upper right")

    # Show the plot
    plt.tight_layout()

    if not ticker:
        ticker = "all"

    plt.savefig(f"training_{ticker}.png")



# Training Function
def train_model(model, dataloader_trian, dataloader_test, criterion, optimizer, device, epochs=10, ticker=None):
    model.to(device)
    print(f"Training on {device}")
    train_losses = []
    test_losses = []
    learning_rates = []
    scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 50, 150], gamma=0.5)
    for epoch in range(epochs):
        model.train()
        total_loss = torch.zeros(1).to(device)
        loop_obj = tqdm(dataloader_trian)
        c = 0
        for batch_X, batch_y in loop_obj:

            optimizer.zero_grad()

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            c += 1
            
            loss.backward()
            optimizer.step()
            loop_obj.set_description(f"Loss {total_loss/c}")

        
        test_total_loss = 0
        for batch_test_X, batch_test_y in dataloader_test:
            batch_test_X, batch_test_y = batch_test_X.to(device), batch_test_y.to(device) 

            test_outputs = model(batch_test_X)
            
            test_loss = criterion(test_outputs, batch_test_y)
            test_total_loss += test_loss.item()

        train_losses.append(total_loss.item() / len(dataloader_trian))
        test_losses.append(test_total_loss / len(dataloader_test))
        learning_rates.append(scheduler1.get_last_lr())
           
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item() / len(dataloader_trian)}, Learning Rate: {scheduler1.get_last_lr()}, Test Loss: {test_total_loss / len(dataloader_test)}")
        scheduler1.step()

    plot_training(epochs, train_losses, test_losses, learning_rates, ticker)
    last_epoch_loss = total_loss.item() / len(dataloader_trian)
    last_epoch_test_loss = test_total_loss / len(dataloader_test)

    return model, last_epoch_loss, last_epoch_test_loss

# Example Usage
if __name__ == "__main__":

    n = 4  # Sequence length
    start_time = datetime.datetime.strptime("01-01-2014", "%d-%m-%Y")
    start_time_week = TradeDataset.date_to_week(start_time)

    end_time = datetime.datetime.strptime("28-02-2017", "%d-%m-%Y")
    end_time_week = TradeDataset.date_to_week(end_time)
    print(f"Training from week {start_time_week} to week {end_time_week}")

    start_test_time = datetime.datetime.strptime("01-03-2017", "%d-%m-%Y")
    start_test_time_week = TradeDataset.date_to_week(start_test_time)

    end_test_time = datetime.datetime.strptime("30-06-2017", "%d-%m-%Y")
    end_test_time_week = TradeDataset.date_to_week(end_test_time)
    print(f"Training from week {start_test_time_week} to week {end_test_time_week}")

    dataset_file_name = "dataset2.csv"
    n_weeks_predicted = 4
    #ticker = "LEG"
    ticker = None

    # Create dataset and dataloader for train
    dataset = GroupedSequenceDataset(dataset_file_name, ticker=ticker, n_weeks=n_weeks_predicted, start_week=start_time_week, end_week=end_time_week, seq_length=n)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Train has {len(dataset)} samples")

    # Create dataset and dataloader for test
    test_dataset = GroupedSequenceDataset(dataset_file_name, ticker=ticker, n_weeks=n_weeks_predicted, start_week=start_test_time_week, end_week=end_test_time_week, scaler=dataset.scaler, seq_length=n)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    print(f"Train has {len(test_dataset)} samples")

    # Initialize model
    input_dim = 26
    model = TransformerModel(output_dim=n_weeks_predicted, 
                    feature_size=input_dim, seq_len=n, 
                    n_head=2, num_layers=1,
                    dropout=0.5, dim_feedforward=8)

    torchinfo.summary(model, input_size=(16, n, input_dim), col_names=["input_size", "output_size", "num_params", "kernel_size"], device="cpu")
    torchview.draw_graph(model, input_size=(16, n, input_dim), device="cpu", filename="model_graph.png", graph_name="Transformer model", save_graph=True)
    
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    model, _, _ = train_model(model, dataloader, test_dataloader, criterion, optimizer, device, epochs=50, ticker=ticker)
    torch.save(model.state_dict(), os.path.join("models", f"{ticker}_{n}_{end_test_time_week}")) 
