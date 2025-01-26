import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import gaussian_filter1d

hidden_dim = 32
num_layers = 2
num_epochs = 200
batch_size = 16
learning_rate = 0.002
dropout_prob = 0.2
TICK = 'MORN'
early_stopping_patience = 25

# Utworzenie datasetu do trenowania modelu GRU
class GroupedSequenceDataset(Dataset):
    def __init__(self, csv_file: str, group_column: str = "tick", seq_length: int = 8, n_weeks: int = 4):
        data = pd.read_csv(csv_file)
        data.fillna(0, inplace=True)
        data = data[data['tick'] == TICK]
        self.group_column = group_column
        self.seq_length = seq_length
        self.n_weeks = n_weeks
        self.groups = [group for _, group in data.groupby(group_column)]
        self.sequences = []
        for group in self.groups:
            self._create_sequences(group)
        print(f"Created {len(self.sequences)} sequences")

    def _create_sequences(self, group):
        num_sequences = len(group) - self.seq_length
        if num_sequences <= 0:
            return
        for i in range(num_sequences):
            sequence = group.iloc[i:i + self.seq_length]
            self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        X = torch.tensor(sequence[[
            'price', 'qty', 'owned', 'delta_owned', 'value', 'is_dir', 'is_ceo',
            'is_major_steakholder', 'd_to_filling', '52w_GDP_change', '104w_GDP_change',
            'FEDFUNDS', '26w_FEDFUNDS_change', '52w_FEDFUNDS_change', '-1w_change'
        ]].values, dtype=torch.float32)
        y = torch.tensor(sequence[[f"{i}w_change" for i in range(1, self.n_weeks + 1)]].iloc[-1].values, dtype=torch.float32)
        return X, y

# Ładowanie danych
csv_file = "dataset.csv"
dataset = GroupedSequenceDataset(csv_file=csv_file, seq_length=8, n_weeks=4)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Połączenie próbek w jedną listę
X_batches, y_batches = [], []
for X_batch, y_batch in data_loader:
    X_batches.append(X_batch.numpy())
    y_batches.append(y_batch.numpy())

X = np.concatenate(X_batches)
y = np.concatenate(y_batches)

# Podział na zbiór treningowy losowo i testowy w kolejności
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Testowy zbiór jest sekwencyjny
split_ratio = 0.5  # Dzielimy zbiór tymczasowy na pół

X_validation, X_test, y_validation, y_test = X_temp[:int(len(X_temp) * split_ratio)], X_temp[int(len(X_temp) * split_ratio):], y_temp[:int(len(y_temp) * split_ratio)], y_temp[int(len(y_temp) * split_ratio):]

# Skalowanie danych przez StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
y_train_scaled = scaler_y.fit_transform(y_train)
X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
y_test_scaled = scaler_y.transform(y_test)
X_validation_scaled = scaler_X.transform(X_validation.reshape(-1, X_validation.shape[2])).reshape(X_validation.shape)
y_validation_scaled = scaler_y.transform(y_validation)

# Definicja modelu GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # Get the output of the last time step
        out = self.dropout(out)  # Apply dropout to the output
        out = self.fc(out)
        return out

# Konfiguracja modelu
input_dim = X_train.shape[2]

output_dim = y_train.shape[1]
model = GRUModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim, dropout_prob=dropout_prob)

# MSE loss i Adam optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# Dodanie ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.8,
    patience=10,
)



# Ładowanie danych do DataLoader
train_loader = DataLoader(
    list(zip(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32))),
    batch_size=batch_size,
    shuffle=True
)
validation_loader = DataLoader(
    list(zip(torch.tensor(X_validation_scaled, dtype=torch.float32), torch.tensor(y_validation_scaled, dtype=torch.float32))),
    batch_size=batch_size
)
test_loader = DataLoader(
    list(zip(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test_scaled, dtype=torch.float32))),
    batch_size=batch_size
)

# Uczenie modelu
train_losses = []
validation_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # Obliczanie błędu na oryginalnej skali dla zbioru treningowego
    model.eval()
    train_predictions, train_actuals = [], []
    with torch.no_grad():
        for X_train_batch, y_train_batch in train_loader:
            train_outputs = model(X_train_batch)
            train_predictions.append(train_outputs.numpy())
            train_actuals.append(y_train_batch.numpy())

    train_predictions = np.concatenate(train_predictions)
    train_actuals = np.concatenate(train_actuals)
    train_predictions_original = scaler_y.inverse_transform(train_predictions)
    train_actuals_original = scaler_y.inverse_transform(train_actuals)
    train_mse_original = np.mean((train_actuals_original - train_predictions_original) ** 2)

    train_losses.append(train_mse_original)

    # Walidacja modelu
    validation_loss = 0.0
    validation_predictions, validation_actuals = [], []
    with torch.no_grad():
        for X_val_batch, y_val_batch in validation_loader:
            val_outputs = model(X_val_batch)
            validation_predictions.append(val_outputs.numpy())
            validation_actuals.append(y_val_batch.numpy())

    validation_predictions = np.concatenate(validation_predictions)
    validation_actuals = np.concatenate(validation_actuals)
    validation_predictions_original = scaler_y.inverse_transform(validation_predictions)
    validation_actuals_original = scaler_y.inverse_transform(validation_actuals)
    validation_mse_original = np.mean((validation_actuals_original - validation_predictions_original) ** 2)
    scheduler.step(validation_mse_original)

    validation_losses.append(validation_mse_original)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training MSE: {train_mse_original:.4f}, Validation MSE: {validation_mse_original:.4f}")

# Wykres strat treningowych i walidacyjnych w czasie
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training MSE over epochs")
plt.plot(validation_losses, label="Validation MSE over epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"{TICK} -Training and Validation Loss Over Time")
plt.legend()
plt.grid()


# Testowanie modelu
model.eval()


predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predictions.append(outputs.numpy())
        actuals.append(y_batch.numpy())



predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

# Odwrócenie skalowania
predictions_original = scaler_y.inverse_transform(predictions)
actuals_original = scaler_y.inverse_transform(actuals)

mse = np.mean((actuals_original - predictions_original) ** 2)
print(f"Mean Squared Error (MSE): {mse:.4f}")
plt.savefig(f"companies/{TICK}validations-mse{mse:.2f}-hidden_dim{hidden_dim}-layers{num_layers}-epochs{num_epochs}-batch_size{batch_size}-learing_rate{learning_rate}-dropout{dropout_prob}.png")

# Zapisanie modelu
torch.save(model, f"{TICK}_{mse:.2f}.pth")

# Wykresy dla kolejnych tygodni
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for week in range(4):
    smoothed_predictions = gaussian_filter1d(predictions_original[:, week], sigma=2)
    smoothed_actuals = gaussian_filter1d(actuals_original[:, week], sigma=2)
    weeks = np.arange(len(smoothed_predictions)) + 1
    axes[week].plot(weeks, smoothed_predictions, label=f"Predicted {week + 1} week change")
    axes[week].plot(weeks, smoothed_actuals, label=f"Actual {week + 1} week change")
    axes[week].set_xlabel("Weeks")
    axes[week].set_ylabel("Value")
    axes[week].set_title(f"{TICK} Predictions vs Actuals ({week + 1} week change)")
    axes[week].legend()

plt.tight_layout()
plt.savefig(f"companies/{TICK}predictions-mse{mse:.2f}-hidden_dim{hidden_dim}-layers{num_layers}-epochs{num_epochs}-batch_size{batch_size}-learing_rate{learning_rate}-dropout{dropout_prob}.png")
