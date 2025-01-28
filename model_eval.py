from torch.utils.data import DataLoader
from model import TransformerModel, GroupedSequenceDataset, train_model
from torch import nn
from torch import optim
import torch
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d

tickers = ["MORN", "LEG", "AAPL", "CRM"]
TRAIN_SIZE = 5*52 # ok. 5 lat danych treningowych
TEST_SIZE = 26 # 24 tygodnie na sprawdzenie modelu
NEXT_TEST_OFFSET = 26 
dataset_file_name = "dataset2.csv"
n = 8 # number of weeks that the model `sees` at one time


def plot_training(weeks_train, gt_train, pred_train, weeks_test, gt_test, pred_test, ticker):

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
    axs = axs.ravel()  # Flatten the 2D array of axes into a 1D array for easy iteration

    for i in range(4):  # Hard-coded for 4 elements
        axs[i].plot(weeks_train, gaussian_filter1d(pred_train[i], sigma=1e-6), label=f'Predicted Train')
        axs[i].plot(weeks_train, gt_train[i], label=f'Ground Truth Train')
        axs[i].plot(weeks_test, gaussian_filter1d(pred_test[i], sigma=1e-6), label=f'Predicted Test')
        axs[i].plot(weeks_test, gt_test[i], label=f'Ground Truth Test')

        axs[i].set_title(f"Percenatage change in {i + 1} weeks")  # Placeholder for title
        axs[i].set_xlabel("Weeks")  # Placeholder for x-axis label
        axs[i].set_ylabel(f"Percentage change ")  # Placeholder for y-axis label
        axs[i].legend()  # Add legend to each subplot

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(f"Ticker_{ticker}.png")

def create_boxplots(train_losses: dict, test_losses: dict):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 grid of subplots
    axs = axs.ravel()  # Flatten the 2D array of axes into a 1D array for easy iteration

    for i, key in enumerate(train_losses.keys()):  # Iterate through keys (common between dicts)
        data = [train_losses[key], test_losses[key]]
        axs[i].boxplot(data, labels=['Train', 'Test'])
        axs[i].set_title(f"Loss Distribution for {key}")  # Placeholder for title
        axs[i].set_ylabel("Loss Values")  # Placeholder for y-axis label

    plt.tight_layout()  # Adjust spacing between subplots
    plt.savefig(f"Boxplots.png")

train_losses = dict() 
test_losses = dict()
for ticker in tickers:
    # Trenowanie modelu dla jednoego ticker
    start_week = 300
    train_losses[ticker] = []
    test_losses[ticker] = []
    for i in range(5):

        # Setting up time bounds for the test
        start_time_week = start_week
        end_time_week = start_time_week + TRAIN_SIZE
        start_test_time_week = end_time_week + 1
        end_test_time_week = start_test_time_week + TEST_SIZE
        start_week += NEXT_TEST_OFFSET

        # Create dataset and dataloader for train
        dataset = GroupedSequenceDataset(dataset_file_name, ticker=ticker, start_week=start_time_week, end_week=end_time_week, seq_length=n)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Create dataset and dataloader for test
        test_dataset = GroupedSequenceDataset(dataset_file_name, ticker=ticker, start_week=start_test_time_week, end_week=end_test_time_week, scaler=dataset.scaler, seq_length=n)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


        model_args = {
            "output_dim": 4, 
            "feature_size": 26, 
            "seq_len": n, 
            "n_head": 2, 
            "num_layers": 1, 
            "dropout": 0.5, 
            "dim_feedforward": 8
            }

        model = TransformerModel(**model_args)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_trained, last_epoch_train_loss, last_epoch_test_loss = train_model(model, dataloader, test_dataloader, criterion, optimizer, device, 250)
        train_losses[ticker].append(last_epoch_train_loss)
        test_losses[ticker].append(last_epoch_test_loss)

        # get model inference for train and test set 
        unshuffled_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        train_weeks = []
        train_results = []
        train_ground_truth = []
        train_week = dataset.min_week 

        for batch_X, batch_y in unshuffled_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device) 
            batch_predcitions = model(batch_X) # this should be 1 (batch_size) x 4 (num_inputs)

            train_weeks.append(train_week)
            train_results.append(batch_predcitions[0].to("cpu").detach().tolist())
            train_ground_truth.append(batch_y[0].to("cpu").detach().tolist())
            train_week += 1

        train_res_transposed =list(map(list, zip(*train_results))) 
        train_gt_transposed =list(map(list, zip(*train_ground_truth))) 

        unshuffled_test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_weeks = []
        test_results = []
        test_ground_truth = []
        test_week = test_dataset.min_week 

        for batch_test_X, batch_test_y in unshuffled_test_dataloader:
            batch_test_X, batch_test_y = batch_test_X.to(device), batch_test_y.to(device) 
            batch_test_predictions = model(batch_test_X)

            test_weeks.append(test_week)
            test_results.append(batch_test_predictions[0].to("cpu").detach().tolist())
            test_ground_truth.append(batch_test_y[0].to("cpu").detach().tolist())
            test_week += 1

        test_res_transposed =list(map(list, zip(*test_results))) 
        test_gt_transposed =list(map(list, zip(*test_ground_truth))) 

        if i == 4:
            plot_training(train_weeks, train_gt_transposed, train_res_transposed, test_weeks, test_gt_transposed, test_res_transposed, ticker)

create_boxplots(train_losses, test_losses)


 