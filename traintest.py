import argparse
import os
import sys
import logging
import time
from datetime import datetime
import pytz
import json

import numpy as np
import torch
from torch import nn
from torchinfo import summary
from utils import *
from dataProcess import *

import STID

class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                torch.save(model.state_dict(), f'{SAVING_PATH}/{MODEL_NAME}_{DATASET}_{LOSS}_EarlyStop.pt')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'{SAVING_PATH}/{MODEL_NAME}_{DATASET}_{LOSS}.pt')
        self.val_loss_min = val_loss


@torch.no_grad()
def model_evaluate(model, dataLoader, criterion, debug=False):
    model.eval()
    
    batch_losses = []
    y_true, y_pred = [], []
    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch, NULL_VAL)
        batch_losses.append(loss.item())

        if not debug:
            continue
        
        y_batch = SCALER.inverse_transform(y_batch)
        if y_batch.is_cuda:
            y_true.append(y_batch.cpu().numpy())
        else:
            y_true.append(y_batch.numpy())

        if out_batch.is_cuda:
            y_pred.append(out_batch.cpu().numpy())
        else:
            y_pred.append(out_batch.numpy())
    
    if debug:
        return np.mean(batch_losses), y_true, y_pred
    else:
        return np.mean(batch_losses)


@torch.no_grad()
def model_predict(model, dataLoader):
    y_pred, y_true = [], []
    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        
        out_batch = SCALER.inverse_transform(out_batch)
        # y_batch = SCALER.inverse_transform(y_batch)
        
        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        
        y_true.append(y_batch)
        y_pred.append(out_batch)

    y_true = np.vstack(y_true).squeeze()
    y_pred = np.vstack(y_pred).squeeze()
    return y_pred, y_true


def oneStepForward(model, dataLoader, optimizer, scheduler, criterion):
    model.train()
    batch_losses = []

    for x_batch, y_batch in dataLoader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch, NULL_VAL)
        batch_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.mean(batch_losses)
    scheduler.step()
    
    return epoch_loss
        
    

def model_train(model, trainLoader, valLoader, optimizer, scheduler, criterion,
                epochs=400, early_stopping=None, compile_model=True, verbose=1, logger=None, plot=True, save=False):

    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)
    
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        train_loss = oneStepForward(model, trainLoader, optimizer, scheduler, criterion)
        train_losses.append(train_loss)

        val_loss = model_evaluate(model, valLoader, criterion)
        val_losses.append(val_loss)

        early_stopping(val_loss, model)

        if logger:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(torch.load(f'{SAVING_PATH}/{MODEL_NAME}_{DATASET}_{LOSS}_EarlyStop.pt'))
            break
            
    if not early_stopping.early_stop:
        torch.save(model.state_dict(), f'{SAVING_PATH}/{MODEL_NAME}_{DATASET}_{LOSS}.pt') 

    if plot:
        metric_plot(train_losses, val_losses)
    
    return model

@torch.no_grad()
def model_test(model, dataLoader, logger=None):
    model.eval()

    if logger:
        logger.info("--------- Model Test ---------")

    start_time = time.time()
    y_pred, y_true = model_predict(model, dataLoader)
    end_time = time.time()
    np.save(f"{SAVING_PATH}/TrueVal.npy", y_true)
    np.save(f"{SAVING_PATH}/PredVal.npy", y_pred)

    rmse_all, mae_all, mape_all = getMetric(y_pred, y_true, NULL_VAL)
    if logger:
        logger.info(f"All steps RMSE = {rmse_all:.5f}, MAE = {mae_all:.5f}, MAPE = {mape_all:.5f}\n")

    for i in range(y_pred.shape[1]):
        rmse, mae, mape = getMetric(y_pred[:, i, :], y_true[:, i, :], NULL_VAL)
        if logger:
            logger.info(f"--- Step {i+1} RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}\n")

    logger.info(f"Inference time:{(end_time - start_time):.2f}s.")
    
        
    
def metric_plot(train_losses, val_losses):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("darkgrid")
    sns.set_context("paper")
    plt.figure(figsize=(59, 6))

    data = pd.DataFrame({
        'iteration': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    
    sns.lineplot(data=data, x='iteration', y='train_loss', label='train_loss', marker='x')
    sns.lineplot(data=data, x='iteration', y='val_loss', label='val_loss', marker='o')
    plt.xlabel("iteration", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(title='Function', title_fontsize='13', fontsize='11')
    plt.savefig(f'{SAVING_PATH}/loss.png')


def save_args_to_file(args):
    with open(f'{SAVING_PATH}/PARAS.json', 'w') as f:
        json.dump(vars(args), f, indent=4) 


if __name__ == "__main__":
    """
    INITIALIZATION
    """

    # Parameter setting
    print("Model Initialization...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="STID", help='Model name to save output in file')
    parser.add_argument('-s', '--step', default=12, type=int, help='Step to predict (*5 minute).')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Number of batches to train in each epoch.')
    parser.add_argument('-d', '--dataset', type=str, default='TEST', help='Dataset name.')
    parser.add_argument('--data_path', type=str, default='./data/test.csv', help='Path of dataset file')
    parser.add_argument('--null', type=float, default=30, help="Null val to ignore.")
    parser.add_argument('--gpu', default=-1, type=int, help='ID of the gpu to run on. -1(default) means GPU with most free memory will be chosen.')
    parser.add_argument('--verbose', default=1, type=int, help='Default is 1, if you donn\'t want any log, please / set verbose to 0.')
    parser.add_argument('-l', "--loss", type=str, default='MAE', help='MAE, MSE, RMSE, MAPE, Hyrbrid')
    parser.add_argument('-c', "--coeffcient", type=float, default=0.01, help='HYBRID coeffcient, e.g. a * MAE + MAPE.')
    parser.add_argument('-e', "--epochs", type=int, default=200, help='Length of each epoch.')
    parser.add_argument('--json', type=str, default='./start.json', help='Pass parameters for inference.')
    parser.add_argument('--nodes', type=int, default=None, help='Number of nodes of input data.')
    

    args = parser.parse_args()
    MODEL_NAME = args.model
    STEP = args.step
    DATASET = args.dataset
    DATA_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    NULL_VAL = args.null
    LOSS = args.loss
    EPOCHS = args.epochs

    with open(DATA_PATH, 'r') as f:
        header = f.readline().strip()
        column_count = len(header.split(','))
    NODES = column_count - 1
    args.nodes = NODES
    
    current_time = datetime.now(pytz.utc)
    tokyo_timezone = pytz.timezone('Asia/Tokyo')
    current_time = current_time.astimezone(tokyo_timezone)
    formatted_time = current_time.strftime("%y%m%d%H%M%S")
    
    SAVING_PATH = f"./saved_models/{MODEL_NAME}_{DATASET}_{formatted_time}"
    
    if not os.path.exists(SAVING_PATH):
        os.makedirs(SAVING_PATH)
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    if not os.path.exists('./cache'):
        os.makedirs('./cache')

    save_args_to_file(args)
    
    log_file = f"./logs/{formatted_time}_{MODEL_NAME}_{DATASET}_training.log"
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    
    GPU_ID = select_gpu() if args.gpu == -1 else args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    #Data Loading
    print(f"Loading dataset located in {DATASET} ...")
    trainLoader, valLoader, testLoader, SCALER = load_data(DATASET, DATA_PATH, batch_size=BATCH_SIZE, step=STEP)
    print(f"{DATASET} loading is finished!")
    # trainLoader, valLoader, testLoader, SCALER = generate_train_val_test(DATA_PATH, batch_size=BATCH_SIZE, step=STEP)
    
    
    # Model initializaiton
    model = STID.STID(NODES, input_len=STEP, output_len=STEP).to(DEVICE)

    print(f"{MODEL_NAME} initialization is finished.")

    criterion = myLoss(LOSS)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-3,
        weight_decay=0.0008,
        eps=1e-8,
    )
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20,30],
        gamma=0.1,
    )
    
    early_stopping = EarlyStopping(patience=10, delta=1e-5)
    
    """
    MODEL TRANING
    """
    
    model = model_train(
        model, trainLoader, valLoader, optimizer, scheduler, criterion, epochs=EPOCHS, early_stopping=early_stopping, logger=logger
    )
    
    print('Tranning is over, test starts...')
    
    model_test(model, testLoader, logger=logger)

    print('Testing is over!')

    JSON_FILE = args.json
    with open(JSON_FILE, 'r') as f:
        infer_paras = json.load(f)

    infer_paras['inference_parameters']['model_path'] = SAVING_PATH

    with open(JSON_FILE, 'w') as f:
        json.dump(infer_paras, f, indent=2)
    
        
    
    
    