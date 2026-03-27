from utils import *
from dataProcess import StandardScaler
import argparse
import os
import pandas as pd
import sys
import logging
import gpustat
import json
import numpy as np
import STID
from glob import glob
from datetime import timedelta


def generate_graph_seq2seq_io_data(
        df, step, add_time_in_day=True, add_day_in_week=True
):
    
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values.astype(np.float32), axis=-1)
    data_list = [data]
    
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0)).astype(np.float32)
        data_list.append(time_in_day)

    """Label Encoding"""
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 1), dtype=np.float32)
        dayofweek_array = df.index.dayofweek.to_numpy(np.float32)
        day_in_week[:, :, 0] = np.tile(dayofweek_array[:, np.newaxis], (1, num_nodes))
        data_list.append(day_in_week)
    
    data = np.concatenate(data_list, axis=-1)
    x, y = [], []
    # t is the index of the last observation.
    x_offsets = np.sort(np.arange(1-step, 1, 1))
    min_t = abs(min(x_offsets))
    
    for t in range(min_t, num_samples):
        x_t = data[t + x_offsets, ...]
        x.append(x_t)
    x = np.stack(x, axis=0)
    return x


def gen_test_data(DATA_PATH, TRAIN_DATA_PATH):
    test_df = pd.read_csv(TRAIN_DATA_PATH, nrows=0, index_col=0)
    train_ids = list(test_df)

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True, nrows=288)
    idx = df.index
    tdf = pd.DataFrame(0, index=idx, columns=train_ids)
    common_columns = df.columns.intersection(train_ids)
    tdf[common_columns] = df[common_columns]
    
    x = generate_graph_seq2seq_io_data(
        tdf,
        step=STEP,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    return x


def save_results(output,id2mesh, DATA_PATH, TRAIN_DATA_PATH):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True, nrows=288)
    
    test_columns = pd.read_csv(TRAIN_DATA_PATH, nrows=0, index_col=0).columns
    train_columns = df.columns
    common_columns = train_columns.intersection(test_columns)
    
    temp = pd.DataFrame(output, columns=test_columns)
    odf = pd.DataFrame(0, index=temp.index, columns=train_columns)
    odf[common_columns] = temp[common_columns]

    odf.rename(columns=id2mesh, inplace=True)
    odf.index = df.index[-STEP:] + timedelta(minutes=STEP*5)
    ocsv = odf.stack().reset_index()
    ocsv.columns = ['time', 'mesh', 'ex_volume']
    ocsv['ex_volume'] = ocsv['ex_volume'].astype(np.int32)

    res_name = f"{DATA_PATH.split('/')[-1][:-4]}_predict.csv"
    ocsv.to_csv(f'{OUTPUT_PATH}/{res_name}', index=False)
    print(f"Writting is finished! Results are saved in {OUTPUT_PATH}.")
    

if __name__ == "__main__":
    
    """ Inference Initialization """
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model_path", type=str, default='./saved_models/STID_kochi_M_202401_241026232700', help='Path of the directory saved xx.pth, parameters, etc.')
    parser.add_argument("--mesh_file", type=str, default='./data/oneDayData/kochi/meshes.csv',
                       help='File path to the mesh mapping file.')
    
    parser.add_argument('-d', '--data_path', type=str, default='./data/test.csv', help='Path of dataset file')
    parser.add_argument('-o', '--output_path', type=str, default='./results', help='the path to save the prediction results')
    
    args = parser.parse_args()
    MODEL_PATH = args.model_path
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    MESH_FILE = args.mesh_file

    # Loading Parameters    
    with open(os.path.join(MODEL_PATH, 'PARAS.json'), 'r') as file:
        parameters = json.load(file)
        
    NODES = parameters['nodes']
    
    STEP = parameters['step']
    BATCH_SIZE = parameters['batch_size']
    TRAIN_DATA_PATH = parameters['data_path']
    
    # Loading model
    model = STID.STID(NODES, input_len=STEP, output_len=STEP)
    state_dict = torch.load(glob(f'{MODEL_PATH}/*.pt')[0])
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        new_state_dict[new_key] = v
        
    GPU_ID = select_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()

    print("Model Loading is finished!")

    # """Generate Data"""
    MESH_MAP = pd.read_csv(MESH_FILE)   
    id2mesh = { str(item['id']):item['mesh'] for _, item in MESH_MAP.iterrows()}

    x = gen_test_data(DATA_PATH, TRAIN_DATA_PATH)
    
    scaler = StandardScaler(mean=x[..., 0].mean(), std=x[..., 0].std())
    x[..., 0] = scaler.transform(x[..., 0])
    
    input = x[-1:, ::]
    input = torch.FloatTensor(input)

    print("Data is ready, start to inference...")

    output = model(input.to(DEVICE)).cpu().detach().numpy()
    output = np.vstack(output).squeeze()
    output = scaler.inverse_transform(output)

    print("Inferencing finished, ready to write to the output csv file.")

    save_results(output, id2mesh, DATA_PATH, TRAIN_DATA_PATH)

    
    

    
    
    
        