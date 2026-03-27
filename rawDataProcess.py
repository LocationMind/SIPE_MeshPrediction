import jpmesh
import pandas as pd
import numpy as np
import os
import argparse
import glob
import json
from utils import *

MESH_ID = []
ID_INDEX = {}


def fill_missing(data, method='ffill'):
    if method not in ['ffill', 'bfill', 'interpolate']:
        print('Not Implemented, using default forward fill.')
        method = 'ffill'
    
    # Generate 5 Min Time range.
    data['time'] = pd.to_datetime(data['time'])
    start_datetime = pd.to_datetime(data.iloc[0]['time'].date())
    end_datetime = start_datetime + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq='5T')
    
    missing_times = date_range.difference(data['time'])
    
    if len(missing_times) == 0:
        print('No missing detected!')
        data['time'] = data['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return data

    data_to_fill = pd.DataFrame(columns=data.columns)  # Initialize outside the loop

    for mst in missing_times:
        prev_time = mst - pd.Timedelta(minutes=5)

        # Handle missing entry at the start of the day
        if (mst - prev_time).days > 0:
            prev_mesh = data['mesh'].unique()
            prev_vols = [0 for _ in range(len(prev_mesh))]
        else:
            prev_mesh = data.loc[data['time'] == prev_time, 'mesh'].values
            prev_vols = data.loc[data['time'] == prev_time, 'ex_volume'].values

        next_time = mst + pd.Timedelta(minutes=5)

        # Handle missing entry at the end of the day
        if (next_time - mst).days > 0:
            next_mesh = data['mesh'].unique()
            next_vols = [0 for _ in range(len(next_mesh))]
        else:
            next_mesh = data.loc[data['time'] == next_time, 'mesh'].values
            next_vols = data.loc[data['time'] == next_time, 'ex_volume'].values
        
        fill_dict = {}

        if method == 'ffill':
            fill_dict['mesh'] = next_mesh
            fill_dict['ex_volume'] = next_vols
            fill_dict['time'] = mst

        elif method == 'bfill':
            fill_dict['mesh'] = prev_mesh
            fill_dict['ex_volume'] = prev_vols
            fill_dict['time'] = mst

        elif method == 'interpolate':
            avg = {prev_mesh[i]: prev_vols[i] for i in range(len(prev_mesh))}
            
            for i in range(len(next_mesh)):
                if next_mesh[i] in avg:
                    avg[next_mesh[i]] += next_vols[i]
                    avg[next_mesh[i]] = int(avg[next_mesh[i]] / 2)
                else:
                    avg[next_mesh[i]] = next_vols[i]
            
            fill_dict['mesh'] = list(avg.keys())
            fill_dict['ex_volume'] = list(avg.values())
            fill_dict['time'] = mst

        # Append the new row to data_to_fill
        new_rows = pd.DataFrame(fill_dict)
        data_to_fill = pd.concat([data_to_fill, new_rows], ignore_index=True)

    res = pd.concat([data, data_to_fill], ignore_index=True).sort_values('time')
    res['time'] = res['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return res


def dataMerge(mesh_list, data_path, output_dir='./cache', data_type='m'):
    one_day_data = pd.DataFrame(columns=['time', 'mesh', 'ex_volume'])
    date = None
    output_file = None

    for data_file in os.listdir(data_path):
        full_path = os.path.join(data_path, data_file)
        
        date = data_file[7:15]
        output_file = f'{output_dir}/data_{TYPE}_{date}.csv'
        
        if os.path.exists(output_file):
            print(f'{output_file} data already exists, skip!')
            return list(pd.read_csv(output_file)['mesh'].unique())

        
        if data_file[5] != data_type.lower():
            continue

        if data_file.endswith('.csv.gz'):
            try:
                # Load national data Japan.
                df = pd.read_csv(full_path, compression='gzip')
            except Exception as e:
                print(f"{type(e).__name__}, please check {data_file} file for detail!")
                continue

            target_area = df[df['mesh'].isin(mesh_list)]
            one_day_data = pd.concat([one_day_data, target_area])
        
        else:
            print(f'Skip non-csv file: {data_file}')
            
    one_day_data = fill_missing(one_day_data, method='interpolate')
    one_day_data.to_csv(output_file, index=False)
    
    print(f'{output_file} data is extracted successfully! ')
    
    return list(one_day_data['mesh'].unique())


def gen_STID_Data(data_path, output_path, NODES):
    files = sorted(glob.glob(f'{data_path}/*_{TYPE}_*.csv'))
    
    start_date = f'{YEAR}-{MONTH}-01'
    end_date = pd.to_datetime(start_date) + pd.DateOffset(months=1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='5T')[:-1]

    stid_df = pd.DataFrame(0, index=date_range, columns=MESH_ID)
    
    for file in files:
        print(f"Processing {file}...")
        data = pd.read_csv(file)
        data.sort_values(['time', 'mesh'])
        data['time'] = pd.to_datetime(data['time'])
        
        for _, d in data.iterrows():
            mesh_id = d['mesh']
            volume = d['ex_volume']
            ts = d['time']
            stid_df.at[ts, mesh_id] = volume
    
    stid_df.rename(columns=ID_INDEX, inplace=True)
    
    stid_df.to_csv(f'{output_path}/{DATASET}_{TYPE}_{YEAR}{MONTH}.csv')
    
    return


if __name__ == "__main__":
    
    """
    Set the default Parameters.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', "--json", type=str, default=None, help="Configuration json file path. If specificated, then read parameters from json file.")

    parser.add_argument("-d", "--dataset", type=str, default="unknown")
    parser.add_argument("-i", "--input_path", type=str, default="/data/xpop_mesh4_agg")
    parser.add_argument("-o", "--output_path", type=str, default="./data/oneDayData", help="input root path of one day aggregated data.")
    parser.add_argument("-s", "--stid_path", type=str, default="./data/trainData", help="output root path of one month aggregated data.")
    parser.add_argument("-t", "--data_type", type=str, default="m", help="Data type of mydaiz data, m-->move, l-->latest.")
    parser.add_argument("-y", "--year", type=str, default="2024", help="The year of input data.")
    parser.add_argument("-m", "--month", type=str, default="05", help="The month of input data.")
    
    args = parser.parse_args()
    input_root = args.input_path
    output_root = args.output_path
    stid_path = args.stid_path
    TYPE = args.data_type
    DATASET = args.dataset
    YEAR = args.year
    MONTH = args.month

    print(f'Raw data processing starts...')
    # Generate Mehses from input bbox.
    boundary_file = os.path.join(output_root, DATASET)
    MESH_MAP = pd.read_csv(f'{boundary_file}/meshes.csv')
    target_meshes = MESH_MAP['mesh'].values
    input_path = os.path.join(os.path.join(input_root, YEAR), MONTH)
    output_path = os.path.join(os.path.join(os.path.join(output_root, DATASET), YEAR), MONTH)

    # Create the dirctory if not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Iterate data by date
    for child_dir in sorted(os.listdir(input_path)):
        data_path = os.path.join(input_path, child_dir)
        uni_mesh = dataMerge(target_meshes, data_path, output_path, TYPE)
        MESH_ID += uni_mesh
        
    print(f'Data extraction and merging is finished!')

    MESH_ID = sorted(list(set(MESH_ID)))

    print(f'{len(MESH_ID)} unique meshes are founded.')
    
    for i in range(len(target_meshes)):
        ID_INDEX[target_meshes[i]] = i

    gen_STID_Data(output_path, stid_path, len(MESH_ID))
    
    print(f'STID data is successfully saved under {stid_path}!')
