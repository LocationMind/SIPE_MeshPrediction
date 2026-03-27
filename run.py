import pandas as pd
import os
import argparse
import json
from utils import *
from traintest import *
import subprocess

def genMesh(bbox, path):
    target_mesh = meshGenerator(bbox)
    mesh_id = [ i+1 for i in range(len(target_mesh)) ]
    mesh_table = pd.DataFrame({"mesh":target_mesh, "id":mesh_id})
    mesh_table.to_csv(f"{path}/meshes.csv", index=False)
    print("Mesh Generation is Finished.")
    

def run_training_script(PARAMETERS):
    script = 'traintest.py'
    command = ['python', script] + PARAMETERS
    subprocess.run(command, capture_output=False, text=False)
    

def run_process_script(PARAMETERS):
    script = 'rawDataProcess.py'
    command = ['python', script] + PARAMETERS
    subprocess.run(command, capture_output=False, text=False)
    

def run_inference_script(PARAMETERS):
    script = 'inference.py'
    command = ['python', script] + PARAMETERS
    subprocess.run(command, capture_output=False, text=False)


def pocessed_check():
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--parameters", 
                        type=str, default='start.json', 
                        help='Initial parameters (json file) to run the scripts.')
    
    parser.add_argument("-m", "--mode", type=int, default=1, 
                        help= "4 Modes are supported.\
                        Mode 1: processing + training; \
                        Mode 2: processing; \
                        Mode 3: training;\
                        Mode 4: inferencing"
                        )
    
    parser.add_argument("-g", "--genMesh", action='store_true', 
                        help='Boolean. True means generate meshes from specified bbox parameters.')

    parser.add_argument("-t", "--test_data", type=str, default=None, 
                        help='File path of test data, if specified, then the specified will be used.')

    parser.add_argument("-d", "--dataset", type=str, default=None, help='City/Prefecture name of specified bbox.')
    parser.add_argument(
        "-b", "--bbox", 
        type=parse_bbox, 
        default=None,
        help="Bounding box in the format 'left, bottom, right, up'.")
    
    args = parser.parse_args()
    
    # Load the Default parameter
    JSON_FILE = args.parameters
    with open(JSON_FILE, 'r') as f:
        PARAS = json.load(f)
    
    if args.dataset:
        PARAS['dataset'] = args.dataset
        
    DATASET = PARAS['dataset']
    OUTPUT_PATH = PARAS['output_path']
    STID_PATH = PARAS['stid_path']
    TYPE = PARAS['data_type']
    YEAR = PARAS['year']
    MONTH = PARAS['month']
    MODE = args.mode
    
    STID_FILE = f'{STID_PATH}/{DATASET}_{TYPE}_{YEAR}{MONTH}.csv'
    
    # Read training parameters
    if MODE == 1 or MODE == 3 or MODE == 50:
        training_parameters = [
            '--dataset', f'{DATASET}_{TYPE.upper()}_{YEAR}{MONTH}',
            '--data_path', STID_FILE,
        ]
        TRAIN_PARAS = PARAS['training_parameters']
        for k, v in TRAIN_PARAS.items():
            training_parameters.append(f'--{k}')
            training_parameters.append(str(v))
            
    if MODE == 1 or MODE == 2 or MODE == 50:
        process_parameters = []
        for k, v in PARAS.items():
            if k != 'training_parameters' and k != 'inference_parameters':
                process_parameters.append(f'--{k}')
                process_parameters.append(str(v))
        
    
    if args.genMesh:
        if args.bbox is None:
            raise ValueError("Please specify the boundary box in given format!")
        else:
            genMesh(args.bbox, os.path.join(OUTPUT_PATH, DATASET))
    
    if MODE == 1:
        
        if os.path.exists(STID_FILE):
            print(f'Detected training data: test.csv already exists. \n Processing is over.')
        else:
            run_process_script(process_parameters)
        
        run_training_script(training_parameters)
        
    elif MODE == 2:
        
        if os.path.exists(STID_FILE):
            print(f'Detected training data: test.csv already exists. \n Processing is over.')
        else:
            run_process_script(process_parameters)
        
    elif MODE == 3:
        run_training_script(training_parameters)
        
    elif MODE == 4:
        # Reload the parameters, Since it maybe modified by the training script.
        with open(JSON_FILE, 'r') as f:
            PARAS = json.load(f)
            
        INFER_PARAS = PARAS['inference_parameters']
        inference_parameters = []

        if args.test_data:
            INFER_PARAS['data_path'] = args.test_data
                
        for k, v in INFER_PARAS.items():
            inference_parameters.append(f'--{k}')
            inference_parameters.append(str(v))
        
        run_inference_script(inference_parameters)
        
    elif MODE == 50:
        params_file = JSON_FILE
        if os.path.exists(STID_FILE):
            print(f'Detected training data: test.csv already exists. \n Processing is over.')
        else:
            run_process_script(params_file)
        
        run_training_script(training_parameters)

        with open(JSON_FILE, 'r') as f:
            PARAS = json.load(f)
            
        INFER_PARAS = PARAS['inference_parameters']
        inference_parameters = []

        if args.test_data:
            INFER_PARAS['data_path'] = args.test_data
                
        for k, v in INFER_PARAS.items():
            inference_parameters.append(f'--{k}')
            inference_parameters.append(str(v))
        
        run_inference_script(inference_parameters)

    else:
        print(f"Mode {args.mode} is not supported. Please choose MODE from 1 to 4.")
    
    

    
