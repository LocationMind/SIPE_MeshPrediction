"""
Mesh Generation Function
Metrics Function
"""

from jpmesh import Angle, Coordinate, HalfMesh
import numpy as np
import pandas as pd
import gpustat
import torch
import os



def parse_bbox(bbox_str):
    try:
        left, bottom, right, up = map(float, bbox_str.split(','))
        return (left, bottom, right, up)
    except ValueError:
        raise argparse.ArgumentTypeError("BBox must be in the format 'left, bottom, right, up' with numbers")



def coor2mesh(bbox, delx=0.001, dely=0.001):
    """
    bbox: boundary box for the area you want to obtain mesh IDs.
        -->  bbox (max_lon, max_lat, min_lon, min_lat)
    
    delx, dely: iteration step size, in degree.
    """
    
    max_lon, max_lat, min_lon, min_lat = bbox
    tx, ty = min_lat, min_lon
    result = set()
    
    while tx <= max_lat:
        ty = min_lon
        
        while ty <= max_lon:
            coordinate = Coordinate(lon=Angle.from_degree(tx), lat=Angle.from_degree(ty))
            result.add(HalfMesh.from_coordinate(coordinate).code)
            ty += dely
        tx += delx
    
    return sorted(np.array(list(result), dtype=np.int32))


def meshGenerator(bbox):
    max_lat = max(bbox[0], bbox[2])
    max_lon = max(bbox[1], bbox[3])
    min_lat = min(bbox[0], bbox[2])
    min_lon = min(bbox[1], bbox[3])

    bbox = (max_lat, max_lon,  min_lat, min_lon)
    target_grids = coor2mesh(bbox)

    print(f'{len(target_grids)} target meshes are found in given bbox!')
    
    return target_grids


def masked_mse(preds, labels, null_val=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    

def masked_rmse(preds, labels, null_val=0):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    

def masked_mae(preds, labels, null_val=0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=0):
    if torch.isnan(torch.tensor(null_val)):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels > null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    
    # Avoid nan error.
    labels_safe = torch.where(labels == 0, torch.ones_like(labels), labels)
    loss = torch.abs(preds-labels)/labels_safe
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

class myLoss:
    def __init__(self, name):
        if name.upper() not in ["MAE", "MSE", "RMSE", "MAPE", "HYBRID"]:
            raise NotImplementedError
        else:
            self.name = name.upper()
            
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=1e-3, constant=(0.05, 1)):
        if self.name == "MAE":
            return masked_mae(preds, labels, null_val)
        elif self.name == "MSE":
            return masked_mse(preds, labels, null_val)
        elif self.name == "RMSE":
            return masked_rmse(preds, labels, null_val)
        elif self.name == "MAPE":
            return masked_mape(preds, labels, null_val)
        elif self.name == "HYBRID":
            a, b = constant
            return a * masked_mae(preds, labels, null_val) + b * masked_mape(preds, labels, null_val)
        

def getMetric(y_pred, y_true, null_val=0):
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
        
    rmse = masked_rmse(y_pred, y_true, null_val)
    mae = masked_mae(y_pred, y_true, null_val)
    mape = masked_mape(y_pred, y_true, null_val)

    return rmse, mae, mape


def select_gpu():
    # if no gpu is available, return CPU.
    if not torch.cuda.is_available():
        return 'cpu'
        
    mem = []
    # list(set(X)) is done to shuffle the array
    gpus = list(set(range(torch.cuda.device_count()))) 
    
    for i in gpus:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        mem.append(gpu_stats.jsonify()["gpus"][i]["memory.used"])
        
    return str(gpus[np.argmin(mem)])