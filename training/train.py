from copy import deepcopy
from utility_funcs.cfg_loader import load_yaml 
from training.train_model import train_model

def train():
    cfg = load_yaml("configs/training.yaml")
    for name in cfg["models"]:
        train_model(deepcopy(cfg), name)
    print("Done.")