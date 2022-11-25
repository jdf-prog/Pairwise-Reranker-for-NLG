import random
import os
import numpy as np
import torch
import argparse
from typing import List, Union
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def empty2None(x):
    if x == '':
        return None
    elif isinstance(x, str):
        return x
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def empty2zero(x):
    if x == '':
        return 0
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')
