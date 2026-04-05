import torch
import numpy as np
import random
import os

# Use this script if you want to reproduce results of show*.py
# Call it right in front of the rest of your code.
# Neural nets are not deterministic by default

def set_determinism(seed=42):
    """Locks all random number generators for reproducible training."""
    # Python & Standard Library
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch Core
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU
        
        # 4. The CUDA Execution Engine (The bottleneck)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Call this immediately at the start of your script
set_determinism(1)