"""
File to test whether Apple's Metal GPU is supported on your device
"""

import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
    print ("MPS supported")
else:
    print ("MPS device not found.")