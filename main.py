"""
MAIN MODULE FOR CIFAR10 DATA POISONING PIPELINE USING BAIT ATTACK.
"""

###########
# IMPORTS #
###########

import torch

####################
# GLOBAL VARIABLES #
####################

# Sets the global training device (order: Nvidia > Intel Arc > CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

if __name__ == "__main__":

    