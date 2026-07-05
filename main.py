"""
MAIN MODULE FOR CIFAR10 DATA POISONING PIPELINE USING BAIT ATTACK.
"""

###########
# IMPORTS #
###########

try:
    # https://intel.github.io/intel-extension-for-pytorch/cpu/latest/tutorials/cheat_sheet.html
    import intel_extension_for_pytorch as ipex
    USE_IPEX = True
except:
    USE_IPEX = False

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from data.cifar10 import setup_dataloaders, train_transforms, test_transforms
from backdoor.backdoor import bait_backdoor

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

print(f"Selected device: {DEVICE}")

# Training parameters
BATCH_SIZE = 256
LR         = 0.001
EPOCHS     = 100

# Backdoor parameters
TARGET_CLASS = 0
POISON_RATIO = 0.1
STRENGTH     = 1.0
PATTERN_SIZE = 16
IMAGE_SIZE   = 64

if __name__ == "__main__":

    print("=== Running preliminaries ===")

    # Loads dataloaders
    train_loader, test_loader = setup_dataloaders(BATCH_SIZE)

    # Loads model
    model    = torchvision.models.resnet18(weights = None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model    = model.to(DEVICE)

    # Loads backdoor method
    backdoor = bait_backdoor(
        TARGET_CLASS, POISON_RATIO, STRENGTH, PATTERN_SIZE, IMAGE_SIZE, 
        DEVICE, USE_IPEX
    )

    # Sets optimization
    optimizer = optim.SGD(model.parameters(), lr = LR)
    criterion = nn.CrossEntropyLoss(reduction = "sum")

    # Optimization if IPEX is available
    if USE_IPEX:
        model, optimizer = ipex.optimize(model, optimizer = optimizer)

    #################
    # Training Step #
    #################
    
    print("=== Starting Training ===")

    for epoch in range(EPOCHS):

        # Sets performance metric placeholders
        running_acc  = 0
        # running_asr  = # NOTE: TBD
        running_loss = 0

        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Loads data
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data         = train_transforms(data)
            # Injects backdoor
            # Resets optimizer
            optimizer.zero_grad()
            # Forward pass
            predictions = model(data)
            # Computes loss and performs backward pass
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            # Records running metrics
            running_acc  += (torch.argmax(predictions, dim=-1) == labels).detach().cpu().sum() / len(data)
            # running_asr  += # NOTE: TBD
            running_loss += loss.detach().cpu() / len(data)
        
        # Reports epoch performance
        running_acc  /= len(train_loader)
        running_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}] Loss {running_loss:.2f}, Training accuracy {running_acc*100:.2f}%")
    
    ###################
    # Evaluation Step #
    ###################

    model.eval()
    accuracy            = 0
    # attack_success_rate = # NOTE: TBD

    with torch.no_grad():

        # Computes clean accuracy
        for data, labels in test_loader:
            # Loads data
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data         = test_transforms(data)
            # Forward pass
            predictions = model(data)
            # Records accuracy
            accuracy += (torch.argmax(predictions, dim=-1) == labels).detach().cpu().sum() / len(data)
        accuracy /= len(test_loader)

        # Computes attack success rate
        # NOTE: TBD

    # Reports  test-time performance
    print(f"[Test phase] Accuracy            {accuracy*100:.2f}%")
    # print(f"[Test phase] Attack Success Rate {attack_success_rate*100:.2f}%") # NOTE: TBD