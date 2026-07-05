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

from torchvision.utils import save_image, make_grid

from data.cifar10 import setup_dataloaders, train_transforms, test_transforms, unnormalizer
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
LR         = 0.005
EPOCHS     = 3
LR_STEPS   = [35, 45]

# Backdoor parameters
TARGET_CLASS = 3
POISON_RATIO = 0.05
STRENGTH     = 1.0
IMAGE_SIZE   = 32

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
        TARGET_CLASS, POISON_RATIO, STRENGTH, IMAGE_SIZE, 
        DEVICE, USE_IPEX
    )

    # Sets optimization
    optimizer = optim.SGD(model.parameters(), lr = LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(reduction = "sum")

    # Optimization if IPEX is available
    if USE_IPEX:
        model, optimizer = ipex.optimize(model, optimizer = optimizer)

    #################
    # Training Step #
    #################
    
    print("=== Starting Training ===")

    image_saved_checkpoint = False

    for epoch in range(EPOCHS):

        # Updates learning rate
        if epoch in LR_STEPS:
            print(f"[INFO] Update learning rate: {LR} -> {LR / 10}")
            LR /= 10
            for group in optimizer.param_groups:
                group['lr'] /= 10

        # Sets performance metric placeholders
        running_acc  = 0
        running_asr  = 0
        running_loss = 0
        count_clean  = 0
        count_poison = 0
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Loads data
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data         = train_transforms(data)
            if not image_saved_checkpoint:
                saved_benign_data = data.detach().cpu()
            # Injects backdoor
            data, labels, mask = backdoor.inject_backdoor(data, labels)
            # Saves example images
            if not image_saved_checkpoint and mask.sum() > 0:
                saved_benign_data = unnormalizer(saved_benign_data[mask.detach().cpu()])
                saved_poison_data = unnormalizer(data[mask]).detach().cpu()
                difference_data   = saved_poison_data - saved_benign_data
                difference_data  -= difference_data.min()
                difference_data  /= difference_data.max()
                to_save           = \
                    torch.cat([saved_benign_data[:3], saved_poison_data[:3], difference_data[:3]], dim = 0)
                save_image(make_grid(to_save, nrow = 3),"checkpoints/example_poisons.png")
                image_saved_checkpoint = True  
                del saved_benign_data
                del saved_poison_data
                del difference_data           
            # Resets optimizer
            optimizer.zero_grad()
            # Forward pass
            predictions = model(data)
            # Computes loss and performs backward pass
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            # Records running metrics
            running_acc  += (torch.argmax(predictions[~mask], dim=-1) == labels[~mask]).detach().cpu().sum() / len(data[~mask])
            running_asr  += (torch.argmax(predictions[mask], dim=-1) == labels[mask]).detach().cpu().sum() / len(data[mask])
            count_clean  += (~mask).sum().detach().item()
            count_poison += (mask).sum().detach().item()
            running_loss += loss.detach().cpu() / len(data)
        
        # Reports epoch performance
        running_acc  /= len(train_loader)
        running_asr  /= len(train_loader)
        running_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}] Loss {running_loss:.4f}, Training accuracy {running_acc*100:.2f}%, Training ASR: {running_asr*100:.2f}%")
    
    ###################
    # Evaluation Step #
    ###################

    model.eval()
    accuracy            = 0
    attack_success_rate = 0

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
        for data, labels in test_loader:
            # Loads data
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            data         = test_transforms(data)
            # Injects backdoor
            data, labels, _ = backdoor.inject_backdoor(data, labels, test = True)
            # Forward pass
            predictions = model(data)
            # Records accuracy
            attack_success_rate += (torch.argmax(predictions, dim=-1) == labels).detach().cpu().sum() / len(data)
        attack_success_rate /= len(test_loader)

    # Reports  test-time performance
    print(f"[Test phase] Accuracy            {accuracy*100:.2f}%")
    print(f"[Test phase] Attack Success Rate {attack_success_rate*100:.2f}%")

    # Saves weights
    torch.save(model.state_dict(), "checkpoints/backdoored_model.pth")