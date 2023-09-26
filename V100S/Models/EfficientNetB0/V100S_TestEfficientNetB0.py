import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models import efficientnet_b0
from torch.profiler import profile, record_function, ProfilerActivity
from DepthwiseLayer import OptimizedDepthwiseLayer
import time
import csv
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ===================================================================================================
# Turn on the GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ===================================================================================================
# Define Hyperparameters for training
batch_size = 128
EPOCHS = 300

# ===================================================================================================
# data augmentation
train_transform = transforms.Compose([transforms.RandomRotation(5),
                                      transforms.RandomHorizontalFlip(0.3),
                                      transforms.RandomVerticalFlip(0.05),
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                     ])

test_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                     ])

# ===================================================================================================
# load data
train_set = datasets.CIFAR10(root = '../../data', train = True, download = True, transform = train_transform)
train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, drop_last = True)

test_set = datasets.CIFAR10(root = '../../data', train = False, download = True, transform = test_transform)
test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 4, drop_last = True)

# Check data dimension
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# ===================================================================================================
# Creat original efficientnet b0
originalModel = efficientnet_b0(weights=None)
originalModel.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=10, bias=True)
)
originalModel.cuda()

# Loss function and optimizer for original model
loss_fn_Original= nn.CrossEntropyLoss()

# ===================================================================================================
# Create modified efficientnet b0
modifiedModel = efficientnet_b0(weights=None)
modifiedModel.features[1][0].block[0][0] = OptimizedDepthwiseLayer(inputChannel = 32, outputChannel = 32, filterHeight = 3, stride = 1)
modifiedModel.features[2][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 96, outputChannel = 96, filterHeight = 3, stride = 2)
modifiedModel.features[2][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 3, stride = 1)
modifiedModel.features[3][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 144, outputChannel = 144, filterHeight = 5, stride = 2)
modifiedModel.features[3][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 240, outputChannel = 240, filterHeight = 5, stride = 1)
modifiedModel.features[4][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 240, outputChannel = 240, filterHeight = 3, stride = 2)
modifiedModel.features[4][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 3, stride = 1)
modifiedModel.features[4][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 3, stride = 1)
modifiedModel.features[5][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 480, outputChannel = 480, filterHeight = 5, stride = 1)
modifiedModel.features[5][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 1)
modifiedModel.features[5][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 1)
modifiedModel.features[6][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 672, outputChannel = 672, filterHeight = 5, stride = 2)
modifiedModel.features[6][1].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
modifiedModel.features[6][2].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
modifiedModel.features[6][3].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 5, stride = 1)
modifiedModel.features[7][0].block[1][0] = OptimizedDepthwiseLayer(inputChannel = 1152, outputChannel = 1152, filterHeight = 3, stride = 1)
modifiedModel.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=10, bias=True)
)
modifiedModel.cuda()

# Loss function and optimizer for modified model
loss_fn_Modified = nn.CrossEntropyLoss()

# ===================================================================================================
# Warm Up
print("Started Warm Up")
for X, y in test_dataloader:
    X, y = X.to(device), y.to(device)

    predOriginal = originalModel(X)
    lossOrigianl = loss_fn_Original(predOriginal, y)
    lossOrigianl.backward()
    
    predModified = modifiedModel(X)
    lossModified = loss_fn_Modified(predModified, y)
    lossModified.backward()
    
    break
print("Finished Warm Up")


# ===================================================================================================
# Measure Training Performance

# Get the size of test dataset and batch
testDataSize = len(test_dataloader.dataset)

# Modified Model
print("Modified Model Training Started")
forwardTimeModified, backwardTimeModified = 0, 0
modelStart = time.time()
for epoch in range(EPOCHS):
    print(f"    Modified Model Epoch {epoch} started training!")
    currEpochForwardTime, currEpochBackwardTime = 0, 0
    
    if epoch == 0:
        optimizer = optim.SGD(modifiedModel.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
    elif epoch == 150:
        optimizer = optim.SGD(modifiedModel.parameters(), lr = 1e-2, weight_decay = 4e-5, momentum = 0.9)
    elif epoch == 225:
        optimizer = optim.SGD(modifiedModel.parameters(), lr = 1e-3, weight_decay = 4e-5, momentum = 0.9)
        
    # Train the model
    modifiedModel.train()
    for (X, y) in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward Pass Prediction
        forwardStart = time.time()
        pred = modifiedModel(X)
        currEpochForwardTime += time.time() - forwardStart
        forwardTimeModified += time.time() - forwardStart
        
        # Compute Prediction error
        loss = loss_fn_Modified(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        backwardStart = time.time()
        loss.backward()
        currEpochBackwardTime += time.time() - backwardStart
        backwardTimeModified += time.time() - backwardStart
        
        optimizer.step()
        
    print('        Forward Pass Time: {:.3f} s'.format(currEpochForwardTime))
    print('        Backward Pass Time: {:.3f} s'.format(currEpochBackwardTime))
    
    # Check current epoch training result
    correct = 0
    modifiedModel.eval()
    with torch.no_grad():
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = modifiedModel(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= testDataSize
    print(f"        Epoch {epoch} Accuracy: {(100*correct):>0.1f}%")
    print(f"    Modified Model Epoch {epoch} finished training!")

runTimeModified = time.time() - modelStart
print("Modified Model Training Ended")
print('Total Run time of Modified model: {:.3f} s'.format(runTimeModified))
print('Total Forward Pass time of Modified model: {:.3f} s'.format(forwardTimeModified))
print('Total Backward Pass time of Modified model: {:.3f} s'.format(backwardTimeModified))
print("Data Transfer Time to GPU is not measured")

# Original Model
print("Original Model Training Started")
forwardTimeOriginal, backwardTimeOriginal = 0, 0
modelStart = time.time()
for epoch in range(EPOCHS):
    print(f"    Original Model Epoch {epoch} started training!")
    currEpochForwardTime, currEpochBackwardTime = 0, 0
    
    if epoch == 0:
        optimizer = optim.SGD(originalModel.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
    elif epoch == 150:
        optimizer = optim.SGD(originalModel.parameters(), lr = 1e-2, weight_decay = 4e-5, momentum = 0.9)
    elif epoch == 225:
        optimizer = optim.SGD(originalModel.parameters(), lr = 1e-3, weight_decay = 4e-5, momentum = 0.9)
        
    # Train the model
    originalModel.train()
    for (X, y) in train_dataloader:
        X, y = X.to(device), y.to(device)
        
        # Forward Pass Prediction
        forwardStart = time.time()
        pred = originalModel(X)
        currEpochForwardTime += time.time() - forwardStart
        forwardTimeOriginal += time.time() - forwardStart
        
        # Compute Prediction error
        loss = loss_fn_Original(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        backwardStart = time.time()
        loss.backward()
        currEpochBackwardTime += time.time() - backwardStart
        backwardTimeOriginal += time.time() - backwardStart
        
        optimizer.step()
        
    print('        Forward Pass Time: {:.3f} s'.format(currEpochForwardTime))
    print('        Backward Pass Time: {:.3f} s'.format(currEpochBackwardTime))
    
    # Check current epoch training result
    correct = 0
    originalModel.eval()
    with torch.no_grad():
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = originalModel(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        correct /= testDataSize
        print(f"        Epoch {epoch} Accuracy of Original Model: {(100*correct):>0.1f}%")
    print(f"    Original Model Epoch {epoch} finished training!")

runTimeOriginal = time.time() - modelStart
print("Original Model Training Ended")
print('Total Run time of Original model: {:.3f} s'.format(runTimeOriginal))
print('Total Forward Pass time of Original model: {:.3f} s'.format(forwardTimeOriginal))
print('Total Backward Pass time of Original model: {:.3f} s'.format(backwardTimeOriginal))
print("Data Transfer Time to GPU is not measured")

print("Forward speed up : {:.3f} %".format(100 * (forwardTimeOriginal - forwardTimeModified) / forwardTimeOriginal))
print("Train time speed up : {:.3f} %".format(100 * (runTimeOriginal - runTimeModified) / runTimeOriginal))

# ===================================================================================================
# Measure Inference Performance of Models

# Modified Model
correct = 0
forwardTimeModified = 0
modelStart = time.time()
modifiedModel.eval()
with torch.no_grad():
    for (X, y) in test_dataloader:
        X, y = X.to(device), y.to(device)
        
        forwardStart = time.time()
        pred = modifiedModel(X)
        forwardTimeModified += time.time() - forwardStart
    
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
runTimeModified = time.time() - modelStart
print('Modified Model Inference Total time: {:.3f} s'.format(runTimeModified))
print('Modified Model Inference Forward Pass time: {:.3f} s'.format(forwardTimeModified))
correct /= testDataSize
print(f"Test Accuracy of Modified Model: {(100*correct):>0.1f}%")

# Original Model
correct = 0
forwardTimeOriginal = 0
modelStart = time.time()
originalModel.eval()
with torch.no_grad():
    for (X, y) in test_dataloader:
        X, y = X.to(device), y.to(device)
        
        forwardStart = time.time()
        pred = originalModel(X)
        forwardTimeOriginal += time.time() - forwardStart

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

runTimeOriginal = time.time() - modelStart
print('Original Model Inference Total time: {:.3f} s'.format(runTimeOriginal))
print('Original Model Inference Forward Pass time: {:.3f} s'.format(forwardTimeOriginal))
correct /= testDataSize
print(f"Test Accuracy of Original Model: {(100*correct):>0.1f}%")

print("Forward speed up : {:.3f} %".format(100 * (forwardTimeOriginal - forwardTimeModified) / forwardTimeOriginal))
print("Test time speed up : {:.3f} %".format(100 * (runTimeOriginal - runTimeModified) / runTimeOriginal))
