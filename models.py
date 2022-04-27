import torch.nn as nn
import torchvision
import torch

def resnet():
    model= torchvision.models.resnet50(pretrained= True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 10, bias = True)
        )
    return model
