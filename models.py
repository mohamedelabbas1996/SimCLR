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



def resnet_simCLR():

    model= torchvision.models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = True
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 128))    

  
    return model


def resnet_simCLR_classification(path):

    model= resnet_simCLR()
    model.load_state_dict(torch.load(path))

    model.fc = nn.Linear(2048, 10)

    ct = 0
    for child in model.children():
        ct += 1
        if ct < 10:
           for param in child.parameters():
               param.requires_grad = False

    return model
