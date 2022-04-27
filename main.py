from data import data_loader
from config import args
from utils import transform,valid_transform,plot, image_transform
from torch.utils.data import DataLoader
import torch
from models import resnet, resnet_simCLR, resnet_simCLR_classification
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import train
import  Contrastive_loss


path= args.path
batch_size= args.bs


# print(val_loader)
### Model
import argparse
parser= argparse.ArgumentParser()
parser.add_argument('-m','--model_name',help='this is the name of the model',type=str,required=True)

parser.add_argument('-n','--num_epochs',help='this is the number of the epochs',type=int,required=True)
mains_args= vars(parser.parse_args())
num_epochs= mains_args['num_epochs']

if mains_args["model_name"].lower()=='resnet':

    train_loader,val_loader=data_loader.get_train_valid_loader(path,batch_size,random_seed=4,transform=transform,valid_transform=valid_transform)

    model=resnet()
    criterion= nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model_trained, percent, val_loss, val_acc, train_loss, train_acc= train.train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)
    plot(train_loss,val_loss)

elif mains_args["model_name"].lower() == 'simclr':
    
    train_loader, val_loader, finetune_loader = data_loader.get_simCLR_loaders(path,batch_size,random_seed=4,transform=transform,valid_transform=valid_transform)

# self-supervised model

    model_simCLR = resnet_simCLR()
    criterion = Contrastive_loss()
    optimizer = torch.optim.Adam(model_simCLR.parameters(), lr=args.lr, weight_decay=args.wd)
    simCLR_trained, simCLR_train_loss= train.train_simCLR(model_simCLR, criterion, train_loader, optimizer, num_epochs, device)

# fine-tune simCLR

    path = '.pth'
    model_simCLR_classifier = resnet_simCLR_classification(path)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_simCLR_classifier.parameters(), lr=args.lr, weight_decay=args.wd)
    simCLR_clf_trained, simCLR_clf_percent, simCLR_clf_val_loss, simCLR_clf_val_acc, simCLR_clf_train_loss, simCLR_clf_train_acc = train.train(model_simCLR_classifier, criterion, finetune_loader, val_loader, optimizer, num_epochs, device)
    plot(simCLR_clf_train_loss,simCLR_clf_val_loss)

