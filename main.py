from data import data_loader
from config import args
from utils import transform,valid_transform,plot
from torch.utils.data import DataLoader
import torch
from models import resnet, CNN
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import train


path= args.path
batch_size= args.bs

train_loader,val_loader=data_loader.get_train_valid_loader(path,batch_size,random_seed=4,transform=transform,valid_transform=valid_transform)

print(val_loader)
### Model
import argparse
parser= argparse.ArgumentParser()
parser.add_argument('-m','--model_name',help='this is the name of the model',type=str,required=True)

parser.add_argument('-n','--num_epochs',help='this is the number of the epochs',type=int,required=True)
mains_args= vars(parser.parse_args())
num_epochs= mains_args['num_epochs']
if mains_args["model_name"].lower()=='resnet':
    model=resnet()

criterion= nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
model_trained, percent, val_loss, val_acc, train_loss, train_acc= train.train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)
plot(train_loss,val_loss)