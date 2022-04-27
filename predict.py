import torch
import warnings
from PIL import Image
from torchvision import transforms
import models 
from argparse import ArgumentParser
from utils import test_transform
from data import data_loader
from config import args

test_loader= data_loader.get_test_loader(args.path,args.bs,test_transform,shuffle=True,num_workers=4,
                    pin_memory=False)
                    
def predict(device,model,verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')
    model_path = './model/model_ok.pth'
    try:
        checks_if_model_is_loaded = type(model)
    except:
        model = models.resnet50()
    model.eval()
    correct = 0

    for idx, (imgs, targets) in enumerate(test_loader):
      imgs = imgs.to(args.device)
      targets = targets.to(device)
      output = model(imgs)
      pred = output.argmax()

      correct += pred.eq(targets).sum()

    print(correct)

parser= ArgumentParser()
parser.add_argument('-m', '--image_path', help= 'upload the image',required= True)
main_args= vars(parser.parse_args())
image_path= main_args['image_path']

