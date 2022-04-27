import torchvision.transforms as transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),normalize])
                                       
valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def plot(train_loss,val_loss):
    plt.title("Training results: Loss")
    plt.plot(val_loss,label='val_loss')
    plt.plot(train_loss, label="train_loss")
    plt.legend()
    plt.show()