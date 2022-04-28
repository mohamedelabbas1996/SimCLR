# simCLR Implementation using Pytorch 

In this work we attempt to implement the pipeline detailed in the [simCLR_paper](https://arxiv.org/abs/2002.05709), in our implementation we are using ResNet-50 backbone and CIFAR10 training with ImageNet Initialization.


The project is splitted into two parts:

* Self-Supervised learning followed by finetuning for classification.
* Supervised learning using similar settings for benchmarking


<!-- <br> -->

## Default Settings 

* Optimizer Choice:
  1. [LARS optimizer](https://github.com/kakaobrain/torchlars):
     * lr = 0.075 * sqrt(batch_size)
     * eps=1e-8
     * trust_coef=0.001
  2. [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html):
     * lr = 0.3 * batch_size/256
     * Fine-tuning : lr = 0.01



## Run the project 
In this project, we have two steps: training and predicting. In the predict step, you can upload any image from your laptop and predict it. Let's show you how to run the project.

If you do not have venv package, please refer to this [link](https://linuxize.com/post/how-to-create-python-virtual-environments-on-ubuntu-18-04/)
</br>

### Create virtual environment ###

```
$ python3 -m venv ENV_NAME
```
### Activate your environment ###

```
$ source ENV_NAME/bin/activate
```

### Requirement installations ###
To run this, make sure to install all the requirements by:

```
$ pip install -r requirements.txt 
```
### Training the Supervised Model ###

```
$ python3 main.py --model resnet --num_epochs
```

### Training the Self-Supervised Model with Fine tuning ###

```
$ python3 main.py --model simclr --num_epochs
```


### Make prediction #

```
$python3 predict.py --image_path "./data/Images/cat.0.jpg"
```

## Related Papers #

* <a href= 'https://arxiv.org/pdf/1512.03385.pdf'> Resnet </a>
* <a href= 'https://arxiv.org/abs/2002.05709'>  A Simple Framework for Contrastive Learning of Visual Representations</a>


# Contributors #

* [Allassan]()
* [Mohamed Elabass](https://github.com/mohammedelabbas)
* [Fanta SOUMAHORO](https://github.com/soumfatim)
* [Amel Abdelraheem](https://github.com/AmelxJamal)

