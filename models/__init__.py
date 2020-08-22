# -*-coding:utf-8-*-
from .vgg import *
from .resnet import *
from .preresnet import *
from .densenet import *



def get_model(config):
    return globals()[config.architecture](config.num_classes)
