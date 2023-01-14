# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_imagenet import resnet18
from .quan_resnet_imagenet import resnet18_quan, resnet34_quan, vit_image_classification
from .quan_alexnet_imagenet import alexnet_quan
from .quan_resnet50_imagenet import resnet50
from .vit_module.vision_transformer import vit_base_patch16_224
from .quan_resnext_imagenet import resnext50
############## ResNet for CIFAR-10 ###########
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet20
from .quan_resnet_cifar import resnet20_quan
from .bin_resnet_cifar import resnet20_bin, resnet50_bin

############## VGG for CIFAR #############

from .vanilla_models.vanilla_vgg_cifar import vgg11_bn, vgg11
from .quan_vgg_cifar import vgg11_bn_quan, vgg11_quan
from .bin_vgg_cifar import vgg11_bn_bin


############# Mobilenet for ImageNet #######
from .vanilla_models.vanilla_mobilenet_imagenet import mobilenet_v2

from .quan_mobilenet_imagenet import mobilenet_v2_quan