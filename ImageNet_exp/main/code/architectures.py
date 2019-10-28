import torch
from torchvision.models.resnet import resnet50, resnet18
import torch.backends.cudnn as cudnn
from archs.cifar_resnet import resnet as resnet_cifar
from archs.mnist_fc import FC_Net
from archs.mnist_cnn import CNN_Net
from datasets import get_normalize_layer
from torch.nn.functional import interpolate

# resnet50 - the classic ResNet-50, sized for ImageNet
ARCHITECTURES = ["resnet50"]

def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    normalize_layer = get_normalize_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
