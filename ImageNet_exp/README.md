# ImageNet experiment:

## Outline 

### Package version 

 * PyTorch0.4.1
 * python3.6.1

### Disclaimer

 * The codes in the main/ directory are heavily adapted from the [repo](https://github.com/locuslab/smoothing)
 * The folder is not quite polished compared to the MNIST experiment. Please let me know if you see any problems.

### Outline

 * Pre-computed &rho;<sup>-1</sup><sub>r</sub>(0.5) are available in [main/thresholds/imagenet/](main/thresholds/imagenet/).
 * Trained ResNet50 models are availabble in [Google drive](https://drive.google.com/file/d/19p6uN4-37HzF1dD8whXAkjuLcCzMDKOM/view?usp=sharing).
 * To test the model, please first download the ResNet50 models from the Google drive, and put them in [main/models/imagenet/resnet50/](main/models/imagenet/resnet50/). Then you can run the scripts in [main/eval_scripts/imagenet/](main/eval_scripts/imagenet/).
 * To train the model, please check the example scripts in [main/scripts/imagenet/](main/scripts/imagenet/). 
 * To compute your own &rho;<sup>-1</sup><sub>r</sub>(0.5), please check the [folder](compute_rho/).

