# Tight Certificates of Adversarial Robustness for Randomly Smoothed Classifiers:

This repository is for the paper

 * "[Tight Certificates of Adversarial Robustness for Randomly Smoothed Classifiers](https://arxiv.org/pdf/1906.04948.pdf)" by [Guang-He Lee](https://people.csail.mit.edu/guanghe/), [Yang Yuan](http://www.callowbird.com), [Shiyu Chang](http://people.csail.mit.edu/chang87/), and [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/) in NeurIPS 2019.
 * The old title for this paper is "A Stratified Approach to Robustness for Randomly Smoothed Classifiers"

## Experiments 

 * Please see each experiment in the corresponding directory (and the README therein).

## Release Schedule

 * The MNIST experiment has been released. 
 * The ImageNet experiment has been released. (Not carefully checked. Please let me know if you find any problem.)
 * The pre-computed &rho;<sup>-1</sup><sub>r</sub>(0.5) and trained ResNet50 models have been released for the ImageNet experiment.
 * If you want to compute your own &rho;<sup>-1</sup><sub>r</sub>(0.5), please see the examples in the MNIST or ImageNet folder.

## Citation:

If you find this repo useful for your research, please cite the paper

```
@inproceedings{lee2019tight,
  title={Tight Certificates of Adversarial Robustness for Randomly Smoothed Classifiers},
  author={Guang-He Lee and Yang Yuan and Shiyu Chang and Tommi S. Jaakkola},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
