# MNIST experiment:

This repository is for the paper

 * "[A Stratified Approach to Robustness for Randomly Smoothed Classifiers](https://arxiv.org/pdf/1906.04948.pdf)" by [Guang-He Lee](https://people.csail.mit.edu/guanghe/), [Yang Yuan](http://www.callowbird.com), [Shiyu Chang](http://people.csail.mit.edu/chang87/), and [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/) in arxiv.
 * [Project page](http://people.csail.mit.edu/guanghe/locally_linear)

## Package version 

 * PyTorch0.4.1
 * python3.6.1

## Release Schedule

 * This is an initial release for computing \rho^{-1}(0.5) in the MNIST experiment by an email request. The rest of the experiments will be released gradually. If you need some help on some other experiments, please let me know (guanghe@mit.edu) and I will try to see whether I can release it soon. 

## Computing the cardinality of each region |L(u, v; r)|

 * The command will compute the cardinalities for r in [20]

```
cd compute_rho/
python count_mnist.py
``` 

## Computing \rho^{-1}(0.5)

 * When alpha = a / 100 (a is an integer), we run the following command to compute the corresponding \rho. We use a = 80 as an example:

```
python threshold_mnist.py --a 80
```

 * The results will be  in 

```
thresholds/mnist/0.8.txt
```
