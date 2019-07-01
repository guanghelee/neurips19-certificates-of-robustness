# MNIST experiment:

This repository is for the paper

 * "[A Stratified Approach to Robustness for Randomly Smoothed Classifiers](https://arxiv.org/pdf/1906.04948.pdf)" by [Guang-He Lee](https://people.csail.mit.edu/guanghe/), [Yang Yuan](http://www.callowbird.com), [Shiyu Chang](http://people.csail.mit.edu/chang87/), and [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/) in arxiv.

## Outline 

### Package version 

 * PyTorch0.4.1
 * python3.6.1

### Disclaimer

 * The codes in the main/ directory are heavily adapted from the repo: https://github.com/locuslab/smoothing

### The repo structure

 * [compute_rho/](compute_rho) contains the code for computing &rho;<sup>-1</sup><sub>r</sub>(0.5) in Alg. 1.
 * [main/](main) contains the code for training and certifying networks.

## Training and certifying a network smoothed by the discrete perturbation

 * We have a script that automatically runs the training and certifying procedure. Please run the following command (it may take about 2 hours for certifying)

```
cd main/
sh scripts/cnn_bernoulli.sh
```

 * The results with &alpha; = 0.8 will be in the foler models/mnist/cnn/alpha_0.80/

 * To see the results in ACC@r, please run 

```
python code/compute_l0.py models/mnist/cnn/alpha_0.80/test.txt thresholds/mnist/0.8.txt 
```

 * To run the experiments with other &alpha; values, please revise the script, and run the above command with different thresholds (&rho;<sup>-1</sup><sub>r</sub>(0.5)). Note that we have included several &rho;<sup>-1</sup><sub>r</sub>(0.5) for different &alpha; values in thresholds/mnist/

 * Please see below for computing &rho;<sup>-1</sup><sub>r</sub>(0.5) for other &alpha; values.

## Computing a customized / Reproducing &rho;<sup>-1</sup><sub>r</sub>(0.5)

### Computing the cardinality of each region |L(u, v; r)|

 * The command will compute the cardinalities for r in {0,1,...,20}

```
cd compute_rho/
python count_mnist.py
``` 

### Computing &rho;<sup>-1</sup><sub>r</sub>(0.5)

 * When &alpha; = a / 100 (a is an integer), we run the following command to compute the corresponding &rho;. We use a = 80 as an example:

```
python threshold_mnist.py --a 80
```

 * The results will be in 

```
thresholds/mnist/0.8.txt
```
