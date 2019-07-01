# evaluate a smoothed classifier on a dataset
import argparse
import os
#import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from discrete_core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture
from copy import deepcopy
from tqdm import trange

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("flip_alpha", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["eval_train", "valid", "test"], default="test", help="valid or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

K_dict = {'cifar10': 255, 'imagenet': 255, 'mnist': 1}
args.K = K_dict[args.dataset]

args.beta = (1 - args.flip_alpha) / args.K
ratio = args.beta / args.flip_alpha
args.calibrated_alpha = args.flip_alpha - args.beta
# args.calibrated_alpha = (1 - ratio) * args.alpha

print('N / bz', args.N / args.batch)

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    print(checkpoint['epoch'])
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.calibrated_alpha, args.K)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in trange(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        # print(x.shape)
        # exit(0)

        robust_correct = 1
        before_time = time()
        for idx in range(28 * 28):
            idx_i = idx // 28
            idx_j = idx % 28

            distorted_x = deepcopy(x)
            distorted_x[0, idx_i, idx_j] = 1 - distorted_x[0, idx_i, idx_j]
            # predict of g around x
            distorted_x = distorted_x.cuda()
            
            prediction = smoothed_classifier.predict(distorted_x, args.N, args.alpha, args.batch)
            after_time = time()
            correct = int(prediction == label)
            if correct == 0:
                robust_correct = 0
                break

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}/{}\t{}\t{}\t{}".format(
            i, len(dataset), label, robust_correct, time_elapsed), file=f, flush=True)

    f.close()
