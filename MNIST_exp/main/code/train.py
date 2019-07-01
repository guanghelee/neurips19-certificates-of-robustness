# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log
from torch.distributions.bernoulli import Bernoulli
import numpy as np
from prob_utils import get_sigma

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=None, type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--alpha', default=1.0, type=float,
                    help="the probability to remain the same")
parser.add_argument('--sigma', default=0.0, type=float,
                    help="the std for Gaussian perturbation")
parser.add_argument('--perturb', default='bernoulli', type=str,
                    choices=['bernoulli', 'gaussian', 'c_gaussian'])

# for fair comparison w/ the randomized smoothing paper, use momentum w/ lr = 0.1
parser.add_argument('--optimizer', default='momentum', type=str,
                    choices=['momentum', 'nesterov', 'amsgrad'])
# for fair comparison w/ the randomized smoothing paper, do not tune on validation
# only activate it when validation set is available (e.g., mnist)
parser.add_argument('--tune', action='store_true', default=False,
                        help='tuning the iteration on validation')

args = parser.parse_args()
K_dict = {'cifar10': 255, 'imagenet': 255, 'mnist': 1}
args.K = K_dict[args.dataset]

args.beta = (1 - args.alpha) / args.K
ratio = args.beta / args.alpha
args.calibrated_alpha = args.alpha - args.beta
# args.calibrated_alpha = (1 - ratio) * args.alpha

if args.perturb == 'c_gaussian':
    args.perturb = 'gaussian'
    args.sigma = get_sigma(args.alpha)
    print('use gaussian w/ sigma =', args.sigma, 'from alpha =', args.alpha)

print('perturbation:', args.perturb)
if args.perturb == 'bernoulli':
    print('calibrated_alpha =', args.calibrated_alpha, 'K =', args.K)
elif args.perturb == 'gaussian':
    print('sigma =', args.sigma)
# 
if args.dataset == 'imagenet':
    valid = False
else:
    valid = True

def main():
    torch.manual_seed(0)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    train_loader, test_loader = get_dataset(args.dataset, 'train', args.batch, num_workers=args.workers)
    # test_dataset = get_dataset(args.dataset, 'valid')
    # pin_memory = (args.dataset == "imagenet")
    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                           num_workers=args.workers, pin_memory=pin_memory)
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
    #                          num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset)

    logfilename = os.path.join(args.outdir, 'log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = CrossEntropyLoss().cuda()
    if args.optimizer == 'momentum':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'nesterov':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer == 'amsgrad':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)

    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)

    best_acc = 0.
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        before = time.time()
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)
        test_loss, test_acc = test(test_loader, model, criterion, epoch, args)
        after = time.time()

        log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
            epoch, str(datetime.timedelta(seconds=(after - before))),
            scheduler.get_lr()[0], train_loss, train_acc, test_loss, test_acc))

        if args.tune and best_acc < test_acc:
            best_acc = test_acc
            print('saving best model...')
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.best.tar'))

        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    m = Bernoulli(torch.tensor([args.calibrated_alpha]).cuda())

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # make MNIST binary
        if args.dataset == 'mnist':
            inputs = (inputs > 0.5).type(torch.cuda.FloatTensor)

        # augment inputs with noise
        if args.perturb == 'bernoulli':
            mask = m.sample(inputs.shape).squeeze(-1)
            # make sure that the value is normalized
            rand_inputs = torch.randint_like(inputs, low=0, high=args.K+1, device='cuda') / float(args.K)
            inputs = inputs * mask + rand_inputs * (1 - mask)
        elif args.perturb == 'gaussian':
            inputs = inputs + torch.randn_like(inputs, device='cuda') * args.sigma
       
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i+1, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, epoch: int, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()
    m = Bernoulli(torch.tensor([args.calibrated_alpha]).cuda())

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # make MNIST binary
            if args.dataset == 'mnist':
                inputs = (inputs > 0.5).type(torch.cuda.FloatTensor)

            # augment inputs with noise
            if args.perturb == 'bernoulli':
                mask = m.sample(inputs.shape).squeeze(-1)
                # make sure that the value is normalized
                rand_inputs = torch.randint_like(inputs, low=0, high=args.K+1, device='cuda') / float(args.K)
                inputs = inputs * mask + rand_inputs * (1 - mask)
            elif args.perturb == 'gaussian':
                inputs = inputs + torch.randn_like(inputs, device='cuda') * args.sigma
           
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i+1, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
        print('* Epoch: [{0}] Test: \t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                epoch, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()
