import os
from importlib import import_module

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()

        self.selectLoss(args)

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        self.loss_module.to(self.device)
        if args.precision == 'half': self.loss_module.half()
        if not args.device == "cpu" and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )
        cpu = True if args.device == "cpu" else False
        if args.load != '': self.load(ckp.dir, cpu=cpu)

        self.loss_types = [self.loss[i]['type'] for i in range(len(self.loss))]

    def selectLoss(self, args):
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('PFF') >=0:
                module = import_module('loss.lossOrderedPairReconstruction')
                loss_function = getattr(module, 'lossOrderedPairReconstruction')(
                    device=self.device,
                    filterSize=17 #args.pff_filter_size
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

    def definingLosses(self, args):
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if hasattr(nn, loss_type+"Loss"):
                loss_function = getattr(nn, loss_type+"Loss")
            else:
                try:
                    module = import_module("loss."+loss_type)
                    loss_function = getattr(module, "make_loss")(args, type)
                except:
                    print("Loss type ({}) is undefined!!!".format(loss_type))
                    continue

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

    def forward(self, sr, hr, lr=None,type="train"):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'] == 'PFF':
                    assert(not lr == None), "PFF needs input image!!!"
                    loss = l['function'](lr, hr, sr)
                else:
                    loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                if type == 'train':
                    self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                if type == 'train':
                    self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if type == 'train':
            if len(self.loss) > 1:
                self.log[-1, -1] += loss_sum.item()
        # @todo loss class can return multiple losses, and their types
        return losses, [self.loss[i]['type'] for i in range(len(self.loss))]

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))

        return ''.join(log)

    def plot_loss(self, apath, epoch):
        axis = np.linspace(1, epoch, epoch)
        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()

