import time
import os
import datetime
import numpy as np
import torch

# This class holds training states to recover if somehow training stops
# Methods: Save, Load
# Save: model parameters, with epoch number
#       Losses
#       optimizer state
class checkpoint():
    def __init__(self, args):
        self.args = args
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('.', 'experiment', args.save)
        else:
            self.dir = os.path.join('.', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('loss_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('config.txt')) else 'w'
        # self.log_file = open(self.get_path('log.txt'), open_type)
        self.log_model_architecture = 'log_model_architecture.txt'
        self.log_file_name = 'log_train.txt'
        self.log_valid_file_name = 'log_valid.txt'
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.ok = True

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, method, epoch, is_best=False):
        method.model.save(self.get_path('model'), epoch, is_best=is_best)
        method.loss.save(self.dir)
        # method.loss.plot_loss(self.dir, epoch)
        method.optimizer.save(self.dir)

    def load(self, method, is_best=True):
        method.model.load(apath=self.get_path('model'), resume=-1)
        method.loss.load(self.dir)
        method.optimizer.load(self.dir)

    def write_log(self, log, type='train'):
        print(log)
        if type == 'train':
            open_type = 'a' if os.path.exists(self.get_path(self.log_file_name)) else 'w'
            with open(self.get_path(self.log_file_name), open_type) as f:
                f.write(log + '\n')
                f.close()
        elif type == 'validation':
            open_type = 'a' if os.path.exists(self.get_path(self.log_valid_file_name)) else 'w'
            with open(self.get_path(self.log_valid_file_name), open_type) as f:
                f.write(log + '\n')
                f.close()