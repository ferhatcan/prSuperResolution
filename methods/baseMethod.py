# Author: @ferhatcan
# Date: 24/04/20

# This class should include basic capabilities of methods such as
# Training Model with desired options like fine Tuning, start from zero, pre-trained network,
# Testing Model like compare with GT(calculate performance), try with an input image etc.
# save & load network parameters - save_best, save_epoch (checkpoints), resume from saved_epoch etc..
# log training and testing results to tensorboard

import torch
import torch.nn as nn
import torchvision
import numpy as np
from decimal import Decimal

from utils.custom_optimizer import make_optimizer
from utils.timer import timer
from utils.visualization import psnr


class baseMethod():
    def __init__(self, args, loader, my_model, my_loss, ckp, log_writer):
        self.args = args
        self.log_writer = log_writer
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_val = loader.loader_val
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")

        dtype = torch.float
        tmp1 = torch.randn(128, 128, device=self.device, dtype=dtype)
        tmp2 = torch.randn(128, 128, device=self.device, dtype=dtype)
        _, self.loss_types = self.loss(tmp1, tmp2, type='validation')

    def train_batch(self, lr, hr):
        self.optimizer.zero_grad()
        sr = self.model(lr, 0)
        losses, loss_types = self.loss(sr, hr)
        loss = sum(losses)
        loss.backward()
        if self.args.gclip > 0:
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(),
                self.args.gclip
            )
        self.optimizer.step()

        return loss.item(), sr

    def test_batch(self, lr):
        return self.model(lr, 0)

    def train_epoch(self):
        self.loss.step()  # ıf defined loss function has an scheduler then before starting it should be initialized

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        # take the model train mode
        self.model.train()

        # time statistics of data loading and model training
        timer_data, timer_model = timer(), timer()
        val_loss, val_psnr = 1e8, 0

        for batch, (lr, hr) in enumerate(self.loader_train):
            lr, hr = lr.to(self.device), hr.to(self.device)
            timer_data.hold()
            timer_model.tic()

            _, _ = self.train_batch(lr, hr)

            timer_model.hold()

            if (batch + 1) % self.args.log_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            if (batch + 1) % (self.args.validate_every * len(self.loader_train)) == 0:
                curr_val_loss, curr_val_psnr, _, _ = self.evaluation(self.loader_val)
                if curr_val_loss < val_loss:
                    val_loss = curr_val_loss
                    print("Evaluation at epoch {} reached lowest validation set loss: {}"
                          .format(self.optimizer.get_last_epoch() + 1, val_loss))
                if curr_val_psnr > val_psnr:
                    val_psnr = curr_val_psnr
                    self.ckp.save(self, self.optimizer.get_last_epoch() + 1, is_best=True)

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        self.val_psnr_best = val_psnr
        self.val_loss_best = val_loss

    def evaluation(self, loader, take_log=True):
        # take the model validation mode
        self.model.eval()
        # time statistics of data loading and model training
        timer_data_eval, timer_model_eval = timer(), timer()
        total_loss = torch.zeros([len(self.loss_types)], dtype=torch.float)
        psnr_values = []

        for batch, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(self.device), hr.to(self.device)
            timer_data_eval.hold()
            timer_model_eval.tic()

            sr = self.test_batch(lr)
            losses, loss_types = self.loss(sr, hr, type='test')
            total_loss += losses
            # each batch should calculate its psnr
            sr_batch = np.array(sr).transpose(0, 2, 3, 1)
            hr_batch = np.array(hr).transpose(0, 2, 3, 1)
            for i in range(sr_batch.shape[0]):
                psnr_values.append(psnr(sr_batch[i, ...].sequeeze(), hr_batch[i, ...].sequeeze(), data_range=1))

            timer_model_eval.hold()
            timer_data_eval.tic()

        if take_log:
            self.ckp.write_log('[{}]\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                len(self.loader_val.dataset),
                total_loss / len(self.loader_val.dataset),
                sum(psnr_values) / len(psnr_values),
                timer_model_eval.release(),
                timer_data_eval.release()), type='validation')

        # take the model train mode
        self.model.train()
        # @todo there is a bu if log writed time returned is false
        return total_loss / len(self.loader_val.dataset), sum(psnr_values) / len(psnr_values), timer_model_eval.release(), timer_data_eval.release()

    def train(self):
        print("Traning starts ...")
        while not self.terminate():
            self.train_epoch()
            print("[Epoch {:4d}] Training results: \n".format(self.optimizer.get_last_epoch() + 1),
                  "average training loss: {}\t Lowest validation loss: {}\t".format(self.error_last, self.val_loss_best),
                  "best psnr in validation set: {}".format(self.val_psnr_best))

    def test(self, test_mode="dataset"):
        print("Test starts ...")
        if test_mode == "dataset":
            test_loss, test_psnr, data_time, model_time = self.evaluation(self.loader_test, take_log=False)

    def saveModel(self):
        self.ckp.save(self, self.optimizer.get_last_epoch() + 1, is_best=False)

    def loadModel(self):
        self.ckp.load(self, is_best=True) # model_latest will be reloaded

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epoch_num
