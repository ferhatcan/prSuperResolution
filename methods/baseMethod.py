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

        return losses, sr

    def test_batch(self, lr, interpolate=False):
        if interpolate:
            result = torch.nn.functional.interpolate(lr, scale_factor=self.scale, mode='bicubic')
        else:
            result = self.model(lr, 0)
        return result

    def train_epoch(self):
        self.loss.step()  # ıf defined loss function has an scheduler then before starting it should be initialized

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        # TensorBoard save samples lr-hr pairs
        lr_batch, hr_batch = next(iter(self.loader_train))
        lr_batch, hr_batch = lr_batch.to(self.device), hr_batch.to(self.device)
        sr_batch = self.model(lr_batch, 0)
        lr_img_grid = torchvision.utils.make_grid(lr_batch, nrow=lr_batch.shape[0])
        hr_sr_img_grid = torchvision.utils.make_grid(torch.cat((hr_batch, sr_batch),0), nrow=hr_batch.shape[0])
        self.log_writer.add_image('LR-images', lr_img_grid, self.optimizer.get_last_epoch() + 1)
        self.log_writer.add_image('SR-HR image pairs', hr_sr_img_grid, self.optimizer.get_last_epoch() + 1)

        # take the model train mode
        self.model.train()

        # time statistics of data loading and model training
        timer_data, timer_model = timer(), timer()
        total_losses = [0] * len(self.loss_types)

        for batch, (lr, hr) in enumerate(self.loader_train):
            lr, hr = lr.to(self.device), hr.to(self.device)
            timer_data.hold()
            timer_model.tic()

            losses, _ = self.train_batch(lr, hr)
            total_losses = [total_losses[i] + (loss.item()) for i, loss in enumerate(losses)]
            total_losses.append(sum(total_losses))

            timer_model.hold()

            if (batch + 1) % self.args.log_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
                for i, log_type in enumerate(self.loss_types):
                    self.log_writer.add_scalar("train_loss/" + log_type, total_losses[i] / (batch + 1),
                                       (self.optimizer.get_last_epoch() + 1) * len(self.loader_train) + (batch + 1))
                total_losses = [0] * len(self.loss_types)

            if (batch + 1) % int(self.args.validate_every * len(self.loader_train)) == 0:
                curr_val_loss, curr_val_psnr, _, _ = self.evaluation(self.loader_val)
                effective_loss = curr_val_loss[-1] / len(self.loader_val)
                if effective_loss < self.val_loss:
                    self.val_loss = effective_loss
                    print("Evaluation at epoch {} reached lowest validation set effective loss: {:.4f}"
                          .format(self.optimizer.get_last_epoch() + 1, self.val_loss))
                if curr_val_psnr > self.val_psnr:
                    self.val_psnr = curr_val_psnr
                    self.ckp.save(self, self.optimizer.get_last_epoch() + 1, is_best=True)
                for i, log_type in enumerate(self.loss_types):
                    self.log_writer.add_scalar("validation_loss/" + log_type, curr_val_loss[i],
                                       (self.optimizer.get_last_epoch() + 1) * len(self.loader_train) + (batch + 1))
                self.log_writer.add_scalar("validation_psnr", curr_val_psnr,
                                           (self.optimizer.get_last_epoch() + 1) * len(self.loader_train) + (batch + 1))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        self.val_psnr_best = self.val_psnr
        self.val_loss_best = self.val_loss

    def evaluation(self, loader, take_log=True, evaluate_bicubic=False):
        # take the model validation mode
        self.model.eval()
        # time statistics of data loading and model training
        timer_data_eval, timer_model_eval = timer(), timer()
        total_losses = [0] * len(self.loss_types)
        psnr_values = []
        batch_counter = 0

        for batch, (lr, hr) in enumerate(loader):
            batch_counter += 1

            lr, hr = lr.to(self.device), hr.to(self.device)
            timer_data_eval.hold()
            timer_model_eval.tic()

            sr = self.test_batch(lr, evaluate_bicubic)
            losses, loss_types = self.loss(sr, hr, type='test')
            total_losses = [total_losses[i] + (loss.item()) for i, loss in enumerate(losses)]
            total_losses.append(sum(total_losses))
            # each batch should calculate its psnr
            # @todo psnr calculation can be done in GPU
            sr_batch = np.array(sr.cpu().detach()).transpose(0, 2, 3, 1)
            hr_batch = np.array(hr.cpu().detach()).transpose(0, 2, 3, 1)
            for i in range(sr_batch.shape[0]):
                psnr_values.append(psnr(sr_batch[i, ...].squeeze(), hr_batch[i, ...].squeeze(), data_range=1))

            timer_model_eval.hold()
            timer_data_eval.tic()

        total_model_eval_time = timer_model_eval.release()
        total_data_eval_time = timer_data_eval.release()

        if take_log:
            log_string = "[{}]".format(len(self.loader_val.dataset))
            for i, log_type in enumerate(self.loss_types):
                log_string += "\t{}:{:.4f}".format(log_type, total_losses[i] / batch_counter)
            log_string += "\t{:.4f}\t{:.1f}+{:.1f}s".format(sum(psnr_values) / len(psnr_values),
                                                        total_model_eval_time, total_data_eval_time)
            self.ckp.write_log(log_string, type='validation')

        # take the model train mode
        self.model.train()

        return total_losses, sum(psnr_values) / len(psnr_values), total_model_eval_time, total_data_eval_time

    def train(self):
        print("Traning starts ...")
        self.val_loss, self.val_psnr = 1e8, 0
        while not self.terminate():
            self.train_epoch()
            self.saveModel()
            # self.test(test_mode="dataset")
            print("[Epoch {:4d}] Training results: \n".format(self.optimizer.get_last_epoch() + 1),
                  "average training loss: {}\t Lowest validation loss: {}\t".format(self.error_last, self.val_loss_best),
                  "best psnr in validation set: {}".format(self.val_psnr_best))


    def test(self, test_mode="dataset"):
        print("Test starts ...")
        if test_mode == "dataset":
            self.ckp.load(self)
            test_losses, test_psnr, data_time, model_time = self.evaluation(self.loader_test, take_log=False, evaluate_bicubic=False)
            bicubic_losses, bicubic_psnr, bicubic_data_time, bicubic_model_time = self.evaluation(self.loader_test, take_log=False, evaluate_bicubic=True)

            print("Total test losses: {:.5f}\nAverage test PSNR: {:.2f}\nTotal Test Time: {:.1f}+{:.1f}s"
                  .format(test_losses[-1]/len(self.loader_test), test_psnr, model_time, data_time))
            print("Total bicubic losses: {:.5f}\nAverage bicubic PSNR: {:.2f}\nTotal Bicubic Time: {:.1f}+{:.1f}s"
                  .format(bicubic_losses[-1] / len(self.loader_test), bicubic_psnr, bicubic_model_time, bicubic_data_time))

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
