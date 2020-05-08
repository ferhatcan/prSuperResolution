import torch
from torchvision.transforms import functional as tvF
import numpy as np
import PIL.Image as Image

from dataloaders.irChallangeDataset import irChallangeDataset
from dataloaders.pffDataset import pffDataset

import model
import loss
from options import options
from methods.EDSR_training_method import EDSR_training_method
from methods.PFF_method import PFF_method
from utils.checkpoint import checkpoint

from utils.visualization import imshow_image_grid
from utils.visualization import psnr, ssim

from torch.utils.tensorboard import SummaryWriter
import datetime

CONFIG_FILE_NAME = "./configs/PFFx2_fineTuning.ini"

args = options(CONFIG_FILE_NAME)
print("The system will use following resource: {:}".format(args.device))
print("Experiment Name: " + args.experiment_name)
print("Experiment will be saved to " + args.save_path)

torch.manual_seed(args.seed)
ckp = checkpoint(args)

tb_experiment_name = "runs/" + args.model + "x{}".format(args.scale) + "/" \
                  + args.experiment_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d')
writer = SummaryWriter(tb_experiment_name)


def imshow(loader):
    data = next(iter(loader))
    lr_batch = [torch.nn.functional.interpolate(data[0][i, ...].unsqueeze(0), scale_factor=args.hr_shape[0]/data[0].shape[2],
                                       mode='bicubic', align_corners=True).squeeze() for i in range(data[0].shape[0])]
    lr_batch = torch.stack(lr_batch, dim=0)
    lr_batch = (np.array(lr_batch).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    hr_batch = (np.array(data[1]).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate([lr_batch, hr_batch], axis=0)), grid=[2, hr_batch.shape[0]], figSize=10)


def main():
    if ckp.ok:
        loader = pffDataset(args)
        _model = model.Model(args, ckp)
        _loss = loss.Loss(args, ckp)
        _method = PFF_method(args, loader, _model, _loss, ckp, log_writer=writer)
        # imshow(loader.loader_test)
        if args.test_visualize:
            total_batch = 2
            for _ in range(total_batch):
                lr, sr, hr = _method.test_single()
                for i in range(lr.shape[0]):
                    print("[LR-HR, SR-HR] psnr: {:.2f}, {:.2f}".format(psnr(lr[i, ...].squeeze(), hr[i, ...].squeeze()),
                                                                    psnr(sr[i, ...].squeeze(), hr[i, ...].squeeze())))
                    print("[LR-HR, SR-HR] ssim: {:.2f}, {:.2f}".format(ssim(lr[i, ...].squeeze(), hr[i, ...].squeeze()),
                                                                    ssim(sr[i, ...].squeeze(), hr[i, ...].squeeze())))
                imshow_image_grid(np.array(np.concatenate([lr, sr, hr], axis=0)), grid=[3, hr.shape[0]], figSize=10)
        if args.test_only:
            print("Total number of test image: ", len(loader.loader_test))
            _method.test(test_mode="dataset")
        else:
            _method.train()


if __name__ == '__main__':
    main()
