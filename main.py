import torch
from torchvision.transforms import functional as tvF
import numpy as np
import PIL.Image as Image

from dataloaders.irChallangeDataset import irChallangeDataset
from dataloaders.pffDataset import pffDataset

import model
import loss
from options import args
from methods.EDSR_training_method import EDSR_training_method
from utils.checkpoint import checkpoint

from utils.visualization import imshow_image_grid

from torch.utils.tensorboard import SummaryWriter
import datetime

torch.manual_seed(args.seed)
ckp = checkpoint(args)

tb_experiment_name = "runs/" + args.model + "x{}".format(args.scale) + "/" \
                  + args.experiment_name + "_" + datetime.datetime.now().strftime('%Y-%m-%d')
writer = SummaryWriter(tb_experiment_name)

# def main():
#     global model
#     if args.data_test == ['video']:
#         from videotester import VideoTester
#         model = model.Model(args, checkpoint)
#         t = VideoTester(args, model, checkpoint)
#         t.test()
#     else:
#         if checkpoint.ok:
#             loader = data.Data(args)
#             _model = model.Model(args, checkpoint)
#             _loss = loss.Loss(args, checkpoint) if not args.test_only else None
#             t = Trainer(args, loader, _model, _loss, checkpoint)
#             while not t.terminate():
#                 t.train()
#                 t.test()
#
#             checkpoint.done()

def imshow(loader):
    data = next(iter(loader))
    lr_batch = (np.array(data[0]).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    lr_batch_SR = [tvF.resize(Image.fromarray(lr_batch[i,...].squeeze()), size=args.hr_shape, interpolation=Image.BICUBIC)
                            for i in range(lr_batch.shape[0])]
    if len(lr_batch.shape) == 3:
        lr_batch = [np.array(lr_batch_SR[i])[..., np.newaxis] for i in range(len(lr_batch_SR))]
    else:
        lr_batch = [np.array(lr_batch_SR[i]) for i in range(len(lr_batch_SR))]
    lr_batch = np.stack(lr_batch, axis=0)
    hr_batch = (np.array(data[1]).transpose(0, 2, 3, 1) * 255.0).clip(min=0, max=255).astype(np.uint8)
    imshow_image_grid(np.array(np.concatenate([lr_batch, hr_batch], axis=0)), grid=[2, hr_batch.shape[0]], figSize=10)


def main():
    if ckp.ok:
        loader = pffDataset(args)
        _model = model.Model(args, ckp)
        # _loss = loss.Loss(args, ckp)
        # _method = EDSR_training_method(args, loader, _model, _loss, ckp, log_writer=writer)
        imshow(loader.loader_test)
        # if args.test_only:
        #     _method.test(test_mode="dataset")
        # else:
        #     _method.train()

if __name__ == '__main__':
    main()
