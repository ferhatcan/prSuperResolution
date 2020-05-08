import torch
import numpy as np
import PIL.Image as Image
from torchvision import transforms

from dataloaders.baseDataset import BaseDataset

class pffDataset():
    def __init__(self, args):
        self.args = args

        self.ds_train = upscaleLr(args)

        self.ds_test = upscaleLr(args)

        dataset_size = len(self.ds_train)
        indices = list(range(dataset_size))
        split = int(np.floor(self.args.validation_size * dataset_size))
        if self.args.shuffle_dataset:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

        self.loader_train = torch.utils.data.DataLoader(self.ds_train, batch_size=self.args.batch_size,
                                                        sampler=train_sampler)
        self.loader_val = torch.utils.data.DataLoader(self.ds_train, batch_size=self.args.batch_size,
                                                      sampler=validation_sampler)
        self.loader_test = torch.utils.data.DataLoader(self.ds_test, batch_size=self.args.batch_size, shuffle=False)


class upscaleLr(BaseDataset):
    def __init__(self, args):
        self.args = args
        super(upscaleLr, self).__init__(self.args.train_set_paths,
                                        scale=self.args.scale, include_noise=args.include_noise,
                                        noise_sigma=args.noise_sigma,
                                        noise_mean=args.noise_mean, include_blur=args.include_blur,
                                        blur_radius=args.blur_radius,
                                        normalize=self.args.normalize, randomflips=args.random_flips,
                                        channel_number=3,
                                        hr_shape=self.args.hr_shape, downgrade=args.downgrade)

    def __getitem__(self, index):
        hr_image = Image.open(self.imageFiles[index])
        lr_image, hr_image = self.transform(hr_image)

        # Upscale lr_image
        lr_image = torch.nn.functional.interpolate(lr_image.unsqueeze(0),
                                                   scale_factor=self.scale, mode='bicubic', align_corners=True).squeeze() # bicubic only suppart 4D

        return lr_image, hr_image