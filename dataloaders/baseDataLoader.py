# Author: @ferhatcan
# Date: 24/04/20

import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as tvF
import os
import random
import PIL.Image as Image
from PIL import ImageFilter

# This base dataset loader class will implement the following properties
# +1: Train dataloader & Validation DataLoader
# +2: Adding Noise
# +3: Adding Blur
# +4: Downgrade options
# +5: Transforms
# +6: Random Crops or ..
# 7:

class BaseDatasetLoader(torch.utils.data.Dataset):
    def __init__(self,
                 image_paths,
                 scale=1,
                 include_noise=True,
                 noise_sigma=1,
                 noise_mean=0,
                 include_blur=True,
                 blur_radius = 2,
                 normalize="zeroMean", # ["zeroMean", "between01"]
                 randomflips=True,
                 channel_number=1,
                 hr_shape=[96, 96],
                 downgrade="bicubic", # ["nearest", "bilinear", "bicubic"]
                 ):

        self.image_paths = image_paths
        self.scale = scale
        self.include_noise = include_noise
        self.noise_sigma = noise_sigma
        self.noise_mean = noise_mean
        self.include_blur = include_blur
        self.blur_radius = blur_radius
        self.normalize = normalize
        self.random_flips = randomflips
        self.channel_number = channel_number
        self.hr_shape = hr_shape
        self.downgrade = downgrade

        if self.downgrade == "bicubic":
            self.downgrade = Image.BILINEAR
        elif self.downgrade == "nearest":
            self.downgrade = Image.NEAREST
        elif self.downgrade == "bilinear":
            self.downgrade = Image.BILINEAR

        self.lr_shape = [i // self.scale for i in self.hr_shape]

        self.extensions = ["jpg", "jpeg", "png"]
        self.imageFiles = []
        for image_path in self.image_paths:
            for root, directory, fileNames in os.walk(image_path):
                for file in fileNames:
                    ext = file.split('.')[-1]
                    if ext in self.extensions:
                        self.imageFiles.append(os.path.join(root, file))

        assert (len(image_paths) > 0), "There should be an valid image path"

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        hr_image = Image.open(self.imageFiles[index])
        return self.transform(hr_image)

    def transform(self, image):
        # desired channel number should be checked
        convert_mode = "RGB" if self.channel_number == 3 else "L"
        image = image.convert(convert_mode)

        # Resize input image if its dimensions smaller than desired dimensions
        resize = transforms.Resize(size=self.hr_shape, interpolation=self.downgrade)
        if not (image.width > self.hr_shape[0] and image.height > self.hr_shape[1]):
            image = resize(image)

        # random crop
        crop = transforms.RandomCrop(size=self.hr_shape)
        hr_image = crop(image)

        # downscale to obtain low-resolution image
        resize = transforms.Resize(size=self.lr_shape, interpolation=self.downgrade)
        lr_image = resize(hr_image)

        # apply blur
        if self.include_blur:
            lr_image = lr_image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # apply random transforms
        if self.random_flips:
            # random horizontal flip
            if random.random() > 0.5:
                hr_image = tvF.hflip(hr_image)
                lr_image = tvF.hflip(lr_image)

            # random vertical flip
            if random.random() > 0.5:
                hr_image = tvF.vflip(hr_image)
                lr_image = tvF.vflip(lr_image)

        # apply noise
        lr_image = np.array(lr_image)
        hr_image = np.array(hr_image)
        if self.include_noise:
            lr_image = np.array(np.clip((lr_image +
                                         np.random.normal(self.noise_mean, self.noise_sigma, lr_image.shape)),
                                        a_min=0, a_max=255).astype("uint8"))


        # Transform to tensor
        hr_image = tvF.to_tensor(Image.fromarray(hr_image))
        lr_image = tvF.to_tensor(Image.fromarray(lr_image))

        # apply normalization
        if self.normalize == "zeroMean":
            # todo Mean & STD of the dataset should be given or It can be calculated in a method
            hr_means = (hr_image.mean() for i in range(hr_image.shape[0]))
            lr_means = (lr_image.mean() for i in range(lr_image.shape[0]))
            hr_stds = (hr_image.std() for i in range(hr_image.shape[0]))
            lr_stds = (lr_image.std() for i in range(lr_image.shape[0]))
            hr_image = tvF.normalize(hr_image, hr_means, hr_stds)
            lr_image = tvF.normalize(lr_image, lr_means, lr_stds)
        elif self.normalize == "between01":
            hr_mins = (hr_image.min() for i in range(hr_image.shape[0]))
            lr_mins = (lr_image.min() for i in range(lr_image.shape[0]))
            hr_ranges = (hr_image.max() - hr_image.min() for i in range(hr_image.shape[0]))
            lr_ranges = (lr_image.max() - lr_image.min() for i in range(lr_image.shape[0]))
            hr_image = tvF.normalize(hr_image, hr_mins, hr_ranges)
            lr_image = tvF.normalize(lr_image, lr_mins, lr_ranges)

        return lr_image, hr_image


# Test dataset class

# validation_split = 0.1
# batch_size = 4
# dataset_path = "/home/ferhatcan/Image_Datasets/ir_sr_challange"
# shuffle_dataset = True
# hr_shape = [360, 640]
#
# ir_challange_dataset = BaseDatasetLoader([dataset_path],
#                                          scale=2, normalize="between01",
#                                          hr_shape=hr_shape,
#                                          randomflips=False)
# dataset_size = len(ir_challange_dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset:
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
#
# train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
# validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
#
# train_loader = torch.utils.data.DataLoader(ir_challange_dataset, batch_size=batch_size,
#                                            sampler=train_sampler)
# validation_loader = torch.utils.data.DataLoader(ir_challange_dataset, batch_size=batch_size,
#                                            sampler=validation_sampler)
#
# from utils.visualization import imshow_image_grid
# data = next(iter(train_loader))
# #for i, data in enumerate(train_loader):
#     #print(data[0].size(), data[1].size())
# lr_batch = np.array(data[0]).transpose(0, 2, 3, 1)
# lr_batch_SR = [tvF.resize(Image.fromarray(lr_batch[i,...].squeeze()), size=hr_shape, interpolation=Image.BICUBIC)
#                         for i in range(lr_batch.shape[0])]
# lr_batch = [np.array(lr_batch_SR[i])[..., np.newaxis] for i in range(len(lr_batch_SR))]
# lr_batch = np.stack(lr_batch, axis=0)
# hr_batch = np.array(data[1]).transpose(0, 2, 3, 1)
# imshow_image_grid(np.array(np.concatenate([lr_batch, hr_batch], axis=0)), grid=[2, 4], figSize=10)
#
# tmp = 0
