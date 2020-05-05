import torch
import numpy as np

from dataloaders.baseDataset import BaseDataset

class irChallangeDataset():
    def __init__(self, args):
        self.args = args

        self.ds_train = BaseDataset(self.args.train_set_paths,
                                    scale=self.args.scale, include_noise=args.include_noise,
                                    noise_sigma=args.noise_sigma,
                                    noise_mean=args.noise_mean, include_blur=args.include_blur,
                                    blur_radius=args.blur_radius,
                                    normalize=self.args.normalize, randomflips=args.random_flips,
                                    channel_number=self.args.channel_number,
                                    hr_shape=self.args.hr_shape, downgrade=args.downgrade)

        self.ds_test = BaseDataset(self.args.test_set_paths,
                                   scale=self.args.scale, include_noise=args.include_noise,
                                   noise_sigma=args.noise_sigma,
                                   noise_mean=args.noise_mean, include_blur=args.include_blur,
                                   blur_radius=args.blur_radius,
                                   normalize=self.args.normalize, randomflips=args.random_flips,
                                   channel_number=self.args.channel_number,
                                   hr_shape=self.args.hr_shape, downgrade=args.downgrade)

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




