# It includes input arguments and options used in system

import torch
import os
import pickle

class options():
    def __init__(self):
        self.experiment_name    = "PFF-deneme"
        # Hardware Specs
        self.device             = "gpu" # ["cpu", "gpu"]
        self.seed               = 1 # random seed setting for torch.random operations
        self.n_GPUs             = 1
        self.precision          = "full"
        # Dataset parameters
        self.train_set_paths    = ["/home/ferhatcan/Image_Datasets/ir_sr_challange/train"]
        self.test_set_paths     = ["/home/ferhatcan/Image_Datasets/ir_sr_challange/test"]
        self.rgb_range          = 1 # it is used in loss module, if normalized make it 1
        self.batch_size         = 4
        self.scale              = 2
        self.include_noise      = True
        self.noise_sigma        = 1
        self.noise_mean         = 0
        self.include_blur       = True
        self.blur_radius        = 0.2
        self.normalize          = "between01"
        self.random_flips       = True
        self.channel_number     = 1 # taken image channel
        self.n_colors           = 1 # output image channel this should be handled in dataLoader
        self.hr_shape           = [100, 100]
        self.downgrade          = "bicubic"
        self.validation_size    = 0.2
        self.shuffle_dataset    = True
        # Model parameters
        self.n_resblocks        = 32
        self.n_feats            = 256
        self.res_scale          = 0.1

        # Training Parameters
        self.training           = True
        self.epoch_num          = 1000
        self.loss               = "0.8*MSE+0.2*L1" # "5*VGG54+0.15*GAN" # weight*loss_type + weight*loss_type --> two different loss function
        self.skip_thr           = 1e8 # skip if a batch have high error
        self.image_range        = 1 # 255
        self.log_every          = 100 # batch number
        self.validate_every     = 0.4
        self.log_psnr           = True
        self.log_ssim           = False
        self.log_losses         = True
        self.save_best          = True
        self.save_epoch_model   = True
        self.save_path          = os.path.join("./experiments/", self.experiment_name)
        self.resume             = 0 # -1 --> load latest, 0--> load pre-trained model , else --> load from desired epoch
        self.pre_train          = "./.pre_trained_weights/pff_epoch-445.paramOnly"# "download" # ["download", "PATH/TO/PRE-TRAINED/MODEL"]
        self.only_body          = False # it should be true transfer knowdlegde from RGB
        self.fine_tuning        = False
        self.freeze_initial_layers = False
        self.chop               = False
        self.save_models        = False
        # Testing Parameters
        self.test_only          = True
        self.log_test_result    = True
        self.test_single        = False
        self.test_psnr          = True
        self.test_ssim          = False
        self.test_visualize     = False
        self.test_image_save    = False
        # Model parameters
        self.model              = 'PFF'
        self.self_ensemble      = False
        self.pre_trained_dir    = ""
        # Optimization Specs
        self.learning_rate      = 5e-5
        self.decay              = '200'
        self.decay_factor_gamma = 0.5
        self.optimizer          = "SGD" # options: ['ADAM', 'SGD', 'RMSprop']
        self.momentum           = 0.9 # option for SGD
        self.betas              = (0.9, 0.999) # option for ADAM
        self.epsilon            = 1e-8  # option for ADAM
        self.weight_decay       = 0
        self.gclip              = 0  # gradient clip between [-gclip, gclip], 0 means no clipping
        # Save config
        self.config_name        = "configs/" + self.model + "x{}".format(self.scale) + "_" + self.experiment_name
        self.save_config        = True
        self.load_config        = False
        # Checkpoint
        self.load               = ''
        self.save               = "./ckpts/" + self.model + "x{}".format(self.scale) + "/" + self.experiment_name
        self.reset              = False
        self.data_test          = ''



args = options()
print("The system will use following resource: {:}".format(args.device))
print("Experiment Name: " + args.experiment_name)
print("Experiment will be saved to " + args.save_path)

if args.save_config:
    with open(args.config_name, 'wb') as config_dictionary_file:
        pickle.dump(args, config_dictionary_file)
    print("Options are stored to ", args.config_name)

if args.load_config:
    with open(args.config_name, 'rb') as config_dictionary_file:
        args = pickle.load(config_dictionary_file)
    print("Options are loaded from ", args.config_name)