# It includes input arguments and options used in system
from configparser import ConfigParser


class options:
    def __init__(self, config_file_name):
        self.config = ConfigParser()
        self.config.read(config_file_name)

        self.parseDefaults()
        self.parseHarware()
        self.parseDataset()
        self.parseModel()
        self.parseOptimization()
        self.parseOptimization()
        self.parseTraining()
        self.parseTesting()
        self.parseCheckpoint()

    def parseDefaults(self):
        self.experiment_name = self.config["DEFAULT"]["experiment_name"]

    def parseHarware(self):
        self.device     = self.config["HARDWARE"]["device"]
        self.seed       = int(self.config["HARDWARE"]["seed"])
        self.n_GPUs     = int(self.config["HARDWARE"]["n_GPUs"])
        self.precision  = self.config["HARDWARE"]["precision"]

    def parseDataset(self):
        self.train_set_paths    = self.config["DATASET"]["train_set_paths"].split(',')
        self.test_set_paths     = self.config["DATASET"]["test_set_paths"].split(',')
        self.rgb_range          = int(self.config["DATASET"]["rgb_range"])
        self.batch_size         = int(self.config["DATASET"]["batch_size"])
        self.scale              = int(self.config["DATASET"]["scale"])
        self.include_noise      = self.config["DATASET"].getboolean("include_noise")
        self.noise_sigma        = float(self.config["DATASET"]["noise_sigma"])
        self.noise_mean         = float(self.config["DATASET"]["noise_mean"])
        self.include_blur       = self.config["DATASET"].getboolean("include_blur")
        self.blur_radius        = float(self.config["DATASET"]["blur_radius"])
        self.normalize          = self.config["DATASET"]["normalize"]
        self.random_flips       = self.config["DATASET"].getboolean("random_flips")
        self.channel_number     = int(self.config["DATASET"]["channel_number"])
        self.n_colors           = int(self.config["DATASET"]["n_colors"])
        self.hr_shape           = list(map(int, self.config["DATASET"]["hr_shape"].split(',')))
        self.downgrade          = self.config["DATASET"]["downgrade"]
        self.validation_size    = float(self.config["DATASET"]["validation_size"])
        self.shuffle_dataset    = self.config["DATASET"].getboolean("shuffle_dataset")

    def parseModel(self):
        self.model          = self.config["MODEL"]["model"]
        self.self_ensemble  = self.config["MODEL"].getboolean("self_ensemble")
        if self.model == 'EDSR':
            self.n_resblocks    = int(self.config["EDSR"]["n_resblocks"])
            self.n_feats        = int(self.config["EDSR"]["n_feats"])
            self.res_scale      = float(self.config["EDSR"]["res_scale"])
        elif self.model == 'PFF':
            self.emb_dimension  = int(self.config["PFF"]["emb_dimension"])
            self.filterSize     = int(self.config["PFF"]["filterSize"])
            self.pretrained     = self.config["PFF"].getboolean("pretrained")
        else:
            self.n_resblocks    = 32
            self.n_feats        = 256
            self.res_scale      = 0.1
            self.emb_dimension  = 16
            self.filterSize     = 17
            self.pretrained     = True

    def parseOptimization(self):
        self.learning_rate          = float(self.config["OPTIMIZATION"]["learning_rate"])
        self.decay                  = self.config["OPTIMIZATION"]["decay"]
        self.decay_factor_gamma     = float(self.config["OPTIMIZATION"]["decay_factor_gamma"])
        self.optimizer              = self.config["OPTIMIZATION"]["optimizer"]
        self.momentum               = float(self.config["OPTIMIZATION"]["momentum"])
        self.betas                  = list(map(float, self.config["OPTIMIZATION"]["betas"].split(',')))
        self.epsilon                = float(self.config["OPTIMIZATION"]["epsilon"])
        self.weight_decay           = float(self.config["OPTIMIZATION"]["weight_decay"])
        self.gclip                  = float(self.config["OPTIMIZATION"]["gclip"])

    def parseCheckpoint(self):
        if self.config["CHECKPOINT"].getboolean("load"):
            self.load = "./ckpts/" + self.model + "x{}".format(self.scale) + "/" + self.experiment_name
        else:
            self.load = ''
        if self.config["CHECKPOINT"].getboolean("save"):
            self.save = "./ckpts/" + self.model + "x{}".format(self.scale) + "/" + self.experiment_name
        else:
            self.save = ''

        self.reset      = self.config["CHECKPOINT"].getboolean("reset")
        self.data_test  = self.config["CHECKPOINT"]["data_test"]

    def parseTraining(self):
        self.training           = self.config["TRAINING"].getboolean("training")
        self.epoch_num          = int(self.config["TRAINING"]["epoch_num"])
        self.loss               = self.config["TRAINING"]["loss"]
        self.skip_thr           = float(self.config["TRAINING"]["skip_thr"])
        self.image_range        = int(self.config["TRAINING"]["image_range"])
        self.log_every          = int(self.config["TRAINING"]["log_every"])
        self.validate_every     = float(self.config["TRAINING"]["validate_every"])
        self.log_psnr           = self.config["TRAINING"].getboolean("log_psnr")
        self.log_ssim           = self.config["TRAINING"].getboolean("log_ssim")
        self.save_path          = "/experiment/" + self.experiment_name
        self.pre_train          = self.config["TRAINING"]["pre_train"]
        self.only_body          = self.config["TRAINING"].getboolean("only_body")
        self.fine_tuning        = self.config["TRAINING"].getboolean("fine_tuning")
        self.freeze_initial_layers = self.config["TRAINING"].getboolean("freeze_initial_layers")
        self.chop               = self.config["TRAINING"].getboolean("chop")
        self.save_models        = self.config["TRAINING"].getboolean("save_models")

        if self.pre_train in ["load_latest", "load_best"]:
            self.resume = -1  # -1 --> load latest, 0--> load pre-trained model , else --> load from desired epoch
        else:
            self.resume = 0

    def parseTesting(self):
        self.test_only              = self.config["TESTING"].getboolean("test_only")
        self.log_test_result        = self.config["TESTING"].getboolean("log_test_result")
        self.test_single            = self.config["TESTING"].getboolean("test_single")
        self.test_psnr              = self.config["TESTING"].getboolean("test_psnr")
        self.test_ssim              = self.config["TESTING"].getboolean("test_ssim")
        self.test_visualize         = self.config["TESTING"].getboolean("test_visualize")
        self.test_image_save        = self.config["TESTING"].getboolean("test_image_save")
