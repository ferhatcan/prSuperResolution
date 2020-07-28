import torch
import copy
import torch.nn as nn
import torchvision


class EncoderDecoder(nn.Module):
    def __init__(self, args, load_model_path=None):
        super(EncoderDecoder, self).__init__()
        self.args = args

        self.encoder = Encoder()
        self.decoder = Decoder()

        if not load_model_path is None:
            self.load_state_dict(torch.load(load_model_path))

    def forward(self, image_ir, image_eo):
        feat_ir, feat_eo = self.encoder(image_ir, image_eo)
        image_ir, image_eo = self.decoder(feat_ir, feat_eo)
        return image_ir, image_eo


class EncoderFusionDecoder(nn.Module):
    def __init__(self, args, load_path=None):
        super(EncoderFusionDecoder, self).__init__()
        model = EncoderDecoder(args=args, load_model_path=load_path)
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.fusion_layer = nn.Conv2d(4096, 2048, kernel_size=(1,1))

    def forward(self, image_ir, image_eo):
        feat_ir, feat_eo = self.encoder(image_ir, image_eo)
        fusion = torch.cat((feat_ir, feat_eo), dim=1)
        output = self.decoder.forward_only_ir(self.fusion_layer(fusion))
        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=(4, 4), stride=stride, padding=(1, 1),
                           groups=groups, bias=False)
        self.bn2 = norm_layer(inplanes)
        self.conv3 = conv1x1(inplanes, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        if stride != 1 or self.inplanes != planes:
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=(2, 2), stride=stride, bias=False),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext50_32x4d', pretrained=True)

        self.encoder_layer_shared = copy.deepcopy(model.layer1[0])

        self.encoder_layer_2_eo = copy.deepcopy(model.layer2[0])
        self.encoder_layer_3_eo = copy.deepcopy(model.layer3[0])
        self.encoder_layer_4_eo = copy.deepcopy(model.layer4[0])

        self.encoder_layer_2_ir = copy.deepcopy(model.layer2[0])
        self.encoder_layer_3_ir = copy.deepcopy(model.layer3[0])
        self.encoder_layer_4_ir = copy.deepcopy(model.layer4[0])

        self.conv1_eo = model.conv1
        self.conv1_ir = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.encoder_shared = nn.Sequential(
            model.bn1,
            model.relu,
            model.maxpool,
            self.encoder_layer_shared,
        )
        self.encoder_eo = nn.Sequential(
            self.encoder_layer_2_eo,
            self.encoder_layer_3_eo,
            self.encoder_layer_4_eo
        )

        self.encoder_ir = nn.Sequential(
            self.encoder_layer_2_ir,
            self.encoder_layer_3_ir,
            self.encoder_layer_4_ir
        )

    def forward(self, image_ir, image_eo):
        h_ir = self.conv1_ir(image_ir)
        h_eo = self.conv1_eo(image_eo)

        h_ir = self.encoder_shared(h_ir)
        h_ir = self.encoder_ir(h_ir)

        h_eo = self.encoder_shared(h_eo)
        h_eo = self.encoder_eo(h_eo)

        return h_ir, h_eo


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1_ir = Bottleneck(2048, 512, stride=2, groups=32, base_width=4)
        self.layer2_ir = Bottleneck(512, 128, stride=2, groups=32, base_width=4)
        self.layer3_ir = Bottleneck(128, 32, stride=2, groups=32, base_width=4)
        self.layer3_2_ir = Bottleneck(32, 32, stride=2, groups=32, base_width=4)

        self.layer1_eo = Bottleneck(2048, 512, stride=2, groups=32, base_width=4)
        self.layer2_eo = Bottleneck(512, 128, stride=2, groups=32, base_width=4)
        self.layer3_eo = Bottleneck(128, 32, stride=2, groups=32, base_width=4)

        self.layer_shared_4 = Bottleneck(32, 8, stride=2, groups=32, base_width=4)
        self.layer_shared_5 = Bottleneck(8, 3, stride=2, groups=8, base_width=4)
        self.tanh = nn.Tanh()

        self.decoder_ir = nn.Sequential(
            self.layer1_ir,
            self.layer2_ir,
            self.layer3_ir,
            self.layer3_2_ir
        )

        self.decoder_eo = nn.Sequential(
            self.layer1_eo,
            self.layer2_eo,
            self.layer3_eo
        )

        self.decoder_shared = nn.Sequential(
            self.layer_shared_4,
            self.layer_shared_5,
            self.tanh
        )

    def forward(self, feat_ir, feat_eo):
        h_ir = self.decoder_ir(feat_ir)
        h_ir = self.decoder_shared(h_ir)

        h_eo = self.decoder_eo(feat_eo)
        h_eo = self.decoder_shared(h_eo)

        return h_ir, h_eo

    def forward_only_ir(self, feat_ir):
        h_ir = self.decoder_ir(feat_ir)
        h_ir = self.decoder_shared(h_ir)
        return h_ir


# testing puposes

# from dataloaders.div2k import div2K
# from dataloaders.irChallangeDataset import irChallangeDataset
# from options import options
# import matplotlib.pyplot as plt
#
#
# CONFIG_FILE_NAME = "./../../configs/PFFx2_fineTuning_KAIST.ini"
# args = options(CONFIG_FILE_NAME)
#
# args.train_set_paths = ('/media/ferhatcan/New Volume/Image_Datasets/ir_sr_challange/train/640_flir_hr', )
# args.test_set_paths = ('/media/ferhatcan/New Volume/Image_Datasets/ir_sr_challange/test/640_flir_hr', )
#
# dl_div2k = div2K(args, train_path=('/media/ferhatcan/New Volume/Image_Datasets/div2k/images/train/DIV2K_train_HR', ),
#                  test_path= ('/media/ferhatcan/New Volume/Image_Datasets/div2k/images/validation/DIV2K_valid_HR', ))
#
# dl_ir = irChallangeDataset(args)
#
# lr_eo, hr_eo = next(iter(dl_div2k.loader_train))
# lr_ir, hr_ir = next(iter(dl_ir.loader_train))
#
#
# # plt.ion()
# # image = hr_eo[0, ...].unsqueeze(dim=0)
# # im = image.permute(0, 2, 3, 1).squeeze()
# # plt.imshow(hr_eo[0, 0, ...].squeeze())
# # plt.waitforbuttonpress()
# #
# # plt.figure()
# # plt.imshow(im)
#
# model = EncoderDecoder(args=args)
# loss_function = nn.MSELoss()
#
# optim = torch.optim.Adam(model.parameters(), lr=0.0001)
#
# model.to('cuda')
# hr_eo = hr_eo.to('cuda')
# lr_ir = lr_ir.to('cuda')
# hr_ir = hr_ir.to('cuda')
#
# model.train()
# sr_ir, eo = model(lr_ir, hr_eo)
# loss_ir = loss_function(sr_ir, hr_ir)
# loss_eo = loss_function(eo, hr_eo)
# loss = loss_eo + loss_ir
# loss.backward()
# for p in model.encoder.encoder_shared.parameters():
#     p.grad.data = 0.5 * p.grad.data
# for p in model.decoder.decoder_shared.parameters():
#     p.grad.data = 0.5 * p.grad.data
#
# optim.step()
# model.zero_grad()
#
# torch.save(model.state_dict(), './../../.pre_trained_weights/encoderDecoder.pth')
#
# modelfusion = EncoderFusionDecoder(args=args, load_path='./../../.pre_trained_weights/encoderDecoder.pth')
# modelfusion.to('cuda')
# lr_eo = lr_eo.to('cuda')
# sr_ir = modelfusion(lr_ir, lr_eo)
#
# print(modelfusion)
# tmp = 0