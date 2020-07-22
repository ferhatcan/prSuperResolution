import torch
import torch.nn as nn
import torch.nn.functional as F

def make_loss(args, type):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == "gpu" else "cpu")
    return lossOrderedPairReconstruction(device, args.filterSize)

# Loss function is used in Predictive Filter Flow
class lossOrderedPairReconstruction(nn.Module):
    def __init__(self, device='cpu', filterSize=11):
        super(lossOrderedPairReconstruction, self).__init__()
        self.device = device
        self.filterSize = filterSize
        self.filterSize2Channel = self.filterSize ** 2
        self.reconstructImage = 0

    def forward(self, image1, image2, filters_img1_to_img2):
        N, C, H, W = image1.size()
        self.reconstructImage = self.rgbImageFilterFlow(image1, filters_img1_to_img2)
        diff = self.reconstructImage - image2
        diff = torch.abs(diff)
        totloss = torch.sum(torch.sum(torch.sum(torch.sum(diff))))
        return totloss / (N * C * H * W)

    def rgbImageFilterFlow(self, img, filters):
        inputChannelSize = 1
        outputChannelSize = 1
        N = img.size(0)
        paddingFunc = nn.ZeroPad2d(int(self.filterSize / 2))
        img = paddingFunc(img)
        imgSize = [img.size(2), img.size(3)]

        out_R = F.unfold(img[:, 0, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_R = out_R.view(N, out_R.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_R = paddingFunc(out_R)
        out_R = torch.mul(out_R, filters)
        out_R = torch.sum(out_R, dim=1).unsqueeze(1)

        out_G = F.unfold(img[:, 1, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_G = out_G.view(N, out_G.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_G = paddingFunc(out_G)
        out_G = torch.mul(out_G, filters)
        out_G = torch.sum(out_G, dim=1).unsqueeze(1)

        out_B = F.unfold(img[:, 2, :, :].unsqueeze(1), (self.filterSize, self.filterSize))
        out_B = out_B.view(N, out_B.size(1), imgSize[0] - self.filterSize + 1, imgSize[1] - self.filterSize + 1)
        # out_B = paddingFunc(out_B)
        out_B = torch.mul(out_B, filters)
        out_B = torch.sum(out_B, dim=1).unsqueeze(1)
        return torch.cat([out_R, out_G, out_B], 1)

