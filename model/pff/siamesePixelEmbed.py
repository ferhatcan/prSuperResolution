import torch.nn as nn
import torch.nn.functional as F

from model.pff.pixel_embedding_model import PixelEmbedModelResNet18


class SiamesePixelEmbed(nn.Module):
    def __init__(self, args, emb_dimension=64, filterSize=11, device='cpu', pretrained=False):
        super(SiamesePixelEmbed, self).__init__()
        self.args = args
        self.device = device
        self.emb_dimension = emb_dimension
        self.PEMbase = PixelEmbedModelResNet18(emb_dimension=self.emb_dimension, pretrained=pretrained)
        self.rawEmbFeature1 = 0
        self.rawEmbFeature2 = 0
        self.embFeature1_to_2 = 0
        self.embFeature1_to_2 = 0
        self.filterSize = filterSize
        self.filterSize2Channel = self.filterSize**2

        self.ordered_embedding = nn.Sequential(
            nn.Conv2d(self.emb_dimension, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.filterSize2Channel),
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.filterSize2Channel),
            nn.Conv2d(self.filterSize2Channel, self.filterSize2Channel, kernel_size=3, padding=1, bias=True)
        )


    def forward(self, inputs1, inputs2):
        self.rawEmbFeature1 = self.PEMbase.forward(inputs1)

        self.embFeature1_to_2 = self.ordered_embedding(self.rawEmbFeature1)
        self.embFeature1_to_2 = F.softmax(self.embFeature1_to_2, 1)

        return self.embFeature1_to_2

