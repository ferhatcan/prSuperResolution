from .siamesePixelEmbed import *

__all__ = ['SiamesePixelEmbed']

def make_model(args, parent=False):
    return SiamesePixelEmbed(args, emb_dimension=16, filterSize=17, device=args.device, pretrained=True)