from .encoderDecoder import *

__all__ = ['EncoderDecoder']

def make_model(args, parent=False):
    return EncoderDecoder(args)