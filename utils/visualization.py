# Author: @ferhatcan
# Date: 24/04/20

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr

plt.ion()

def imshow_single_image(img, title="Showing Image"):
    img = np.array(img)
    if img.shape[-1] == 3:
        cmap = "viridis"
    else:
        cmap = "gray"
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()
    plt.pause(10)
    plt.close()

def imshow_image_grid(imgGrid, grid="", titles="", figSize=16):
    total_image = imgGrid.shape[0]
    if titles == "":
        titles = ["image {:d}".format(i+1) for i in range(total_image)]
    if grid == "":
        grid = (3, (total_image + 1)//3)
    if imgGrid.shape[-1] == 3:
        cmap = "viridis"
    else:
        cmap = "gray"
    aspect_ratio = (imgGrid.shape[1] / imgGrid.shape[2]) * (grid[0] / grid[1])
    figSize = figSize
    fig = plt.figure(figsize=(figSize, figSize*aspect_ratio) ,constrained_layout=True)
    # fig.suptitle(title)
    ax = plt.subplot(grid[0], grid[1], 1)
    for i in range(total_image):
        ax = plt.subplot(grid[0], grid[1], i+1, sharex=ax, sharey=ax)
        img = imgGrid[i, :].squeeze()
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        ax.title.set_text(titles[i%len(titles)])
    fig.set_constrained_layout_pads(w_pad=2. / 72., h_pad=2. / 72.,
                                    hspace=0.1, wspace=0.)
    plt.show()
    plt.pause(10)
    plt.close()

def psnr(img1, img2):
    return compare_psnr(img1, img2, data_range=255)

def ssim(img1, img2):
    multichannel = False if img1.shape[-1] == 1 else True
    return compare_ssim(img1, img2, data_range=255, multichannel=multichannel)


# Test Parts for above functions
# img = Image.open("0001.jpg")
# img_np = np.array(img)
# imgGrid = np.array([np.clip((img_np[:,:,0] + np.random.normal(0, 4, img_np[:,:,0].shape)), a_min=0, a_max=255)
#                             for _ in range(9)])
# imshow_image_grid(imgGrid, titles=["SR", "LR"])
# print(psnr(img_np, img_np + np.random.normal(0, 1, img_np.shape)))
# print(ssim(img_np, img_np + np.random.normal(0, 1, img_np.shape)))
