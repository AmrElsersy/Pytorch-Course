from PennFudan_Dataset import PennnFudanDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import Subset, DataLoader
from PIL import Image
import os, time, copy, random
import numpy as np

def get_coloured_mask(mask):

    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# select Image
index = 0

root = "PennFudanPed/PNGImages"
paths = os.listdir("PennFudanPed/PNGImages")
path = os.path.join(root, paths[index])

# path = os.path.join(root, "PennPed00065.png")

transform = torchvision.transforms.ToTensor()
image = Image.open(path)
image = transform(image)

with torch.no_grad():
    model.eval()
    output = model([image])[0] # [0] because we pass 1 image

    # print(output)

    import cv2
    boxes = output["boxes"].numpy()
    # convert dark masking into one-hot labeled
    # masks contains 0 and a low gray value, so it will be considered as 0 
    # convert to numpy to deal with it by openCV
    masks = (output["masks"] >= 0.5).squeeze().numpy()    

    img = cv2.imread(path)
    original = img

    for i, mask in enumerate( masks ):
        mask = get_coloured_mask(mask)
        mask = mask.reshape(img.shape)

        img = cv2.addWeighted(img, 1, mask, 0.5, 0)

        # cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2],boxes[i][3]) , (255,0,0))

    cv2.imshow("original", original)
    cv2.imshow("masked",img)

    cv2.waitKey(0)
