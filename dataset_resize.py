import os
import cv2
from matplotlib import image
import numpy as np



for image in os.listdir('dataset/images'):
    img = np.load(os.path.join('dataset/images/', image))
    mask_file = image.replace('image_', 'mask_')
    mask = np.load(os.path.join('dataset/masks/', mask_file))
    img_height = img.shape[0]
    img_width = img.shape[1]
    scale = min(256 / img_height, 256 / img_width)
    new_height = int(img_height * scale)
    new_width = int(img_width * scale)
    image_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    pad_top = (256 - new_height) // 2
    pad_bottom = 256 - new_height - pad_top
    pad_left = (256 - new_width) // 2
    pad_right = 256 - new_width - pad_left
    image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    assert image_padded.shape == (256, 256)
    assert mask_padded.shape == (256, 256)
    np.save(os.path.join('dataset/images_resized/', image), image_padded)
    np.save(os.path.join('dataset/masks_resized/', mask_file), mask_padded)



