import numpy as np
from torchvision import transforms
from PIL import Image


class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class SetFlip(object):

    def __init__(self, flip):
        self.flip = flip

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class land_transform(object):
    def __init__(self, output_size, scale, flip_reflect):
        self.output_size = output_size
        self.scale = scale
        self.flip_reflect = flip_reflect.astype(int) - 1

    def __call__(self, land, flip, offset_x, offset_y):
        land_label = np.zeros((len(land) / 2))
        land[0:len(land):2] = (land[0:len(land):2] - offset_x) / float(self.scale)
        land[1:len(land):2] = (land[1:len(land):2] - offset_y) / float(self.scale)
        # change the landmark orders when flipping
        if flip:
            land[0:len(land):2] = self.output_size - 1 - land[0:len(land):2]
            land[0:len(land):2] = land[0:len(land):2][self.flip_reflect]
            land[1:len(land):2] = land[1:len(land):2][self.flip_reflect]

        # landmark location refinement for predefined AU centers
        ruler = abs(land[2 * 22] - land[2 * 25])

        land[2 * 4 + 1] = land[2 * 4 + 1] - ruler / 2
        land[2 * 5 + 1] = land[2 * 5 + 1] - ruler / 2
        land[2 * 1 + 1] = land[2 * 1 + 1] - ruler / 3
        land[2 * 8 + 1] = land[2 * 8 + 1] - ruler / 3
        land[2 * 2 + 1] = land[2 * 2 + 1] + ruler / 3
        land[2 * 7 + 1] = land[2 * 7 + 1] + ruler / 3
        land[2 * 24 + 1] = land[2 * 24 + 1] + ruler
        land[2 * 29 + 1] = land[2 * 29 + 1] + ruler
        land[2 * 15 + 1] = land[2 * 15 + 1] - ruler / 2
        land[2 * 17 + 1] = land[2 * 17 + 1] - ruler / 2
        land[2 * 39 + 1] = land[2 * 39 + 1] + ruler / 2
        land[2 * 41 + 1] = land[2 * 41 + 1] + ruler / 2

        land = (np.around(land)).astype(int)

        for i in range(len(land) / 2):
            land[2 * i] = min(max(land[2 * i], 0), self.output_size - 1)
            land[2 * i + 1] = min(max(land[2 * i + 1], 0), self.output_size - 1)

            land_label[i] = land[2 * i + 1] * self.output_size + land[2 * i]

        return land_label


class image_train(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, flip, offset_x, offset_y):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            PlaceCrop(self.crop_size, offset_x, offset_y),
            SetFlip(flip),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def image_test(crop_size=176):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])