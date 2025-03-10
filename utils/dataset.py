import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        self.num_classes = len(params['names'])

        # Read labels
        cache = self.load_label(filenames)
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.filenames = list(cache.keys())  # update
        self.n = len(shapes)  # number of samples
        self.indices = range(self.n)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            shapes = None
            # Load MOSAIC
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)
            shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)
        nl = len(label)  # number of labels
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        if self.augment:
            # Albumentations
            image, label = self.albumentations(image, label)
            nl = len(label)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    label[:, 2] = 1 - label[:, 2]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, shapes

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        # Load original image without resizing
        image = cv2.imread(self.filenames[i])
        return image, image.shape[:2]

    # def load_mosaic(self, index, params):
    #     label4 = []
    #     if isinstance(self.input_size, tuple):
    #         target_h, target_w = self.input_size
    #         mosaic_h, mosaic_w = 2 * target_h, 2 * target_w
    #         border = (-target_h // 2, -target_w // 2)
    #     else:
    #         mosaic_h = mosaic_w = 2 * self.input_size
    #         border = [-self.input_size // 2, -self.input_size // 2]

    #     image4 = numpy.full((mosaic_h, mosaic_w, 3), 0, dtype=numpy.uint8)
    #     xc = int(random.uniform(-border[0], mosaic_w + border[1]))
    #     yc = int(random.uniform(-border[0], mosaic_h + border[1]))
    #     indices = [index] + random.choices(self.indices, k=3)
    #     random.shuffle(indices)

    #     for i, index in enumerate(indices):
    #         # Load and resize image to target size
    #         image, _ = self.load_image(index)
    #         image, _, _ = resize(image, self.input_size, self.augment)
    #         h, w = image.shape[:2]

    #         # Position the image in the mosaic
    #         if i == 0:  # top left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, mosaic_w), yc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(mosaic_h, yc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, mosaic_w), min(mosaic_h, yc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

    #         image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
    #         padw = x1a - x1b
    #         padh = y1a - y1b

    #         # Labels
    #         label = self.labels[index].copy()
    #         if label.size:
    #             label[:, 1:] = wh2xy(label[:, 1:], w, h, padw, padh)
    #         label4.append(label)

    #     # Concat/clip labels
    #     label4 = numpy.concatenate(label4, 0)
    #     label4[:, 1:] = numpy.clip(label4[:, 1:], 0, mosaic_w if isinstance(self.input_size, tuple) else 2 * self.input_size)

    #     # Augment
    #     image4, label4 = random_perspective(image4, label4, params, border)

    #     # Resize mosaic to input_size
    #     image4, ratio, pad = resize(image4, self.input_size, self.augment)
    #     if label4.size:
    #         label4[:, 1:] = wh2xy(label4[:, 1:], image4.shape[1], image4.shape[0], pad[0], pad[1])

    #     return image4, label4




    def load_mosaic(self, index, params):
        label4 = []
        if isinstance(self.input_size, tuple):
            target_h, target_w = self.input_size
            mosaic_h, mosaic_w = 2 * target_h, 2 * target_w
            # Separate border values for height and width
            border_h = -target_h // 2
            border_w = -target_w // 2
        else:
            mosaic_h = mosaic_w = 2 * self.input_size
            border_h = border_w = -self.input_size // 2

        image4 = numpy.full((mosaic_h, mosaic_w, 3), 0, dtype=numpy.uint8)
        
        # Corrected: Use width border for xc and height border for yc
        xc = int(random.uniform(-border_w, mosaic_w + border_w))
        yc = int(random.uniform(-border_h, mosaic_h + border_h))
        
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load and resize image to target size
            image, _ = self.load_image(index)
            image, _, _ = resize(image, self.input_size, self.augment)
            h, w = image.shape[:2]  # h=target_h, w=target_w after resize

            # Position the image in the mosaic
            if i == 0:  # top left
                x1a = max(xc - w, 0)
                y1a = max(yc - h, 0)
                x2a = xc
                y2a = yc
                x1b = w - (x2a - x1a)
                y1b = h - (y2a - y1a)
                x2b = w
                y2b = h
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - h, 0)
                x2a = min(xc + w, mosaic_w)
                y2a = yc
                x1b = 0
                y1b = h - (y2a - y1a)
                x2b = min(w, x2a - x1a)
                y2b = h
            elif i == 2:  # bottom left
                x1a = max(xc - w, 0)
                y1a = yc
                x2a = xc
                y2a = min(mosaic_h, yc + h)
                x1b = w - (x2a - x1a)
                y1b = 0
                x2b = w
                y2b = min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + w, mosaic_w)
                y2a = min(mosaic_h, yc + h)
                x1b = 0
                y1b = 0
                x2b = min(w, x2a - x1a)
                y2b = min(y2a - y1a, h)

            # Ensure valid region before slicing
            if y2a > y1a and x2a > x1a:
                image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            else:
                continue  # Skip invalid placements

            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], w, h, padw, padh)
            label4.append(label)

        # Concat/clip labels
        label4 = numpy.concatenate(label4, 0) if label4 else numpy.zeros((0, 5), dtype=numpy.float32)
        # Clip labels to mosaic dimensions
        label4[:, 1::2] = numpy.clip(label4[:, 1::2], 0, mosaic_w)  # x coordinates
        label4[:, 2::2] = numpy.clip(label4[:, 2::2], 0, mosaic_h)  # y coordinates
import math
import os
import random

import cv2
import numpy
import torch
from PIL import Image
from torch.utils import data

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.input_size = input_size
        self.num_classes = len(params['names'])

        # Read labels
        cache = self.load_label(filenames)
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)
        self.filenames = list(cache.keys())  # update
        self.n = len(shapes)  # number of samples
        self.indices = range(self.n)
        # Albumentations (optional, only used if package is installed)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]

        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            shapes = None
            # Load MOSAIC
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)

                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image
            image, shape = self.load_image(index)
            h, w = image.shape[:2]

            # Resize
            image, ratio, pad = resize(image, self.input_size, self.augment)
            shapes = shape, ((h / shape[0], w / shape[1]), pad)  # for COCO mAP rescaling

            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            if self.augment:
                image, label = random_perspective(image, label, params)
        nl = len(label)  # number of labels
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])

        if self.augment:
            # Albumentations
            image, label = self.albumentations(image, label)
            nl = len(label)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = numpy.flipud(image)
                if nl:
                    label[:, 2] = 1 - label[:, 2]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = numpy.fliplr(image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)

        # Convert HWC to CHW, BGR to RGB
        sample = image.transpose((2, 0, 1))[::-1]
        sample = numpy.ascontiguousarray(sample)

        return torch.from_numpy(sample), target, shapes

    def __len__(self):
        return len(self.filenames)

    def load_image(self, i):
        # Load original image without resizing
        image = cv2.imread(self.filenames[i])
        return image, image.shape[:2]

    # def load_mosaic(self, index, params):
    #     label4 = []
    #     if isinstance(self.input_size, tuple):
    #         target_h, target_w = self.input_size
    #         mosaic_h, mosaic_w = 2 * target_h, 2 * target_w
    #         border = (-target_h // 2, -target_w // 2)
    #     else:
    #         mosaic_h = mosaic_w = 2 * self.input_size
    #         border = [-self.input_size // 2, -self.input_size // 2]

    #     image4 = numpy.full((mosaic_h, mosaic_w, 3), 0, dtype=numpy.uint8)
    #     xc = int(random.uniform(-border[0], mosaic_w + border[1]))
    #     yc = int(random.uniform(-border[0], mosaic_h + border[1]))
    #     indices = [index] + random.choices(self.indices, k=3)
    #     random.shuffle(indices)

    #     for i, index in enumerate(indices):
    #         # Load and resize image to target size
    #         image, _ = self.load_image(index)
    #         image, _, _ = resize(image, self.input_size, self.augment)
    #         h, w = image.shape[:2]

    #         # Position the image in the mosaic
    #         if i == 0:  # top left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
    #         elif i == 1:  # top right
    #             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, mosaic_w), yc
    #             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
    #         elif i == 2:  # bottom left
    #             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(mosaic_h, yc + h)
    #             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
    #         elif i == 3:  # bottom right
    #             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, mosaic_w), min(mosaic_h, yc + h)
    #             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

    #         image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
    #         padw = x1a - x1b
    #         padh = y1a - y1b

    #         # Labels
    #         label = self.labels[index].copy()
    #         if label.size:
    #             label[:, 1:] = wh2xy(label[:, 1:], w, h, padw, padh)
    #         label4.append(label)

    #     # Concat/clip labels
    #     label4 = numpy.concatenate(label4, 0)
    #     label4[:, 1:] = numpy.clip(label4[:, 1:], 0, mosaic_w if isinstance(self.input_size, tuple) else 2 * self.input_size)

    #     # Augment
    #     image4, label4 = random_perspective(image4, label4, params, border)

    #     # Resize mosaic to input_size
    #     image4, ratio, pad = resize(image4, self.input_size, self.augment)
    #     if label4.size:
    #         label4[:, 1:] = wh2xy(label4[:, 1:], image4.shape[1], image4.shape[0], pad[0], pad[1])

    #     return image4, label4




    def load_mosaic(self, index, params):
        label4 = []
        if isinstance(self.input_size, tuple):
            target_h, target_w = self.input_size
            mosaic_h, mosaic_w = 2 * target_h, 2 * target_w
            # Separate border values for height and width
            border_h = -target_h // 2
            border_w = -target_w // 2
        else:
            mosaic_h = mosaic_w = 2 * self.input_size
            border_h = border_w = -self.input_size // 2

        image4 = numpy.full((mosaic_h, mosaic_w, 3), 0, dtype=numpy.uint8)
        
        # Corrected: Use width border for xc and height border for yc
        xc = int(random.uniform(-border_w, mosaic_w + border_w))
        yc = int(random.uniform(-border_h, mosaic_h + border_h))
        
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)

        for i, index in enumerate(indices):
            # Load and resize image to target size
            image, _ = self.load_image(index)
            image, _, _ = resize(image, self.input_size, self.augment)
            h, w = image.shape[:2]  # h=target_h, w=target_w after resize

            # Position the image in the mosaic
            if i == 0:  # top left
                x1a = max(xc - w, 0)
                y1a = max(yc - h, 0)
                x2a = xc
                y2a = yc
                x1b = w - (x2a - x1a)
                y1b = h - (y2a - y1a)
                x2b = w
                y2b = h
            elif i == 1:  # top right
                x1a = xc
                y1a = max(yc - h, 0)
                x2a = min(xc + w, mosaic_w)
                y2a = yc
                x1b = 0
                y1b = h - (y2a - y1a)
                x2b = min(w, x2a - x1a)
                y2b = h
            elif i == 2:  # bottom left
                x1a = max(xc - w, 0)
                y1a = yc
                x2a = xc
                y2a = min(mosaic_h, yc + h)
                x1b = w - (x2a - x1a)
                y1b = 0
                x2b = w
                y2b = min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a = xc
                y1a = yc
                x2a = min(xc + w, mosaic_w)
                y2a = min(mosaic_h, yc + h)
                x1b = 0
                y1b = 0
                x2b = min(w, x2a - x1a)
                y2b = min(y2a - y1a, h)

            # Ensure valid region before slicing
            if y2a > y1a and x2a > x1a:
                image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            else:
                continue  # Skip invalid placements

            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], w, h, padw, padh)
            label4.append(label)

        # Concat/clip labels
        label4 = numpy.concatenate(label4, 0) if label4 else numpy.zeros((0, 5), dtype=numpy.float32)
        # Clip x and y coordinates separately
        label4[:, 1::2] = numpy.clip(label4[:, 1::2], 0, mosaic_w)  # x coordinates
        label4[:, 2::2] = numpy.clip(label4[:, 2::2], 0, mosaic_h)  # y coordinates

        # Augment
        image4, label4 = random_perspective(image4, label4, params, (border_h, border_w))

        # Resize mosaic to input_size
        image4, ratio, pad = resize(image4, self.input_size, self.augment)
        if label4.size:
            label4[:, 1:] = wh2xy(label4[:, 1:], image4.shape[1], image4.shape[0], pad[0], pad[1])

        return image4, label4

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

    def load_label(self, filenames):
        path = f'{os.path.dirname(filenames[0])}.cache'
        if os.path.exists(path):
            return torch.load(path)
        x = {}
        for filename in filenames:
            try:
                # verify images
                with open(filename, 'rb') as f:
                    image = Image.open(f)
                    image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, 'labels require 5 columns'
                        assert (label >= 0).all(), 'negative label values'
                        assert (label[:, 1:] <= 1).all(), 'non-normalized coordinates'
                        
                        # Validate class IDs
                        invalid_classes = label[label[:, 0] >= self.num_classes]
                        if len(invalid_classes) > 0:
                            print(f"Warning: Found invalid class IDs in {filename}")
                            print("Invalid class IDs:", invalid_classes[:, 0])
                            # Filter out invalid classes
                            label = label[label[:, 0] < self.num_classes]
                            
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
                if filename:
                    x[filename] = [label, shape]
            except FileNotFoundError:
                pass
        torch.save(x, path)
        return x


def resize(image, input_size, augment):
    shape = image.shape[:2]  # current shape [height, width]
    
    if isinstance(input_size, (tuple, list)) and input_size[0] != input_size[1]:
        target_h, target_w = input_size
        # Always maintain aspect ratio with rectangular inputs
        scale = min(target_h / shape[0], target_w / shape[1])
        new_h = int(round(shape[0] * scale))
        new_w = int(round(shape[1] * scale))
        
        # Resize with preserved aspect ratio
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Add gray borders to reach target size
        dh = target_h - new_h
        dw = target_w - new_w
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        image_resized = cv2.copyMakeBorder(image_resized, top, bottom, left, right, 
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        ratio = (scale, scale)
        pad = (left, top)
        return image_resized, ratio, pad
    else:
        # Handle square input_size
        input_size = input_size[0] if isinstance(input_size, (tuple, list)) else input_size
        r = min(input_size / shape[0], input_size / shape[1])
        if not augment:
            r = min(r, 1.0)
        new_size = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        image_resized = cv2.resize(image, new_size, interpolation=resample() if augment else cv2.INTER_LINEAR)
        dw = input_size - new_size[0]
        dh = input_size - new_size[1]
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        image_resized = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image_resized, (r, r), (dw // 2, dh // 2)


def wh2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    # Convert nx4 boxes
    # from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def xy2wh(x, w=640, h=640):
    # warning: inplace clip
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - 1E-3)  # x1, x2
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - 1E-3)  # y1, y2

    # Convert nx4 boxes
    # from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


def augment_hsv(image, params):
    # HSV color-space augmentation
    h = params['hsv_h']
    s = params['hsv_s']
    v = params['hsv_v']

    r = numpy.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)


def random_perspective(samples, targets, params, border=(0, 0)):
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = numpy.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)

    # Rotation and Scale
    rotate = numpy.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = numpy.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        samples = cv2.warpAffine(samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

        # filter candidates
        indices = candidates(box1=targets[:, 1:5].T * s, box2=new.T)
        targets = targets[indices]
        targets[:, 1:5] = new[indices]

    return samples, targets


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    alpha = numpy.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as album

            transforms = [album.Blur(p=0.01),
                          album.CLAHE(p=0.01),
                          album.ToGray(p=0.01),
                          album.MedianBlur(p=0.01)]
            self.transform = album.Compose(transforms,
                                           album.BboxParams('yolo', ['class_labels']))

        except ImportError:  # package not installed, skip
            pass

    def __call__(self, image, label):
        if self.transform:
            x = self.transform(image=image,
                               bboxes=label[:, 1:],
                               class_labels=label[:, 0])
            image = x['image']
            label = numpy.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image, label
