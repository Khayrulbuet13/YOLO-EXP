import math
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'

class Dataset(data.Dataset):
    def __init__(self, filenames, input_size, params, augment):
        self.params = params
        self.mosaic = augment
        self.augment = augment
        self.num_classes = len(params['names'])
        
        # Handle input_size (int -> tuple)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_height, self.input_width = input_size

        # Read labels
        cache = self.load_label(filenames)
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.filenames = list(cache.keys())
        self.n = len(shapes)
        self.indices = range(self.n)
        self.albumentations = Albumentations()

    def __getitem__(self, index):
        index = self.indices[index]
        params = self.params
        mosaic = self.mosaic and random.random() < params['mosaic']

        if mosaic:
            # Load mosaic (modified for rectangular)
            image, label = self.load_mosaic(index, params)
            # MixUp augmentation
            if random.random() < params['mix_up']:
                index = random.choice(self.indices)
                mix_image1, mix_label1 = image, label
                mix_image2, mix_label2 = self.load_mosaic(index, params)
                image, label = mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
        else:
            # Load image (no resizing here)
            image, (orig_h, orig_w) = self.load_image(index)
            
            # Resize + pad to target input_size
            image, ratio, (pad_w, pad_h) = resize(image, (self.input_height, self.input_width), self.augment)
            
            # Adjust labels
            label = self.labels[index].copy()
            if label.size:
                # ratio is (r, r) for aspect ratio preservation
                label[:, 1:] = wh2xy(label[:, 1:], 
                                  ratio[1] * orig_w,  # scaled width = orig_w * r
                                  ratio[0] * orig_h,  # scaled height = orig_h * r
                                  pad_w, pad_h)
                
            if self.augment:
                image, label = random_perspective(image, label, params)

        # Post-processing (same as before)
        nl = len(label)
        if nl:
            label[:, 1:5] = xy2wh(label[:, 1:5], image.shape[1], image.shape[0])
        
        if self.augment:
            image, label = self.albumentations(image, label)
            nl = len(label)  # update after albumentations
            # HSV color-space
            augment_hsv(image, params)
            # Flip up-down
            if random.random() < params['flip_ud']:
                image = np.flipud(image)
                if nl:
                    label[:, 2] = 1 - label[:, 2]
            # Flip left-right
            if random.random() < params['flip_lr']:
                image = np.fliplr(image)
                if nl:
                    label[:, 1] = 1 - label[:, 1]

        target = torch.zeros((nl, 6))
        if nl:
            target[:, 1:] = torch.from_numpy(label)
        
        sample = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        sample = np.ascontiguousarray(sample)
        
        # For non-mosaic mode, return shapes for COCO mAP rescaling
        shapes = None
        if not mosaic:
            # CORRECTED: Use ratio=(r, r) instead of (orig_h/input_h, orig_w/input_w)
            shapes = (orig_h, orig_w), (ratio, (pad_w, pad_h))
            
        return torch.from_numpy(sample), target, shapes

    def load_image(self, i):
        """Load image without resizing"""
        image = cv2.imread(self.filenames[i])
        return image, image.shape[:2]  # (h, w)

    def load_mosaic(self, index, params):
        """Modified for rectangular input_size"""
        h, w = self.input_height, self.input_width
        mosaic_border = (-h // 2, -w // 2)
        image4 = np.full((2 * h, 2 * w, 3), 0, dtype=np.uint8)
        label4 = []
        
        # Center of mosaic
        xc = int(random.uniform(-mosaic_border[1], 2 * w + mosaic_border[1]))
        yc = int(random.uniform(-mosaic_border[0], 2 * h + mosaic_border[0]))
        
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        
        for i, index in enumerate(indices):
            # Load and resize image to fit in quadrant
            image, (orig_h, orig_w) = self.load_image(index)
            image, ratio, (pad_w, pad_h) = resize(image, (h, w), augment=True)
            new_h, new_w = image.shape[:2]
            
            # Place in mosaic
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), max(yc - new_h, 0), xc, yc
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - new_h, 0), min(xc + new_w, 2 * w), yc
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), min(new_w, x2a - x1a), new_h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), yc, xc, min(2 * h, yc + new_h)
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, min(new_h, y2a - y1a)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + new_w, 2 * w), min(2 * h, yc + new_h)
                x1b, y1b, x2b, y2b = 0, 0, min(new_w, x2a - x1a), min(new_h, y2a - y1a)
            
            # Paste into mosaic
            image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            
            # Adjust labels
            pad = (x1a - x1b, y1a - y1b)  # offsets
            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = wh2xy(label[:, 1:], new_w, new_h, pad[0], pad[1])
            label4.append(label)
        
        # Clip labels and apply perspective
        label4 = np.concatenate(label4, 0)
        np.clip(label4[:, 1:], 0, 2 * w, out=label4[:, 1:])
        image4, label4 = random_perspective(image4, label4, params, mosaic_border)
        return image4, label4

    def __len__(self):
        return len(self.filenames)

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
                        label = np.array(label, dtype=np.float32)
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
                            
                        _, i = np.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = np.zeros((0, 5), dtype=np.float32)
                else:
                    label = np.zeros((0, 5), dtype=np.float32)
                if filename:
                    x[filename] = [label, shape]
            except FileNotFoundError:
                pass
        torch.save(x, path)
        return x

    @staticmethod
    def collate_fn(batch):
        samples, targets, shapes = zip(*batch)
        for i, item in enumerate(targets):
            item[:, 0] = i  # add target image index
        return torch.stack(samples, 0), torch.cat(targets, 0), shapes

def resize(image, target_size, augment):
    """Resize image to target (height, width) with padding"""
    target_h, target_w = target_size
    h, w = image.shape[:2]
    r = min(target_h / h, target_w / w)
    if not augment:
        r = min(r, 1.0)
    
    new_h, new_w = int(round(h * r)), int(round(w * r))
    if (h, w) != (new_h, new_w):
        image = cv2.resize(image, (new_w, new_h), 
                          interpolation=resample() if augment else cv2.INTER_LINEAR)
    
    # Compute padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    # Add padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image, (r, r), (left, top)  # Return scaling ratio and (left, top) padding

def wh2xy(x, scaled_w, scaled_h, pad_left, pad_top):
    """Convert normalized [x, y, w, h] to absolute [x1, y1, x2, y2] with padding"""
    y = np.copy(x)
    y[:, 0] = x[:, 0] * scaled_w + pad_left  # x center -> x1
    y[:, 1] = x[:, 1] * scaled_h + pad_top   # y center -> y1
    y[:, 2] = x[:, 2] * scaled_w             # width
    y[:, 3] = x[:, 3] * scaled_h             # height
    
    y[:, 0] -= y[:, 2] / 2  # x1
    y[:, 1] -= y[:, 3] / 2  # y1
    y[:, 2] += y[:, 0]      # x2
    y[:, 3] += y[:, 1]      # y2
    return y

def xy2wh(x, img_w, img_h):
    """Convert [x1, y1, x2, y2] to normalized [x, y, w, h]"""
    x[:, [0, 2]] = x[:, [0, 2]].clip(0, img_w - 1e-3)
    x[:, [1, 3]] = x[:, [1, 3]].clip(0, img_h - 1e-3)
    
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / img_w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / img_h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / img_w        # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / img_h        # height
    return y

def resample():
    """Random interpolation method for image resizing"""
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

    r = np.random.uniform(-1, 1, 3) * [h, s, v] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = np.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = np.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed

def candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)

def random_perspective(samples, targets, params, border=(0, 0)):
    h = samples.shape[0] + border[0] * 2
    w = samples.shape[1] + border[1] * 2

    # Center
    center = np.eye(3)
    center[0, 2] = -samples.shape[1] / 2  # x translation (pixels)
    center[1, 2] = -samples.shape[0] / 2  # y translation (pixels)

    # Perspective
    perspective = np.eye(3)

    # Rotation and Scale
    rotate = np.eye(3)
    a = random.uniform(-params['degrees'], params['degrees'])
    s = random.uniform(1 - params['scale'], 1 + params['scale'])
    rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    shear = np.eye(3)
    shear[0, 1] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)
    shear[1, 0] = math.tan(random.uniform(-params['shear'], params['shear']) * math.pi / 180)

    # Translation
    translate = np.eye(3)
    translate[0, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * w
    translate[1, 2] = random.uniform(0.5 - params['translate'], 0.5 + params['translate']) * h

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translate @ shear @ rotate @ perspective @ center
    if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():  # image changed
        samples = cv2.warpAffine(samples, matrix[:2], dsize=(w, h), borderValue=(0, 0, 0))

    # Transform label coordinates
    n = len(targets)
    if n:
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

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
    alpha = np.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * alpha + image2 * (1 - alpha)).astype(np.uint8)
    label = np.concatenate((label1, label2), 0)
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
            label = np.array([[c, *b] for c, b in zip(x['class_labels'], x['bboxes'])])
        return image, label
