import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
import PIL
from PIL import Image
from skimage import transform


def hflip(image, target):
    flipped_image = image
    if random.random() < 0.5:
        flipped_image = np.flip(flipped_image, (1))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask, (1))
    if random.random() < 0.5:
        flipped_image = np.flip(flipped_image, (2))
        target = target.copy()
        if "masks" in target:
            mask = target['masks']
            target['masks'] = np.flip(mask, (2))
    rotate_choice = int(random.random() * 4)
    flipped_image = np.rot90(flipped_image, k=rotate_choice, axes=(1, 2))
    if "masks" in target:
        mask = target['masks']
        target['masks'] = np.rot90(mask, k=rotate_choice, axes=(1, 2))
    return flipped_image, target


def crop(image, target, region):
    i, j, h, w = region
    cropped_image = image[:, i:i + h, j:j + w]

    # should we do something wrt the original size?
    target["size"] = [h, w]

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        mask = target["masks"]
        target['masks'] = mask[:, i:i + h, j:j + w]

    return cropped_image, target


def pad(image, target, region):
    z, x, y = image.shape
    nx, ny = region
    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    # print("source shape:",(z,x,y), "target shape:",(nx,ny))

    if x > nx and y > ny:
        slice_padded = image[:, x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_padded = np.zeros((z, nx, ny), dtype=np.float32)
        if x <= nx and y > ny:
            slice_padded[:, x_c:x_c + x, :] = image[:, :, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_padded[:, :, y_c:y_c + y] = image[:, x_s:x_s + nx, :]
        else:
            slice_padded[:, x_c:x_c + x, y_c:y_c + y] = image[:, :, :]

    if "masks" in target:
        mask = target["masks"]
        if x > nx and y > ny:
            mask_padded = mask[:, x_s:x_s + nx, y_s:y_s + ny]
        else:
            mask_padded = np.zeros((z, nx, ny), dtype=np.float32)
            if x <= nx and y > ny:
                mask_padded[:, x_c:x_c + x, :] = mask[:, :, y_s:y_s + ny]
            elif x > nx and y <= ny:
                mask_padded[:, :, y_c:y_c + y] = mask[:, x_s:x_s + nx, :]
            else:
                mask_padded[:, x_c:x_c + x, y_c:y_c + y] = mask[:, :, :]

        target['masks'] = mask_padded

    return slice_padded, target


def resize(image, target, size):
    min_scale = size[0]
    max_scale = size[1]
    img_width = image.shape[1]
    img_height = image.shape[2]
    target_scale = random.uniform(min_scale, max_scale)
    rescaled_width = int(target_scale * img_width)
    rescaled_height = int(target_scale * img_height)
    # random.randint(min_size, max_size)
    rescaled_size = [rescaled_width, rescaled_height]
    image = image.copy()
    image = torch.from_numpy(image)
    rescaled_image = F.resize(image, rescaled_size, interpolation=PIL.Image.NEAREST)
    rescaled_image = rescaled_image.numpy()

    if target is None:
        return rescaled_image, None

    target = target.copy()
    w = rescaled_width
    h = rescaled_height
    target["size"] = torch.tensor([w, h])

    if "masks" in target:
        mask = target['masks']
        # interpolate_mask = mask[:, None].copy()
        interpolate_mask = mask.copy()
        interpolate_mask = torch.from_numpy(interpolate_mask)
        mask = F.resize(interpolate_mask, rescaled_size, interpolation=PIL.Image.NEAREST)
        # mask = interpolate(interpolate_mask.float(), size, mode="nearest")[:, 0] > 0.5
        mask = mask.numpy()
        target['masks'] = mask

    return rescaled_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)


class Rescale(object):
    def __init__(self):
        pass

    def __call__(self, imgs, targets=None):
        img = imgs[0]
        scale_vector_img = imgs[1]
        target = targets[0]
        scale_vector_target = targets[1]
        img = transform.rescale(img[0, :, :],
                                scale_vector_img,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        img = np.expand_dims(img, axis=0)
        if "masks" in target:
            mask = target['masks']
            mask = transform.rescale(mask[0, :, :],
                                     scale_vector_target,
                                     order=0,
                                     preserve_range=True,
                                     multichannel=False,
                                     anti_aliasing=False,
                                     mode='constant')
            mask = np.expand_dims(mask, axis=0)
            target['masks'] = mask
        return img, target


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class PadOrCropToSize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        crop_height, crop_width = self.size
        padded_img, padded_target = pad(img, target, (crop_height, crop_width))
        return padded_img, padded_target


class RandomRotate(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def __call__(self, img, target):
        angle = self.get_params(self.degrees)
        img = img.copy()
        rotated_img = F.rotate(torch.from_numpy(img), angle, PIL.Image.NEAREST, self.expand, self.center)
        rotated_img = rotated_img.numpy()
        #  if "masks" in target:
        mask = target['masks']
        mask = mask.copy()
        rotated_mask = F.rotate(torch.from_numpy(mask), angle, PIL.Image.NEAREST, self.expand, self.center)
        rotated_mask = rotated_mask.numpy()
        target["masks"] = rotated_mask
        return rotated_img, target


class RandomColorJitter(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        RGB_img = np.repeat(img, 3, axis=0)
        RGB_img = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0)(torch.from_numpy(RGB_img))
        gray_img = T.Grayscale(num_output_channels=1)(RGB_img)
        return gray_img, target


class CenterRandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        min_scale = self.size[0]
        max_scale = self.size[1]
        image_width = img.shape[1]
        image_height = img.shape[2]
        if random.random() < 0.7:
            target_scale = random.uniform(min_scale, max_scale)
        else:
            target_scale = 1
        crop_height = int(target_scale * image_height)
        crop_width = int(target_scale * image_width)
        crop_top = max(0, image_height - crop_height)
        crop_left = max(0, image_width - crop_width)
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class ToTensor(object):
    def __call__(self, img, target):
        for k, v in target.items():
            if not isinstance(v, str):
                if torch.is_tensor(v) or isinstance(v, (list, tuple)):
                    if torch.is_tensor(v):
                        pass
                    else:
                        target[k] = torch.tensor(v).type(torch.LongTensor)
                else:
                    v = v.copy()
                    target[k] = torch.tensor(v).type(torch.LongTensor)
        if not torch.is_tensor(img):
            img = img.copy()
            img = torch.from_numpy(img)
        return img, target
        # return torch.from_numpy(img), target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        if self.mean is None:
            self.mean = image.mean()
        if self.std is None:
            self.std = image.std()
        image = (image - self.mean) / self.std
        if target is None:
            return image, None

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Compose_cutmix(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target, mask = t(image, target)
        return image, target, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Cutmix(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, prop_range, n_holes=1, random_aspect_ratio=True, within_bounds=True):
        self.n_holes = n_holes
        if isinstance(prop_range, float):
            self.prop_range = (prop_range, prop_range)
        self.random_aspect_ratio = random_aspect_ratio
        self.within_bounds = within_bounds

    def __call__(self, img, target):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # img 1,h,w  label 1,h,w

        label = target["masks"]

        h = label.size(1)
        w = label.size(2)
        n_masks = label.size(0)

        mask_props = np.random.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_holes))
        if self.random_aspect_ratio:
            y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, self.n_holes)) * np.log(mask_props))
            x_props = mask_props / y_props
        else:
            y_props = x_props = np.sqrt(mask_props)

        fac = np.sqrt(1.0 / self.n_holes)
        y_props *= fac
        x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array((h, w)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array((h, w)) * np.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        masks = np.zeros_like(label)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, int(y0):int(y1), int(x0):int(x1)] = 1

        masks = torch.from_numpy(masks)

        return img, target, masks
