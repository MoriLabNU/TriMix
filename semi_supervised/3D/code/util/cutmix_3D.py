import torch
import numpy as np


class Cutmix_3D(object):
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

    def __call__(self, sample):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W, D).
        Returns:
            Tensor: Image with n_holes of dimension length x length x length cut out of it.
        """
        # img 1,h,w,d  label 1,h,w,d

        image, label = sample['image'], sample['label']

        h = image.size(1)
        w = image.size(2)
        d = image.size(3)
        n_masks = image.size(0)

        mask_props = np.random.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_holes))
        if self.random_aspect_ratio:
            # ugly but work
            y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, self.n_holes)) * np.log(mask_props))
            inter = mask_props / y_props
            x_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, self.n_holes)) * np.log(inter))
            z_props = inter / x_props
        else:
            # cubic crop was not validated
            z_props = y_props = x_props = np.cbrt(mask_props)

        fac = np.cbrt(1.0 / self.n_holes)
        y_props *= fac
        x_props *= fac
        z_props *= fac

        sizes = np.round(np.stack([y_props, x_props, z_props], axis=2) * np.array((h, w, d))[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array((h, w, d)) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=-1)
        else:
            centres = np.round(np.array((h, w, d)) * np.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=-1)

        masks = np.zeros_like(image)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, z0, y1, x1, z1 in sample_rectangles:
                masks[i, int(y0):int(y1), int(x0):int(x1), int(z0):int(z1)] = 1

        masks = torch.from_numpy(masks)
        return {'image': image, 'label': label, 'mask': masks}
