import itertools
import os
import random
import re
import h5py
import numpy as np
import torch

from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, num=4, labeled_type="labeled", split='train', transform=None, fold="fold1",
                 sup_type="label"):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.sup_type = sup_type
        self.transform = transform
        self.num = num
        self.labeled_type = labeled_type
        train_ids, test_ids = self._get_fold_ids(fold)
        all_labeled_ids = ["patient{:0>3}".format(
            10 * i) for i in range(1, 11)]
        if self.split == 'train':
            self.all_slices = os.listdir(
                self._base_dir + "/ACDC_training_slices")
            self.sample_list = []
            labeled_ids = [i for i in all_labeled_ids if i in train_ids]
            unlabeled_ids = [i for i in train_ids if i not in labeled_ids]
            if self.labeled_type == "labeled":
                print("Labeled patients IDs", labeled_ids)
                for ids in labeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total labeled {} samples".format(len(self.sample_list)))

                print('************************************************')
                print('labeled sample:', self.sample_list)

            else:
                print("Unlabeled patients IDs", unlabeled_ids)
                for ids in unlabeled_ids:
                    new_data_list = list(filter(lambda x: re.match(
                        '{}.*'.format(ids), x) != None, self.all_slices))
                    self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))

                print('************************************************')
                print('unlabeled sample:', self.sample_list)

        elif self.split == 'val':
            self.all_volumes = os.listdir(
                self._base_dir + "/ACDC_training_volumes")
            self.sample_list = []
            for ids in test_ids:
                new_data_list = list(filter(lambda x: re.match(
                    '{}.*'.format(ids), x) != None, self.all_volumes))
                self.sample_list.extend(new_data_list)

            print("total val {} samples".format(len(self.sample_list)))

            print('************************************************')
            print('val sample:', self.sample_list)

        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]

    def _get_fold_ids(self, fold):
        all_cases_set = ["patient{:0>3}".format(i) for i in range(1, 101)]
        fold1_testing_set = [
            "patient{:0>3}".format(i) for i in range(1, 21)]
        fold1_training_set = [
            i for i in all_cases_set if i not in fold1_testing_set]

        fold2_testing_set = [
            "patient{:0>3}".format(i) for i in range(21, 41)]
        fold2_training_set = [
            i for i in all_cases_set if i not in fold2_testing_set]

        fold3_testing_set = [
            "patient{:0>3}".format(i) for i in range(41, 61)]
        fold3_training_set = [
            i for i in all_cases_set if i not in fold3_testing_set]

        fold4_testing_set = [
            "patient{:0>3}".format(i) for i in range(61, 81)]
        fold4_training_set = [
            i for i in all_cases_set if i not in fold4_testing_set]

        fold5_testing_set = [
            "patient{:0>3}".format(i) for i in range(81, 101)]
        fold5_training_set = [
            i for i in all_cases_set if i not in fold5_testing_set]
        if fold == "fold1":
            return [fold1_training_set, fold1_testing_set]
        elif fold == "fold2":
            return [fold2_training_set, fold2_testing_set]
        elif fold == "fold3":
            return [fold3_training_set, fold3_testing_set]
        elif fold == "fold4":
            return [fold4_training_set, fold4_testing_set]
        elif fold == "fold5":
            return [fold5_training_set, fold5_testing_set]
        else:
            return "ERROR KEY"

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_slices/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/ACDC_training_volumes/{}".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            image = h5f['image'][:]
            label = h5f[self.sup_type][:]
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        else:
            image = h5f['image'][:]
            label = h5f['label'][:]
            sample = {'image': image, 'label': label}
        sample["idx"] = case.split("_")[0]
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label = random_rotate(image, label, cval=4)
            else:
                image, label = random_rotate(image, label, cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


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

    def __call__(self, sample):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # img 1,h,w  label ,h,w

        image = sample['image']
        label = sample['label']

        if len(label.shape) == 2:
            label = label.unsqueeze(0)

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

        if len(label.shape) == 2:
            sample = {'image': image, 'label': label, 'mask': masks}
        else:
            sample = {'image': image, 'label': label.squeeze(0), 'mask': masks}

        return sample
