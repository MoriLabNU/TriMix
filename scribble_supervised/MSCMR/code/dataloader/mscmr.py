import numpy as np
import nibabel as nib

from torch.utils import data
from pathlib import Path

from dataloader import transforms as T


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


class mscmrSeg(data.Dataset):
    def __init__(self, img_folder, lab_folder, transforms, cutmix=None):
        self._transforms = transforms
        self.cutmix = cutmix

        img_paths = list(img_folder.iterdir())
        lab_paths = list(lab_folder.iterdir())

        self.examples = []
        self.img_dict = {}
        self.lab_dict = {}

        for img_path, lab_path in zip(sorted(img_paths), sorted(lab_paths)):
            img = self.read_image(str(img_path))
            img_name = img_path.stem
            self.img_dict.update({img_name: img})
            lab = self.read_label(str(lab_path))
            lab_name = lab_path.stem
            print(img_name, lab_name)
            self.lab_dict.update({lab_name: lab})

            assert img[0].shape[2] == lab[0].shape[2]
            self.examples += [(img_name, lab_name, -1, -1, slice) for slice in range(img[0].shape[2])]

    def __getitem__(self, idx):
        img_name, lab_name, Z, X, Y = self.examples[idx]

        if Z != -1:
            img = self.img_dict[img_name][Z, :, :]
            lab = self.lab_dict[lab_name][Z, :, :]
        elif X != -1:
            img = self.img_dict[img_name][:, X, :]
            lab = self.lab_dict[lab_name][:, X, :]
        elif Y != -1:
            img = self.img_dict[img_name][0][:, :, Y]
            scale_vector_img = self.img_dict[img_name][1]
            lab = self.lab_dict[lab_name][0][:, :, Y]
            scale_vector_lab = self.lab_dict[lab_name][1]
        else:
            raise ValueError(f'invalid index: ({Z}, {X}, {Y})')
        img = np.expand_dims(img, 0)
        lab = np.expand_dims(lab, 0)
        target = {'name': lab_name, 'slice': (Z, X, Y), 'masks': lab, 'orig_size': lab.shape}

        img, target = self._transforms([img, scale_vector_img], [target, scale_vector_lab])
        if self.cutmix is not None:
            img, target, masks = self.cutmix(img, target)
            return img, target['masks'], masks
        else:
            return img, target['masks']

    def read_image(self, img_path):
        img_dat = load_nii(img_path)
        img = img_dat[0]
        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])
        img = img.astype(np.float32)
        return [(img - img.mean()) / img.std(), scale_vector]

    def read_label(self, lab_path):
        lab_dat = load_nii(lab_path)
        lab = lab_dat[0]
        pixel_size = (lab_dat[2].structarr['pixdim'][1], lab_dat[2].structarr['pixdim'][2])
        target_resolution = (1.36719, 1.36719)
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        return [lab, scale_vector]

    def __len__(self):
        return len(self.examples)


def CutMix(prop_range=0.2):
    cutmix = T.Compose_cutmix([T.Cutmix(prop_range=prop_range)])
    return cutmix


def make_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(),
    ])

    return T.Compose([
        T.Rescale(),
        T.RandomHorizontalFlip(),
        T.RandomRotate((0, 360)),
        T.PadOrCropToSize([212, 212]),
        normalize
    ])


def build():
    # set your data path
    root = Path('../dataset/')
    assert root.exists(), f'provided MSCMR path {root} does not exist'
    PATHS = root / "train" / "images", root / "train" / "labels"

    img_folder, lab_folder = PATHS
    img_task, lab_task = img_folder, lab_folder
    dataset = mscmrSeg(img_task, lab_task, transforms=make_transforms(), cutmix=CutMix())

    return dataset
