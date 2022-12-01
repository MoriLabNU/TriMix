import torchvision.transforms.functional as F
import nibabel as nib
import os
import pandas as pd
import glob
import re
import shutil
import random
import numpy as np
import torch

from torch.nn import functional as F
from medpy.metric.binary import dc
from skimage import transform


def makefolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def save_nii(img_path, data, affine, header):
    nimg = nib.Nifti1Image(data, affine=affine, header=header)
    nimg.to_filename(img_path)


def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


@torch.no_grad()
def infer(model, criterion, dataloader_dict, device):
    model.eval()
    criterion.eval()

    test_folder = "/data/zhangke/datasets/scribble_MSCMR/val/images/"
    label_folder = "/data/zhangke/datasets/scribble_MSCMR/val/labels/"
    output_folder = "/data/zhangke/datasets/scribble_MSCMR/self_MSCMR/"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index]

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))

        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:, :, slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               anti_aliasing=True,
                                               mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)

        predictions = []

        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]
            nx = 212
            ny = 212
            x, y = img_slice.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            # Crop section of image for prediction
            if x > nx and y > ny:
                slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
            else:
                slice_cropped = np.zeros((nx, ny))
                if x <= nx and y > ny:
                    slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
                elif x > nx and y <= ny:
                    slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

            img_slice = slice_cropped
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1, 1, nx, ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.to(device)
            img_slice = img_slice.float()

            tasks = dataloader_dict.keys()
            task = random.sample(tasks, 1)[0]

            outputs = model(img_slice, task)

            softmax_out = outputs["pred_masks"]
            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0, ...])

            slice_predictions = np.zeros((4, x, y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:, x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, :, y_s:y_s + ny] = prediction_cropped[:, x_c:x_c + x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:, x_s:x_s + nx, :] = prediction_cropped[:, :, y_c:y_c + y]
                else:
                    slice_predictions[:, :, :] = prediction_cropped[:, x_c:x_c + x, y_c:y_c + y]
            prediction = transform.resize(slice_predictions,
                                          (4, mask.shape[0], mask.shape[1]),
                                          order=1,
                                          preserve_range=True,
                                          anti_aliasing=True,
                                          mode='constant')
            prediction = np.uint8(np.argmax(prediction, axis=0))
            # prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)

    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    # filenames_gt = [f for f in filenames_gt if "01" in f]
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    # filenames_pred = [f for f in filenames_pred if "01" in f]
    file_names = []
    structure_names = []
    # measures per structure:
    dices_list = []
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))
        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        # calculate measures for each structure
        for struc in [3, 1, 2]:
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(
                    gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(output_folder, "stats.csv")
    df.to_csv(csv_path)
    return df


@torch.no_grad()
def infer_trimix_single(model, snapshot_path, task):
    model.eval()

    if task == 'val':
        test_folder = "../dataset/val/images/"
        label_folder = "../dataset/val/labels/"
        output_folder = snapshot_path + '/val/'
    elif task == 'test':
        test_folder = "../dataset/TestSet/images/"
        label_folder = "../dataset/TestSet/labels/"
        output_folder = snapshot_path + '/test/'
    else:
        raise Exception

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index]

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))

        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:, :, slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               anti_aliasing=True,
                                               mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)

        predictions = []

        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]
            nx = 212
            ny = 212
            x, y = img_slice.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            # Crop section of image for prediction
            if x > nx and y > ny:
                slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
            else:
                slice_cropped = np.zeros((nx, ny))
                if x <= nx and y > ny:
                    slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
                elif x > nx and y <= ny:
                    slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

            img_slice = slice_cropped
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1, 1, nx, ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.cuda()
            img_slice = img_slice.float()

            outputs = model(img_slice)

            softmax_out = F.softmax(outputs, dim=1)
            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0, ...])

            slice_predictions = np.zeros((4, x, y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:, x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, :, y_s:y_s + ny] = prediction_cropped[:, x_c:x_c + x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:, x_s:x_s + nx, :] = prediction_cropped[:, :, y_c:y_c + y]
                else:
                    slice_predictions[:, :, :] = prediction_cropped[:, x_c:x_c + x, y_c:y_c + y]
            prediction = transform.resize(slice_predictions,
                                          (4, mask.shape[0], mask.shape[1]),
                                          order=1,
                                          preserve_range=True,
                                          anti_aliasing=True,
                                          mode='constant')
            prediction = np.uint8(np.argmax(prediction, axis=0))
            # prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)

    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    # filenames_gt = [f for f in filenames_gt if "01" in f]
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    # filenames_pred = [f for f in filenames_pred if "01" in f]
    file_names = []
    structure_names = []
    # measures per structure:
    dices_list = []
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))
        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        # calculate measures for each structure
        for struc in [3, 1, 2]:
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(
                    gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(output_folder, "stats.csv")
    df.to_csv(csv_path)

    return np.mean(dices_list)


@torch.no_grad()
def infer_trimix_ensemble(model1, model2, model3, snapshot_path, task):
    model1.eval()
    model2.eval()
    model3.eval()

    if task == 'val':
        test_folder = "../dataset/val/images/"
        label_folder = "../dataset/val/labels/"
        output_folder = snapshot_path + '/val/'
    elif task == 'test':
        test_folder = "../dataset/TestSet/images/"
        label_folder = "../dataset/TestSet/labels/"
        output_folder = snapshot_path + '/test/'
    else:
        raise Exception

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    makefolder(output_folder)

    target_resolution = (1.36719, 1.36719)

    test_files = sorted(os.listdir(test_folder))
    label_files = sorted(os.listdir(label_folder))
    assert len(test_files) == len(label_files)

    # read_image
    for file_index in range(len(test_files)):
        test_file = test_files[file_index]

        label_file = label_files[file_index]
        file_mask = os.path.join(label_folder, label_file)
        mask_dat = load_nii(file_mask)
        mask = mask_dat[0]

        img_path = os.path.join(test_folder, test_file)
        img_dat = load_nii(img_path)
        img = img_dat[0].copy()

        pixel_size = (img_dat[2].structarr['pixdim'][1], img_dat[2].structarr['pixdim'][2])
        scale_vector = (pixel_size[0] / target_resolution[0],
                        pixel_size[1] / target_resolution[1])

        img = img.astype(np.float32)
        img = np.divide((img - np.mean(img)), np.std(img))

        slice_rescaleds = []
        for slice_index in range(img.shape[2]):
            img_slice = np.squeeze(img[:, :, slice_index])
            slice_rescaled = transform.rescale(img_slice,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               anti_aliasing=True,
                                               mode='constant')
            slice_rescaleds.append(slice_rescaled)
        img = np.stack(slice_rescaleds, axis=2)

        predictions = []

        for slice_index in range(img.shape[2]):
            img_slice = img[:, :, slice_index]
            nx = 212
            ny = 212
            x, y = img_slice.shape
            x_s = (x - nx) // 2
            y_s = (y - ny) // 2
            x_c = (nx - x) // 2
            y_c = (ny - y) // 2
            # Crop section of image for prediction
            if x > nx and y > ny:
                slice_cropped = img_slice[x_s:x_s + nx, y_s:y_s + ny]
            else:
                slice_cropped = np.zeros((nx, ny))
                if x <= nx and y > ny:
                    slice_cropped[x_c:x_c + x, :] = img_slice[:, y_s:y_s + ny]
                elif x > nx and y <= ny:
                    slice_cropped[:, y_c:y_c + y] = img_slice[x_s:x_s + nx, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img_slice[:, :]

            img_slice = slice_cropped
            img_slice = np.divide((slice_cropped - np.mean(slice_cropped)), np.std(slice_cropped))
            img_slice = np.reshape(img_slice, (1, 1, nx, ny))

            img_slice = torch.from_numpy(img_slice)
            img_slice = img_slice.cuda()
            img_slice = img_slice.float()

            outputs1 = model1(img_slice)
            outputs2 = model2(img_slice)
            outputs3 = model3(img_slice)

            softmax_out = (F.softmax(outputs1, dim=1) + F.softmax(outputs2, dim=1)
                           + F.softmax(outputs3, dim=1)) / 3

            softmax_out = softmax_out.detach().cpu().numpy()
            prediction_cropped = np.squeeze(softmax_out[0, ...])

            slice_predictions = np.zeros((4, x, y))
            # insert cropped region into original image again
            if x > nx and y > ny:
                slice_predictions[:, x_s:x_s + nx, y_s:y_s + ny] = prediction_cropped
            else:
                if x <= nx and y > ny:
                    slice_predictions[:, :, y_s:y_s + ny] = prediction_cropped[:, x_c:x_c + x, :]
                elif x > nx and y <= ny:
                    slice_predictions[:, x_s:x_s + nx, :] = prediction_cropped[:, :, y_c:y_c + y]
                else:
                    slice_predictions[:, :, :] = prediction_cropped[:, x_c:x_c + x, y_c:y_c + y]
            prediction = transform.resize(slice_predictions,
                                          (4, mask.shape[0], mask.shape[1]),
                                          order=1,
                                          preserve_range=True,
                                          anti_aliasing=True,
                                          mode='constant')
            prediction = np.uint8(np.argmax(prediction, axis=0))
            # prediction = keep_largest_connected_components(prediction)
            predictions.append(prediction)
        prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1, 2, 0))
        dir_pred = os.path.join(output_folder, "predictions")
        makefolder(dir_pred)
        out_file_name = os.path.join(dir_pred, label_file)
        out_affine = mask_dat[1]
        out_header = mask_dat[2]

        save_nii(out_file_name, prediction_arr, out_affine, out_header)

        dir_gt = os.path.join(output_folder, "masks")
        makefolder(dir_gt)
        mask_file_name = os.path.join(dir_gt, label_file)
        save_nii(mask_file_name, mask_dat[0], out_affine, out_header)

    filenames_gt = sorted(glob.glob(os.path.join(dir_gt, '*')), key=natural_order)
    # filenames_gt = [f for f in filenames_gt if "01" in f]
    filenames_pred = sorted(glob.glob(os.path.join(dir_pred, '*')), key=natural_order)
    # filenames_pred = [f for f in filenames_pred if "01" in f]
    file_names = []
    structure_names = []
    # measures per structure:
    dices_list = []
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    count = 0
    for p_gt, p_pred in zip(filenames_gt, filenames_pred):
        if os.path.basename(p_gt) != os.path.basename(p_pred):
            raise ValueError("The two files don't have the same name"
                             " {}, {}.".format(os.path.basename(p_gt),
                                               os.path.basename(p_pred)))
        # load ground truth and prediction
        gt, _, header = load_nii(p_gt)
        pred, _, _ = load_nii(p_pred)
        zooms = header.get_zooms()
        # calculate measures for each structure
        for struc in [3, 1, 2]:
            gt_binary = (gt == struc) * 1
            pred_binary = (pred == struc) * 1
            if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                dices_list.append(1)
            elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(
                    gt_binary) > 0:
                dices_list.append(0)
                count += 1
            else:
                dices_list.append(dc(gt_binary, pred_binary))
            file_names.append(os.path.basename(p_pred))
            structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'struc': structure_names, 'filename': file_names})
    csv_path = os.path.join(output_folder, "stats.csv")
    df.to_csv(csv_path)

    return np.mean(dices_list)
