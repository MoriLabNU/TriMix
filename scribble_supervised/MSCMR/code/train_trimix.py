import os
import logging
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import time
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.nn.modules.loss import CrossEntropyLoss

from dataloader.mscmr import build
from network.UNet import build_UNet
from util.inference import infer_trimix_single, infer_trimix_ensemble


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='trimix', help='model_name')
    parser.add_argument('--max_epoch', type=int, default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--n_classes', type=int, default=4, help='segmentation class')
    parser.add_argument('--lamda_1', type=float, default=1, help='hyperparameter')
    parser.add_argument('--lamda_2', type=float, default=1, help='hyperparameter')
    parser.add_argument('--update_lr', action='store_true', help='whether update learning rate')
    args = parser.parse_args()

    return args


def create_model(args, ema=False):
    model = build_UNet(args)
    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def main():
    # parse arguments
    args = parser_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    snapshot_path = "../model/" + args.exp + "/"
    batch_size = args.batch_size * len(args.gpu.split(','))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log_train.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # create model
    net_1 = create_model(args)
    net_2 = create_model(args)
    net_3 = create_model(args)
    net_1.cuda()
    net_2.cuda()
    net_3.cuda()

    # parameter optimization
    optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=args.base_lr, weight_decay=1e-4)
    optimizer_2 = torch.optim.Adam(net_2.parameters(), lr=args.base_lr, weight_decay=1e-4)
    optimizer_3 = torch.optim.Adam(net_3.parameters(), lr=args.base_lr, weight_decay=1e-4)

    # generate dataloader
    logging.info('Building training dataset...')
    dataset_train_dict = build()
    logging.info('Number of training images: {}'.format(len(dataset_train_dict)))

    train_dataloader = DataLoader(dataset_train_dict, batch_size=batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)

    writer = SummaryWriter(snapshot_path + '/log')
    max_iterations = args.max_epoch * len(train_dataloader)

    # training
    start_time = time.time()

    for epoch in tqdm(range(args.max_epoch), ncols=70):
        train(args, epoch, net_1, net_2, net_3, optimizer_1, optimizer_2, optimizer_3, train_dataloader, writer,
              max_iterations)

        # validation
        if (epoch + 1) % 500 == 0:
            dice_score = infer_trimix_single(net_1, snapshot_path + '/net1/', task='val')
            logging.info('epoch: {}, average Dice score of net_1: {}'.format(epoch + 1, dice_score))
            dice_score = infer_trimix_single(net_2, snapshot_path + '/net2/', task='val')
            logging.info('epoch: {}, average Dice score of net_2: {}'.format(epoch + 1, dice_score))
            dice_score = infer_trimix_single(net_3, snapshot_path + '/net3/', task='val')
            logging.info('epoch: {}, average Dice score of net_3: {}'.format(epoch + 1, dice_score))
            dice_score = infer_trimix_ensemble(net_1, net_2, net_3, snapshot_path + '/ensemble/', task='val')
            logging.info('epoch: {}, average Dice score of ensemble: {}'.format(epoch + 1, dice_score))

    total_time = time.time() - start_time
    logging.info('training time {}'.format(total_time))

    save_mode_path = os.path.join(
        snapshot_path, 'net_1.pth')
    torch.save(net_1.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, 'net_2.pth')
    torch.save(net_2.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, 'net_3.pth')
    torch.save(net_3.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))

    writer.close()

    return 'training finishes.'


def train(args, epoch, net_1, net_2, net_3, optimizer_1, optimizer_2, optimizer_3, train_dataloader, writer,
          max_iterations):
    net_1.train()
    net_2.train()
    net_3.train()

    partial_ce_loss = CrossEntropyLoss(ignore_index=4)
    train_dataloader_iter = iter(train_dataloader)

    save_fig = True

    for step in range(len(train_dataloader)):
        i_iter = epoch * len(train_dataloader) + step

        # get labeled data
        image, scribble, cutmix_mask = train_dataloader_iter.next()
        image = image.cuda()
        cutmix_mask = cutmix_mask.cuda()
        scribble = scribble.cuda()

        ### first forward
        # net1, supervised loss, no mix
        output_1 = net_1(image)
        loss_sup_no_mix_1 = partial_ce_loss(output_1, scribble[:, 0].long())

        # net2, supervised loss, no mix
        output_2 = net_2(image)
        loss_sup_no_mix_2 = partial_ce_loss(output_2, scribble[:, 0].long())

        # net3, supervised loss, no mix
        output_3 = net_3(image)
        loss_sup_no_mix_3 = partial_ce_loss(output_3, scribble[:, 0].long())

        ### perform cutmix
        rand_index_1 = torch.randperm(image.size()[0]).cuda()
        rand_index_2 = torch.randperm(image.size()[0]).cuda()

        # shuffle
        image_shuffle_1 = image
        scribble_shuffle_1 = scribble
        cutmix_mask_shuffle_1 = cutmix_mask

        image_shuffle_2 = image[rand_index_1]
        scribble_shuffle_2 = scribble[rand_index_1]
        cutmix_mask_shuffle_2 = cutmix_mask[rand_index_1]

        image_shuffle_3 = image[rand_index_2]
        scribble_shuffle_3 = scribble[rand_index_2]
        cutmix_mask_shuffle_3 = cutmix_mask[rand_index_2]

        with torch.no_grad():
            output_unsup_1 = output_1
            output_unsup_2 = output_2[rand_index_1]
            output_unsup_3 = output_3[rand_index_2]

        output_unsup_soft_1 = F.softmax(output_unsup_1, dim=1)
        output_unsup_soft_2 = F.softmax(output_unsup_2, dim=1)
        output_unsup_soft_3 = F.softmax(output_unsup_3, dim=1)

        # throw dice to generate strong augmented image, pseudo label and mixed scribble annotation.
        if random.random() > 0.5:
            # for net1
            img_mixed_for_net_1 = cutmix_mask_shuffle_2 * image_shuffle_2 + (
                    1 - cutmix_mask_shuffle_2) * image_shuffle_3
            pseudo_label_for_net_1 = cutmix_mask_shuffle_2 * output_unsup_soft_2 + (
                    1 - cutmix_mask_shuffle_2) * output_unsup_soft_3
            scribble_mixed_for_net_1 = cutmix_mask_shuffle_2 * scribble_shuffle_2 + (
                    1 - cutmix_mask_shuffle_2) * scribble_shuffle_3

            # for net2
            img_mixed_for_net_2 = cutmix_mask_shuffle_1 * image_shuffle_1 + (
                    1 - cutmix_mask_shuffle_1) * image_shuffle_3
            pseudo_label_for_net_2 = cutmix_mask_shuffle_1 * output_unsup_soft_1 + (
                    1 - cutmix_mask_shuffle_1) * output_unsup_soft_3
            scribble_mixed_for_net_2 = cutmix_mask_shuffle_1 * scribble_shuffle_1 + (
                    1 - cutmix_mask_shuffle_1) * scribble_shuffle_3

            # for net3
            img_mixed_for_net_3 = cutmix_mask_shuffle_1 * image_shuffle_1 + (
                    1 - cutmix_mask_shuffle_1) * image_shuffle_2
            pseudo_label_for_net_3 = cutmix_mask_shuffle_1 * output_unsup_soft_1 + (
                    1 - cutmix_mask_shuffle_1) * output_unsup_soft_2
            scribble_mixed_for_net_3 = cutmix_mask_shuffle_1 * scribble_shuffle_1 + (
                    1 - cutmix_mask_shuffle_1) * scribble_shuffle_2

        else:
            # for net1
            img_mixed_for_net_1 = cutmix_mask_shuffle_3 * image_shuffle_3 + (
                    1 - cutmix_mask_shuffle_3) * image_shuffle_2
            pseudo_label_for_net_1 = cutmix_mask_shuffle_3 * output_unsup_soft_3 + (
                    1 - cutmix_mask_shuffle_3) * output_unsup_soft_2
            scribble_mixed_for_net_1 = cutmix_mask_shuffle_3 * scribble_shuffle_3 + (
                    1 - cutmix_mask_shuffle_3) * scribble_shuffle_2

            # for net2
            img_mixed_for_net_2 = cutmix_mask_shuffle_3 * image_shuffle_3 + (
                    1 - cutmix_mask_shuffle_3) * image_shuffle_1
            pseudo_label_for_net_2 = cutmix_mask_shuffle_3 * output_unsup_soft_3 + (
                    1 - cutmix_mask_shuffle_3) * output_unsup_soft_1
            scribble_mixed_for_net_2 = cutmix_mask_shuffle_3 * scribble_shuffle_3 + (
                    1 - cutmix_mask_shuffle_3) * scribble_shuffle_1

            # for net3
            img_mixed_for_net_3 = cutmix_mask_shuffle_2 * image_shuffle_2 + (
                    1 - cutmix_mask_shuffle_2) * image_shuffle_1
            pseudo_label_for_net_3 = cutmix_mask_shuffle_2 * output_unsup_soft_2 + (
                    1 - cutmix_mask_shuffle_2) * output_unsup_soft_1
            scribble_mixed_for_net_3 = cutmix_mask_shuffle_2 * scribble_shuffle_2 + (
                    1 - cutmix_mask_shuffle_2) * scribble_shuffle_1

        ### second forward
        # net1
        output_mixed_net_1 = net_1(img_mixed_for_net_1)

        assert output_mixed_net_1.requires_grad == True
        pseudo_label_for_net_1 = torch.argmax(pseudo_label_for_net_1, dim=1)
        assert pseudo_label_for_net_1.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_mixed_1 = F.cross_entropy(output_mixed_net_1, pseudo_label_for_net_1)
        # cross entropy with mixed scribble
        loss_sup_mixed_1 = partial_ce_loss(output_mixed_net_1, scribble_mixed_for_net_1[:, 0].long())

        # net2
        output_mixed_net_2 = net_2(img_mixed_for_net_2)

        assert output_mixed_net_2.requires_grad == True
        pseudo_label_for_net_2 = torch.argmax(pseudo_label_for_net_2, dim=1)
        assert pseudo_label_for_net_2.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_mixed_2 = F.cross_entropy(output_mixed_net_2, pseudo_label_for_net_2)
        # cross entropy with mixed scribble
        loss_sup_mixed_2 = partial_ce_loss(output_mixed_net_2, scribble_mixed_for_net_2[:, 0].long())

        # net3
        output_mixed_net_3 = net_3(img_mixed_for_net_3)

        assert output_mixed_net_3.requires_grad == True
        pseudo_label_for_net_3 = torch.argmax(pseudo_label_for_net_3, dim=1)
        assert pseudo_label_for_net_3.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_mixed_3 = F.cross_entropy(output_mixed_net_3, pseudo_label_for_net_3)
        # cross entropy with mixed scribble
        loss_sup_mixed_3 = partial_ce_loss(output_mixed_net_3, scribble_mixed_for_net_3[:, 0].long())

        loss_1 = loss_sup_no_mix_1 + args.lamda_1 * loss_sup_mixed_1 + args.lamda_2 * loss_unsup_mixed_1
        loss_2 = loss_sup_no_mix_2 + args.lamda_1 * loss_sup_mixed_2 + args.lamda_2 * loss_unsup_mixed_2
        loss_3 = loss_sup_no_mix_3 + args.lamda_1 * loss_sup_mixed_3 + args.lamda_2 * loss_unsup_mixed_3

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        optimizer_3.zero_grad()
        loss_3.backward()
        optimizer_3.step()

        if args.update_lr:
            # change learning rate
            lr_ = args.base_lr * (1.0 - i_iter / max_iterations) ** 0.9
            for param_group in optimizer_1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_2.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_3.param_groups:
                param_group['lr'] = lr_
        else:
            lr_ = args.base_lr

        # print and write
        logging.info('iter {}, net_1: lr {}, loss {}, loss_sup_no_mix {}, '
                     'loss_sup_mixed {}, loss_unsup_mixed {}'.format(i_iter + 1, lr_,
                                                                     loss_1.item(), loss_sup_no_mix_1.item(),
                                                                     loss_sup_mixed_1.item(),
                                                                     loss_unsup_mixed_1.item()))
        logging.info('iter {}, net_2:, lr {}, loss {}, loss_sup_no_mix {}, '
                     'loss_sup_mixed {}, loss_unsup_mixed {}'.format(i_iter + 1, lr_,
                                                                     loss_2.item(), loss_sup_no_mix_2.item(),
                                                                     loss_sup_mixed_2.item(),
                                                                     loss_unsup_mixed_2.item()))
        logging.info('iter {}, net_3:, lr {}, loss {}, loss_sup_no_mix {}, '
                     'loss_sup_mixed {}, loss_unsup_mixed {}'.format(i_iter + 1, lr_,
                                                                     loss_3.item(), loss_sup_no_mix_3.item(),
                                                                     loss_sup_mixed_3.item(),
                                                                     loss_unsup_mixed_3.item()))

        writer.add_scalar('train/loss_1', loss_1.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_no_mix_1', loss_sup_no_mix_1.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_mixed_1', loss_sup_mixed_1.item(), i_iter + 1)
        writer.add_scalar('train/loss_unsup_mixed_1', loss_unsup_mixed_1.item(), i_iter + 1)

        writer.add_scalar('train/loss_2', loss_2.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_no_mix_2', loss_sup_no_mix_2.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_mixed_2', loss_sup_mixed_2.item(), i_iter + 1)
        writer.add_scalar('train/loss_unsup_mixed_2', loss_unsup_mixed_2.item(), i_iter + 1)

        writer.add_scalar('train/loss_3', loss_3.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_no_mix_3', loss_sup_no_mix_3.item(), i_iter + 1)
        writer.add_scalar('train/loss_sup_mixed_3', loss_sup_mixed_3.item(), i_iter + 1)
        writer.add_scalar('train/loss_unsup_mixed_3', loss_unsup_mixed_3.item(), i_iter + 1)

        writer.add_scalar('train/lr', lr_, i_iter + 1)

        if (i_iter + 1) % 1000 == 0 and save_fig:
            image = make_grid(image_shuffle_1, image_shuffle_1.shape[0], normalize=True)
            writer.add_image('train/image_shuffle_1', image, i_iter + 1)

            image = make_grid(cutmix_mask_shuffle_1.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_1.shape[0],
                              normalize=False)
            writer.add_image('train/cutmix_mask_shuffle_1', image, i_iter + 1)

            image = make_grid(img_mixed_for_net_1, img_mixed_for_net_1.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_1', image, i_iter + 1)

            image = make_grid(pseudo_label_for_net_1.unsqueeze(1), pseudo_label_for_net_1.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_1', image * 50, i_iter + 1)

            image = make_grid(scribble_shuffle_1, scribble_shuffle_1.shape[0], normalize=False)
            writer.add_image('train/scribble_shuffle_1', image * 50, i_iter + 1)

            image = make_grid(scribble_mixed_for_net_1, scribble_mixed_for_net_1.shape[0], normalize=False)
            writer.add_image('train/scribble_mixed_for_net_1', image * 50, i_iter + 1)

            image = make_grid(image_shuffle_2, image_shuffle_2.shape[0], normalize=True)
            writer.add_image('train/image_shuffle_2', image, i_iter + 1)

            image = make_grid(cutmix_mask_shuffle_2.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_2.shape[0],
                              normalize=False)
            writer.add_image('train/cutmix_mask_shuffle_2', image, i_iter + 1)

            image = make_grid(img_mixed_for_net_2, img_mixed_for_net_2.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_2', image, i_iter + 1)

            image = make_grid(pseudo_label_for_net_2.unsqueeze(1), pseudo_label_for_net_2.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_2', image * 50, i_iter + 1)

            image = make_grid(scribble_shuffle_2, scribble_shuffle_2.shape[0], normalize=False)
            writer.add_image('train/scribble_shuffle_2', image * 50, i_iter + 1)

            image = make_grid(scribble_mixed_for_net_2, scribble_mixed_for_net_2.shape[0], normalize=False)
            writer.add_image('train/scribble_mixed_for_net_2', image * 50, i_iter + 1)

            image = make_grid(image_shuffle_3, image_shuffle_3.shape[0], normalize=True)
            writer.add_image('train/image_shuffle_3', image, i_iter + 1)

            image = make_grid(cutmix_mask_shuffle_3.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_3.shape[0],
                              normalize=False)
            writer.add_image('train/cutmix_mask_shuffle_3', image, i_iter + 1)

            image = make_grid(img_mixed_for_net_3, img_mixed_for_net_3.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_3', image, i_iter + 1)

            image = make_grid(pseudo_label_for_net_3.unsqueeze(1), pseudo_label_for_net_3.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_3', image * 50, i_iter + 1)

            image = make_grid(pseudo_label_for_net_3.unsqueeze(1), pseudo_label_for_net_3.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_3', image * 50, i_iter + 1)

            image = make_grid(scribble_shuffle_3, scribble_shuffle_3.shape[0], normalize=False)
            writer.add_image('train/scribble_shuffle_3', image * 50, i_iter + 1)

            image = make_grid(scribble_mixed_for_net_3, scribble_mixed_for_net_3.shape[0], normalize=False)
            writer.add_image('train/scribble_mixed_for_net_3', image * 50, i_iter + 1)

    return


if __name__ == '__main__':
    main()
