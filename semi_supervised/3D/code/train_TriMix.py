import argparse
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
import random
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from network.vnet import VNet
from dataloader.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from util import cutmix_3D, ramps
from util import losses
from util.inference import test_all_case, test_all_case_ensemble_ave


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../dataset/2018LA_Seg_Training Set/',
                        help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='trimix', help='model_name')
    parser.add_argument('--max_iterations', type=int, default=6000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--consistency', type=float, default=1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
    parser.add_argument('--warm_up', action='store_true', help='whether gradually increase hyperparameter.')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--prop_range', type=float, default=0.2, help='prop_range')
    parser.add_argument('--loss1', type=str, default='all', help='prop_range')
    parser.add_argument('--loss2', type=str, default='ce', help='prop_range')
    parser.add_argument('--sam', type=int, default=16, help='prop_range')

    args = parser.parse_args()

    return args


def get_current_consistency_weight(args, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if args.warm_up == True:
        return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    else:
        return args.consistency


def create_model(ema=False):
    # Network definition
    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def main():
    # parse arguments
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    train_data_path = args.root_path
    snapshot_path = "../model/" + args.exp + "/"

    batch_size = args.batch_size * len(args.gpu.split(','))
    max_iterations = args.max_iterations
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    patch_size = (112, 112, 80)

    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # create model
    net_1 = create_model()
    net_1 = net_1.cuda()
    net_2 = create_model()
    net_2 = net_2.cuda()
    net_3 = create_model()
    net_3 = net_3.cuda()

    # load data
    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                           cutmix_3D.Cutmix_3D(prop_range=args.prop_range),
                       ]))

    labeled_idxs = list(range(args.sam))
    unlabeled_idxs = list(range(args.sam, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)

    optimizer_1 = optim.SGD(net_1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(net_2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_3 = optim.SGD(net_3.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = args.base_lr

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        iter_num = train(args, iter_num, net_1, net_2, net_3, optimizer_1, optimizer_2,
                         optimizer_3, train_loader, writer, lr_, snapshot_path)

    writer.close()

    return 'training is finished.'


def train(args, iter_num, net_1, net_2, net_3, optimizer_1, optimizer_2,
          optimizer_3, train_loader, writer, lr_, snapshot_path):
    net_1.train()
    net_2.train()
    net_3.train()

    for i_batch, sampled_batch in enumerate(train_loader):
        iter_num = iter_num + 1

        volume_batch, label_batch, mask_batch = sampled_batch['image'], sampled_batch['label'], sampled_batch['mask']
        volume_batch, label_batch, mask_batch = volume_batch.cuda(), label_batch.cuda(), mask_batch.cuda()

        # labeled data
        img_labeled = volume_batch[:args.labeled_bs]
        ground_truth = label_batch[:args.labeled_bs]

        # impose consistency on all data, following previous method
        # only 4 samples are for batch cutmix, may not achieve satisfactory augmentation
        img_unlabeled = volume_batch
        cutmix_mask = mask_batch
        del volume_batch, label_batch, mask_batch

        #### labeled data, first forward
        # net1 forward
        output_sup_1 = net_1(img_labeled)
        loss_sup_ce_1 = F.cross_entropy(output_sup_1, ground_truth)
        output_sup_soft_1 = F.softmax(output_sup_1, dim=1)
        loss_sup_dice_1 = losses.dice_loss(output_sup_soft_1[:, 1, :, :, :], ground_truth == 1)

        loss_sup_1 = 0.5 * (loss_sup_ce_1 + loss_sup_dice_1)

        # net2 forward
        output_sup_2 = net_2(img_labeled)
        loss_sup_ce_2 = F.cross_entropy(output_sup_2, ground_truth)
        output_sup_soft_2 = F.softmax(output_sup_2, dim=1)
        loss_sup_dice_2 = losses.dice_loss(output_sup_soft_2[:, 1, :, :, :], ground_truth == 1)

        loss_sup_2 = 0.5 * (loss_sup_ce_2 + loss_sup_dice_2)

        # net3 forward
        output_sup_3 = net_3(img_labeled)
        loss_sup_ce_3 = F.cross_entropy(output_sup_3, ground_truth)
        output_sup_soft_3 = F.softmax(output_sup_3, dim=1)
        loss_sup_dice_3 = losses.dice_loss(output_sup_soft_3[:, 1, :, :, :], ground_truth == 1)

        loss_sup_3 = 0.5 * (loss_sup_ce_3 + loss_sup_dice_3)

        # unsupervised
        # first forward
        with torch.no_grad():
            output_unsup_1 = net_1(img_unlabeled)
            output_unsup_2 = net_2(img_unlabeled)
            output_unsup_3 = net_3(img_unlabeled)

        # perform cutmix
        rand_index_1 = torch.randperm(img_unlabeled.size()[0]).cuda()
        rand_index_2 = torch.randperm(img_unlabeled.size()[0]).cuda()

        # shuffle
        # shuffle
        image_unsup_shuffle_1 = img_unlabeled
        cutmix_shuffle_mask_1 = cutmix_mask

        image_unsup_shuffle_2 = img_unlabeled[rand_index_1]
        cutmix_shuffle_mask_2 = cutmix_mask[rand_index_1]

        image_unsup_shuffle_3 = img_unlabeled[rand_index_2]
        cutmix_shuffle_mask_3 = cutmix_mask[rand_index_2]

        with torch.no_grad():
            output_unsup_1 = output_unsup_1
            output_unsup_2 = output_unsup_2[rand_index_1]
            output_unsup_3 = output_unsup_3[rand_index_2]

        output_unsup_soft_1 = F.softmax(output_unsup_1, dim=1)
        output_unsup_soft_2 = F.softmax(output_unsup_2, dim=1)
        output_unsup_soft_3 = F.softmax(output_unsup_3, dim=1)

        # throw dice to generate strong augmented image, pseudo label and mixed scribble annotation.
        if random.random() > 0.5:
            # for net1
            img_mixed_for_net_1 = cutmix_shuffle_mask_2 * image_unsup_shuffle_2 + (
                    1 - cutmix_shuffle_mask_2) * image_unsup_shuffle_3
            pseudo_label_for_net_1 = cutmix_shuffle_mask_2 * output_unsup_soft_2 + (
                    1 - cutmix_shuffle_mask_2) * output_unsup_soft_3

            # for net_2
            img_mixed_for_net_2 = cutmix_shuffle_mask_1 * image_unsup_shuffle_1 + (
                    1 - cutmix_shuffle_mask_1) * image_unsup_shuffle_3
            pseudo_label_for_net_2 = cutmix_shuffle_mask_1 * output_unsup_soft_1 + (
                    1 - cutmix_shuffle_mask_1) * output_unsup_soft_3

            # for net_3
            img_mixed_for_net_3 = cutmix_shuffle_mask_1 * image_unsup_shuffle_1 + (
                    1 - cutmix_shuffle_mask_1) * image_unsup_shuffle_2
            pseudo_label_for_net_3 = cutmix_shuffle_mask_1 * output_unsup_soft_1 + (
                    1 - cutmix_shuffle_mask_1) * output_unsup_soft_2

        else:
            # for net1
            img_mixed_for_net_1 = cutmix_shuffle_mask_3 * image_unsup_shuffle_3 + (
                    1 - cutmix_shuffle_mask_3) * image_unsup_shuffle_2
            pseudo_label_for_net_1 = cutmix_shuffle_mask_3 * output_unsup_soft_3 + (
                    1 - cutmix_shuffle_mask_3) * output_unsup_soft_2

            # for net_2
            img_mixed_for_net_2 = cutmix_shuffle_mask_3 * image_unsup_shuffle_3 + (
                    1 - cutmix_shuffle_mask_3) * image_unsup_shuffle_1
            pseudo_label_for_net_2 = cutmix_shuffle_mask_3 * output_unsup_soft_3 + (
                    1 - cutmix_shuffle_mask_3) * output_unsup_soft_1

            # for net_3
            img_mixed_for_net_3 = cutmix_shuffle_mask_2 * image_unsup_shuffle_2 + (
                    1 - cutmix_shuffle_mask_2) * image_unsup_shuffle_1
            pseudo_label_for_net_3 = cutmix_shuffle_mask_2 * output_unsup_soft_2 + (
                    1 - cutmix_shuffle_mask_2) * output_unsup_soft_1

        ### second forward
        # net1
        output_mixed_net_1 = net_1(img_mixed_for_net_1)
        assert output_mixed_net_1.requires_grad == True
        pseudo_label_for_net_1 = torch.argmax(pseudo_label_for_net_1, dim=1)
        assert pseudo_label_for_net_1.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_1 = F.cross_entropy(output_mixed_net_1, pseudo_label_for_net_1)

        # net_2
        output_mixed_net_2 = net_2(img_mixed_for_net_2)
        assert output_mixed_net_2.requires_grad == True
        pseudo_label_for_net_2 = torch.argmax(pseudo_label_for_net_2, dim=1)
        assert pseudo_label_for_net_2.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_2 = F.cross_entropy(output_mixed_net_2, pseudo_label_for_net_2)

        # net_3
        output_mixed_net_3 = net_3(img_mixed_for_net_3)
        assert output_mixed_net_3.requires_grad == True
        pseudo_label_for_net_3 = torch.argmax(pseudo_label_for_net_3, dim=1)
        assert pseudo_label_for_net_3.requires_grad == False

        # cross entropy with pseudo_label
        loss_unsup_3 = F.cross_entropy(output_mixed_net_3, pseudo_label_for_net_3)

        consistency_weight = get_current_consistency_weight(args, iter_num // 150)

        loss_1 = loss_sup_1 + consistency_weight * loss_unsup_1
        loss_2 = loss_sup_2 + consistency_weight * loss_unsup_2
        loss_3 = loss_sup_3 + consistency_weight * loss_unsup_3

        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        optimizer_3.zero_grad()
        loss_3.backward()
        optimizer_3.step()

        ## change lr
        if iter_num % 2500 == 0:
            lr_ = args.base_lr * 0.1 ** (iter_num // 2500)
            for param_group in optimizer_1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_2.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_3.param_groups:
                param_group['lr'] = lr_

        writer.add_scalar('lr', lr_, iter_num)
        writer.add_scalar('consistency_weight', consistency_weight, iter_num)
        writer.add_scalar('loss/loss_1', loss_1, iter_num)
        writer.add_scalar('loss/loss_unsup_1', loss_unsup_1, iter_num)
        writer.add_scalar('loss/loss_sup_1', loss_sup_1, iter_num)
        writer.add_scalar('loss/loss_2', loss_2, iter_num)
        writer.add_scalar('loss/loss_unsup_2', loss_unsup_2, iter_num)
        writer.add_scalar('loss/loss_sup_2', loss_sup_2, iter_num)
        writer.add_scalar('loss/loss_3', loss_3, iter_num)
        writer.add_scalar('loss/loss_unsup_3', loss_unsup_3, iter_num)
        writer.add_scalar('loss/loss_sup_3', loss_sup_3, iter_num)

        logging.info('iteration %d : weight : %f ' %
                     (iter_num, consistency_weight))
        logging.info('iteration %d : net_1: loss_1 : %f loss_unsup_1: %f, loss_sup_1: %f' %
                     (iter_num, loss_1.item(), loss_unsup_1.item(), loss_sup_1.item()))
        logging.info('iteration %d : net_2: loss_2 : %f loss_unsup_2: %f, loss_sup_2: %f' %
                     (iter_num, loss_2.item(), loss_unsup_2.item(), loss_sup_2.item()))
        logging.info('iteration %d : net_3: loss_3 : %f loss_unsup_3: %f, loss_sup_3: %f' %
                     (iter_num, loss_3.item(), loss_unsup_3.item(), loss_sup_3.item()))

        if iter_num % 6000 == 0:
            save_mode_path1 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_net1.pth')
            torch.save(net_1.state_dict(), save_mode_path1)
            logging.info("save model to {}".format(save_mode_path1))

            save_mode_path2 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_net2.pth')
            torch.save(net_2.state_dict(), save_mode_path2)
            logging.info("save model to {}".format(save_mode_path2))

            save_mode_path3 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '_net3.pth')
            torch.save(net_3.state_dict(), save_mode_path3)
            logging.info("save model to {}".format(save_mode_path3))

        if iter_num % 1000 == 0:
            test(args, net_1, net_2, net_3)
            net_1.train()
            net_2.train()
            net_3.train()

        # save images
        if iter_num % 500 == 0:
            image = image_unsup_shuffle_1[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/image_unsup_shuffle_1', image, iter_num)

            image = cutmix_shuffle_mask_1[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=False)
            writer.add_image('train/cutmix_shuffle_mask_1', image, iter_num)

            image = img_mixed_for_net_1[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_1', image, iter_num)

            image = (pseudo_label_for_net_1.unsqueeze(1))[0].permute(3, 0, 1, 2)
            image = make_grid(image * 50, image.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_1', image, iter_num)

            image = image_unsup_shuffle_2[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/image_unsup_shuffle_2', image, iter_num)

            image = cutmix_shuffle_mask_2[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=False)
            writer.add_image('train/cutmix_shuffle_mask_2', image, iter_num)

            image = img_mixed_for_net_2[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_2', image, iter_num)

            image = (pseudo_label_for_net_2.unsqueeze(1))[0].permute(3, 0, 1, 2)
            image = make_grid(image * 50, image.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_2', image, iter_num)

            image = image_unsup_shuffle_3[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/image_unsup_shuffle_3', image, iter_num)

            image = cutmix_shuffle_mask_3[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=False)
            writer.add_image('train/cutmix_shuffle_mask_3', image, iter_num)

            image = img_mixed_for_net_3[0].permute(3, 0, 1, 2)
            image = make_grid(image, image.shape[0], normalize=True)
            writer.add_image('train/img_mixed_for_net_3', image, iter_num)

            image = (pseudo_label_for_net_3.unsqueeze(1))[0].permute(3, 0, 1, 2)
            image = make_grid(image * 50, image.shape[0], normalize=False)
            writer.add_image('train/pseudo_label_for_net_3', image, iter_num)

    return iter_num


def test(args, net_1, net_2, net_3):
    # start test
    with open(args.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

    net_1.eval()
    net_2.eval()
    net_3.eval()

    avg_metric_1 = test_all_case(net_1, image_list, num_classes=2,
                                 patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                 save_result=False, test_save_path=None)

    avg_metric_2 = test_all_case(net_2, image_list, num_classes=2,
                                 patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                 save_result=False, test_save_path=None)

    avg_metric_3 = test_all_case(net_3, image_list, num_classes=2,
                                 patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                 save_result=False, test_save_path=None)

    ensemble_ave = test_all_case_ensemble_ave(net_1, net_2, net_3, image_list, 2, patch_size=(112, 112, 80),
                                              stride_xy=18, stride_z=4, save_result=False, test_save_path=None,
                                              preproc_fn=None)

    logging.info('average metric of net1: {}'.format(avg_metric_1))
    logging.info('average metric of net2: {}'.format(avg_metric_2))
    logging.info('average metric of net3: {}'.format(avg_metric_3))
    logging.info('average metric of ensemble: {}'.format(ensemble_ave))

    return


if __name__ == "__main__":
    main()
