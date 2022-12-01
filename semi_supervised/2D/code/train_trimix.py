import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from itertools import cycle

from dataloader.dataset_semi import BaseDataSets, RandomGenerator, Cutmix
from network.unet import UNet
from util.inference import test_single_volume, test_single_volume_ensemble


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../dataset/ACDC', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='trimix', help='experiment_name')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='label', help='supervision type')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=60000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--lamda', type=float, default=1, help='hyperparameter')
    parser.add_argument('--update_lr', action='store_false', help='whether update learning rate')
    args = parser.parse_args()

    return args


def main():
    # parse arguments
    args = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

    return


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    net_1 = UNet(in_chns=1, class_num=num_classes).cuda()
    net_2 = UNet(in_chns=1, class_num=num_classes).cuda()
    net_3 = UNet(in_chns=1, class_num=num_classes).cuda()

    db_train_labeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="labeled", fold=args.fold,
                                    split="train", sup_type=args.sup_type, transform=transforms.Compose([
            RandomGenerator(args.patch_size)
        ]))
    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, num=8, labeled_type="unlabeled", fold=args.fold,
                                      split="train", sup_type=args.sup_type, transform=transforms.Compose([
            RandomGenerator(args.patch_size), Cutmix(prop_range=0.2)]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader_labeled = DataLoader(db_train_labeled, batch_size=args.batch_size // 2, shuffle=True,
                                      num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    train_loader_unlabeled = DataLoader(db_train_unlabeled, batch_size=args.batch_size // 2, shuffle=True,
                                        num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False,
                            num_workers=1)

    net_1.train()
    net_2.train()
    net_3.train()

    optimizer_1 = optim.SGD(net_1.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(net_2.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)
    optimizer_3 = optim.SGD(net_3.parameters(), lr=base_lr,
                            momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_loader_labeled)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader_labeled) + 1
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i, data in enumerate(zip(cycle(train_loader_labeled), train_loader_unlabeled)):
            sampled_batch_labeled, sampled_batch_unlabeled = data[0], data[1]

            volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch, mask = sampled_batch_unlabeled['image'].cuda(), sampled_batch_unlabeled[
                'mask'].cuda()

            # labeled data, first forward only
            # net1, supervised loss
            output_1 = net_1(volume_batch)
            loss_sup_1 = ce_loss(output_1, label_batch[:].long())

            # net_2, supervised loss
            output_2 = net_2(volume_batch)
            loss_sup_2 = ce_loss(output_2, label_batch[:].long())

            # net_3, supervised loss
            output_3 = net_3(volume_batch)
            loss_sup_3 = ce_loss(output_3, label_batch[:].long())

            ## unlabeled data
            # first forward
            with torch.no_grad():
                output_unsup_1 = net_1(unlabeled_volume_batch)
                output_unsup_2 = net_2(unlabeled_volume_batch)
                output_unsup_3 = net_3(unlabeled_volume_batch)

            # perform cutmix
            rand_index1 = torch.randperm(unlabeled_volume_batch.size()[0]).cuda()
            rand_index2 = torch.randperm(unlabeled_volume_batch.size()[0]).cuda()

            # shuffle
            image_unsup_shuffle_1 = unlabeled_volume_batch
            cutmix_shuffle_mask_1 = mask

            image_unsup_shuffle_2 = unlabeled_volume_batch[rand_index1]
            cutmix_shuffle_mask_2 = mask[rand_index1]

            image_unsup_shuffle_3 = unlabeled_volume_batch[rand_index2]
            cutmix_shuffle_mask_3 = mask[rand_index2]

            with torch.no_grad():
                output_unsup_1 = output_unsup_1
                output_unsup_2 = output_unsup_2[rand_index1]
                output_unsup_3 = output_unsup_3[rand_index2]

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

            loss_1 = loss_sup_1 + args.lamda * loss_unsup_1
            loss_2 = loss_sup_2 + args.lamda * loss_unsup_2
            loss_3 = loss_sup_3 + args.lamda * loss_unsup_3

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
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_3.param_groups:
                    param_group['lr'] = lr_
            else:
                lr_ = args.base_lr

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss_1, iter_num)
            writer.add_scalar('info/loss_sup', loss_sup_1, iter_num)
            writer.add_scalar('info/loss_unsup', loss_unsup_1, iter_num)

            logging.info(
                'iteration %d : net_1: loss : %f, loss_sup : %f,  loss_unsup : %f' %
                (iter_num, loss_1.item(), loss_sup_1.item(), loss_unsup_1.item()))
            logging.info(
                'iteration %d : net_2: loss : %f, loss_sup : %f,  loss_unsup : %f' %
                (iter_num, loss_2.item(), loss_sup_2.item(), loss_unsup_2.item()))
            logging.info(
                'iteration %d : net_3: loss : %f, loss_sup : %f,  loss_unsup : %f' %
                (iter_num, loss_3.item(), loss_sup_3.item(), loss_unsup_3.item()))

            if iter_num % 5000 == 0:
                image = make_grid(image_unsup_shuffle_1, image_unsup_shuffle_1.shape[0], normalize=True)
                writer.add_image('train/image_unsup_shuffle_1', image, iter_num)

                image = make_grid(cutmix_shuffle_mask_1.type_as(img_mixed_for_net_1), cutmix_shuffle_mask_1.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_shuffle_mask_1', image, iter_num)

                image = make_grid(img_mixed_for_net_1, img_mixed_for_net_1.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_1', image, iter_num)

                image = make_grid(pseudo_label_for_net_1.unsqueeze(1), pseudo_label_for_net_1.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_1', image * 50, iter_num)

                image = make_grid(image_unsup_shuffle_2, image_unsup_shuffle_2.shape[0], normalize=True)
                writer.add_image('train/image_unsup_shuffle_2', image, iter_num)

                image = make_grid(cutmix_shuffle_mask_2.type_as(img_mixed_for_net_1), cutmix_shuffle_mask_2.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_shuffle_mask_2', image, iter_num)

                image = make_grid(img_mixed_for_net_2, img_mixed_for_net_2.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_2', image, iter_num)

                image = make_grid(pseudo_label_for_net_2.unsqueeze(1), pseudo_label_for_net_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_2', image * 50, iter_num)

                image = make_grid(image_unsup_shuffle_3, image_unsup_shuffle_3.shape[0], normalize=True)
                writer.add_image('train/image_unsup_shuffle_3', image, iter_num)

                image = make_grid(cutmix_shuffle_mask_3.type_as(img_mixed_for_net_1), cutmix_shuffle_mask_3.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_shuffle_mask_3', image, iter_num)

                image = make_grid(img_mixed_for_net_3, img_mixed_for_net_3.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_3', image, iter_num)

                image = make_grid(pseudo_label_for_net_3.unsqueeze(1), pseudo_label_for_net_3.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_3', image * 50, iter_num)

                image = make_grid(label_batch.unsqueeze(1), label_batch.shape[0], normalize=False)
                writer.add_image('train/label', image * 50, iter_num)

            # validation
            if iter_num > 0 and iter_num % 10000 == 0:
                net_1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net_1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_net1/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_net1/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_net1/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_net1/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'net_1 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # net_2
                net_2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net_2, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_net2/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_net2/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_net2/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_net2/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'net_2 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # net_3
                net_3.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_loader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], net_3, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_net3/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_net3/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_net3/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_net3/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'net_3 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # ensemble
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_loader):
                    metric_i = test_single_volume_ensemble(
                        sampled_batch["image"], sampled_batch["label"], net_1, net_2, net_3, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info_ensemble/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info_ensemble/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info_ensemble/val_mean_dice', performance, iter_num)
                writer.add_scalar('info_ensemble/val_mean_hd95', mean_hd95, iter_num)

                logging.info(
                    'ensemble : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                net_1.train()
                net_2.train()
                net_3.train()

            if iter_num % 60000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'net1_iter_' + str(iter_num) + '.pth')
                torch.save(net_1.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'net2_iter_' + str(iter_num) + '.pth')
                torch.save(net_2.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'net3_iter_' + str(iter_num) + '.pth')
                torch.save(net_3.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    main()
