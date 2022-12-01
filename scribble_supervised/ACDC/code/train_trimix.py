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

from dataloader.dataset import BaseDataSets, RandomGenerator, Cutmix
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
                        default='scribble', help='supervision type')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=60000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.03,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--lamda_1', type=float, default=1, help='hyperparameter')
    parser.add_argument('--lamda_2', type=float, default=1, help='hyperparameter')
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
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    net_1 = UNet(in_chns=1, class_num=num_classes).cuda()
    net_2 = UNet(in_chns=1, class_num=num_classes).cuda()
    net_3 = UNet(in_chns=1, class_num=num_classes).cuda()

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size), Cutmix(prop_range=0.2)
    ]), fold=args.fold, sup_type=args.sup_type)
    db_val = BaseDataSets(base_dir=args.root_path,
                          fold=args.fold, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_dataloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(db_val, batch_size=1, shuffle=False,
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

    partial_ce_loss = CrossEntropyLoss(ignore_index=4)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(train_dataloader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_dataloader) + 1

    save_fig = True

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_dataloader):

            volume_batch, scribble, cutmix_mask, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch[
                'mask'], sampled_batch['idx']
            volume_batch, scribble, cutmix_mask = volume_batch.cuda(), scribble.cuda(), cutmix_mask.cuda()

            ### first forward
            # net1, supervised loss, no mix
            output_1 = net_1(volume_batch)
            loss_sup_no_mix_1 = partial_ce_loss(output_1, scribble[:, 0].long())

            # net2, supervised loss, no mix
            output_2 = net_2(volume_batch)
            loss_sup_no_mix_2 = partial_ce_loss(output_2, scribble[:, 0].long())

            # net3, supervised loss, no mix
            output_3 = net_3(volume_batch)
            loss_sup_no_mix_3 = partial_ce_loss(output_3, scribble[:, 0].long())

            # perform cutmix
            rand_index_1 = torch.randperm(volume_batch.size()[0]).cuda()
            rand_index_2 = torch.randperm(volume_batch.size()[0]).cuda()

            # shuffle
            image_shuffle_1 = volume_batch
            scribble_shuffle_1 = scribble
            cutmix_mask_shuffle_1 = cutmix_mask

            image_shuffle_2 = volume_batch[rand_index_1]
            scribble_shuffle_2 = scribble[rand_index_1]
            cutmix_mask_shuffle_2 = cutmix_mask[rand_index_1]

            image_shuffle_3 = volume_batch[rand_index_2]
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

            # print and write
            logging.info('iter {}, net_1: lr {}, loss {}, loss_sup_no_mix {}, '
                         'loss_sup_mixed {}, loss_unsup_mixed {}'.format(iter_num, lr_,
                                                                         loss_1.item(), loss_sup_no_mix_1.item(),
                                                                         loss_sup_mixed_1.item(),
                                                                         loss_unsup_mixed_1.item()))
            logging.info('iter {}, net_2:, lr {}, loss {}, loss_sup_no_mix {}, '
                         'loss_sup_mixed {}, loss_unsup_mixed {}'.format(iter_num, lr_,
                                                                         loss_2.item(), loss_sup_no_mix_2.item(),
                                                                         loss_sup_mixed_2.item(),
                                                                         loss_unsup_mixed_2.item()))
            logging.info('iter {}, net_3:, lr {}, loss {}, loss_sup_no_mix {}, '
                         'loss_sup_mixed {}, loss_unsup_mixed {}'.format(iter_num, lr_,
                                                                         loss_3.item(), loss_sup_no_mix_3.item(),
                                                                         loss_sup_mixed_3.item(),
                                                                         loss_unsup_mixed_3.item()))

            writer.add_scalar('train/loss_1', loss_1.item(), iter_num)
            writer.add_scalar('train/loss_sup_no_mix_1', loss_sup_no_mix_1.item(), iter_num)
            writer.add_scalar('train/loss_sup_mixed_1', loss_sup_mixed_1.item(), iter_num)
            writer.add_scalar('train/loss_unsup_mixed_1', loss_unsup_mixed_1.item(), iter_num)

            writer.add_scalar('train/loss_2', loss_2.item(), iter_num)
            writer.add_scalar('train/loss_sup_no_mix_2', loss_sup_no_mix_2.item(), iter_num)
            writer.add_scalar('train/loss_sup_mixed_2', loss_sup_mixed_2.item(), iter_num)
            writer.add_scalar('train/loss_unsup_mixed_2', loss_unsup_mixed_2.item(), iter_num)

            writer.add_scalar('train/loss_3', loss_3.item(), iter_num)
            writer.add_scalar('train/loss_sup_no_mix_3', loss_sup_no_mix_3.item(), iter_num)
            writer.add_scalar('train/loss_sup_mixed_3', loss_sup_mixed_3.item(), iter_num)
            writer.add_scalar('train/loss_unsup_mixed_3', loss_unsup_mixed_3.item(), iter_num)

            writer.add_scalar('train/lr', lr_, iter_num)

            # save fig
            if iter_num % 5000 == 0 and save_fig:
                image = make_grid(image_shuffle_1, image_shuffle_1.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_1', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_1.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_1.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_1', image, iter_num)

                image = make_grid(img_mixed_for_net_1, img_mixed_for_net_1.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_1', image, iter_num)

                image = make_grid(pseudo_label_for_net_1.unsqueeze(1), pseudo_label_for_net_1.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_1', image * 50, iter_num)

                image = make_grid(scribble_shuffle_1, scribble_shuffle_1.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_1', image * 50, iter_num)

                image = make_grid(scribble_mixed_for_net_1, scribble_mixed_for_net_1.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed_for_net_1', image * 50, iter_num)

                image = make_grid(image_shuffle_2, image_shuffle_2.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_2', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_2.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_2.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_2', image, iter_num)

                image = make_grid(img_mixed_for_net_2, img_mixed_for_net_2.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_2', image, iter_num)

                image = make_grid(pseudo_label_for_net_2.unsqueeze(1), pseudo_label_for_net_2.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_2', image * 50, iter_num)

                image = make_grid(scribble_shuffle_2, scribble_shuffle_2.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_2', image * 50, iter_num)

                image = make_grid(scribble_mixed_for_net_2, scribble_mixed_for_net_2.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed_for_net_2', image * 50, iter_num)

                image = make_grid(image_shuffle_3, image_shuffle_3.shape[0], normalize=True)
                writer.add_image('train/image_shuffle_3', image, iter_num)

                image = make_grid(cutmix_mask_shuffle_3.type_as(img_mixed_for_net_1), cutmix_mask_shuffle_3.shape[0],
                                  normalize=False)
                writer.add_image('train/cutmix_mask_shuffle_3', image, iter_num)

                image = make_grid(img_mixed_for_net_3, img_mixed_for_net_3.shape[0], normalize=True)
                writer.add_image('train/img_mixed_for_net_3', image, iter_num)

                image = make_grid(pseudo_label_for_net_3.unsqueeze(1), pseudo_label_for_net_3.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_3', image * 50, iter_num)

                image = make_grid(pseudo_label_for_net_3.unsqueeze(1), pseudo_label_for_net_3.shape[0], normalize=False)
                writer.add_image('train/pseudo_label_for_net_3', image * 50, iter_num)

                image = make_grid(scribble_shuffle_3, scribble_shuffle_3.shape[0], normalize=False)
                writer.add_image('train/scribble_shuffle_3', image * 50, iter_num)

                image = make_grid(scribble_mixed_for_net_3, scribble_mixed_for_net_3.shape[0], normalize=False)
                writer.add_image('train/scribble_mixed_for_net_3', image * 50, iter_num)

            # validation
            if iter_num % 10000 == 0:
                # net1
                net_1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
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
                    'net1 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # net2
                net_2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
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
                    'net2 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # net3
                net_3.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
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
                    'net3 : iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))

                # ensemble
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(val_dataloader):
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
