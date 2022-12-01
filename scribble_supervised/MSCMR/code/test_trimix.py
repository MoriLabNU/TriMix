import argparse
import sys
import logging
import os
import torch

from network.UNet import build_UNet
from util.inference import infer_trimix_single, infer_trimix_ensemble


# parse arguments
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='trimix', help='model_name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--n_classes', type=int, default=4, help='segmentation class')
    args = parser.parse_args()

    return args


def create_model(args, ema=False):
    # Network definition
    net = build_UNet(args)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def main():
    # parse arguments
    args = parser_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    snapshot_path = "../model/" + args.exp + "/"

    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log_test.txt", level=logging.INFO,
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

    save_mode_path = os.path.join(
        snapshot_path, 'net_1.pth')
    net_1.load_state_dict(torch.load(save_mode_path))
    logging.info("load model from {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, 'net_2.pth')
    net_2.load_state_dict(torch.load(save_mode_path))
    logging.info("load model from {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, 'net_3.pth')
    net_3.load_state_dict(torch.load(save_mode_path))
    logging.info("load model from {}".format(save_mode_path))

    dice_score = infer_trimix_single(net_1, snapshot_path + '/net1/', task='test')
    logging.info('average Dice score of net_1: {}'.format(dice_score))
    dice_score = infer_trimix_single(net_2, snapshot_path + '/net2/', task='test')
    logging.info('average Dice score of net_2: {}'.format(dice_score))
    dice_score = infer_trimix_single(net_3, snapshot_path + '/net3/', task='test')
    logging.info('average Dice score of net_3: {}'.format(dice_score))
    dice_score = infer_trimix_ensemble(net_1, net_2, net_3, snapshot_path + '/ensemble/', task='test')
    logging.info('average Dice score of ensemble: {}'.format(dice_score))


if __name__ == '__main__':
    main()
