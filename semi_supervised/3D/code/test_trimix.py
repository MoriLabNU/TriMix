import os
import argparse
import torch

from network.vnet import VNet
from util.inference import test_all_case, test_all_case_ensemble_ave


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='../dataset/2018LA_Seg_Training Set/',
                        help='Name of Experiment')
    parser.add_argument('--exp', type=str, default='trimix', help='model_name')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()

    return args


def main():
    FLAGS = parser_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    snapshot_path = "../model/" + FLAGS.exp + "/"
    test_save_path_1 = "../model/prediction/" + FLAGS.exp + "_post_net_1/"
    if not os.path.exists(test_save_path_1):
        os.makedirs(test_save_path_1)
    test_save_path_2 = "../model/prediction/" + FLAGS.exp + "_post_net_2/"
    if not os.path.exists(test_save_path_2):
        os.makedirs(test_save_path_2)
    test_save_path_3 = "../model/prediction/" + FLAGS.exp + "_post_net_3/"
    if not os.path.exists(test_save_path_3):
        os.makedirs(test_save_path_3)
    test_save_path_ensemble = "../model/prediction/" + FLAGS.exp + "_post_ensemble/"
    if not os.path.exists(test_save_path_ensemble):
        os.makedirs(test_save_path_ensemble)

    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

    test_calculate_metric(snapshot_path, test_save_path_1, test_save_path_2,
                          test_save_path_3, test_save_path_ensemble, image_list)


def test_calculate_metric(snapshot_path, test_save_path_1, test_save_path_2,
                          test_save_path_3, test_save_path_ensemble, image_list):
    net_1 = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
    net_2 = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
    net_3 = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()

    # net_1
    save_mode_path = os.path.join(snapshot_path, 'iter_6000_net1.pth')
    net_1.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net_1.eval()

    avg_metric = test_all_case(net_1, image_list, num_classes=2,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path_1)

    # net_2
    save_mode_path = os.path.join(snapshot_path, 'iter_6000_net2.pth')
    net_2.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net_2.eval()

    avg_metric = test_all_case(net_2, image_list, num_classes=2,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path_2)

    # net_2
    save_mode_path = os.path.join(snapshot_path, 'iter_6000_net3.pth')
    net_3.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net_3.eval()

    avg_metric = test_all_case(net_3, image_list, num_classes=2,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path_3)

    # ensemble
    avg_metric = test_all_case_ensemble_ave(net_1, net_2, net_3, image_list, num_classes=2,
                                            patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                                            save_result=True, test_save_path=test_save_path_ensemble)

    return


if __name__ == '__main__':
    main()
