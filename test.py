import argparse
import os
import torch.optim as optim
import torch.utils.data as util_data
import itertools

import network
import pre_process as prep
import lr_schedule
from util import *
from data_list import ImageList_au, ImageList_land_au

optim_dict = {'SGD': optim.SGD, 'Adam': optim.Adam}


def main(config):
    ## set loss criterion
    use_gpu = torch.cuda.is_available()
    
    ## prepare data
    dsets = {}
    dset_loaders = {}
    dsets['source'] = {}
    dset_loaders['source'] = {}

    dsets['source']['test'] = ImageList_au(config.src_test_path_prefix,
                                          transform=prep.image_test(crop_size=config.crop_size))

    dset_loaders['source']['test'] = util_data.DataLoader(dsets['source']['test'], batch_size=config.eval_batch_size,
                                                         shuffle=False, num_workers=config.num_workers)

    dsets['target'] = {}
    dset_loaders['target'] = {}

    dsets['target']['test'] = ImageList_au(config.tgt_test_path_prefix,
                                          transform=prep.image_test(crop_size=config.crop_size))

    dset_loaders['target']['test'] = util_data.DataLoader(dsets['target']['test'], batch_size=config.eval_batch_size,
                                                         shuffle=False, num_workers=config.num_workers)

    ## set network modules
    base_net = network.network_dict[config.base_net]()
    land_enc = network.network_dict[config.land_enc](land_num=config.land_num)
    au_enc = network.network_dict[config.au_enc](au_num=config.au_num)
    invar_shape_enc = network.network_dict[config.invar_shape_enc]()
    feat_gen = network.network_dict[config.feat_gen]()

    if use_gpu:
        base_net = base_net.cuda()
        land_enc = land_enc.cuda()
        au_enc = au_enc.cuda()
        invar_shape_enc = invar_shape_enc.cuda()
        feat_gen = feat_gen.cuda()

    base_net.train(False)
    land_enc.train(False)
    au_enc.train(False)
    invar_shape_enc.train(False)
    feat_gen.train(False)

    print(base_net, land_enc, au_enc, invar_shape_enc, feat_gen)

    if not os.path.exists(config.write_path_prefix + config.mode):
        os.makedirs(config.write_path_prefix + config.mode)
    if not os.path.exists(config.write_res_prefix + config.mode):
        os.makedirs(config.write_res_prefix + config.mode)

    test_type = 'target'  # 'source'

    if config.start_epoch <= 0:
        raise (RuntimeError('start_epoch should be larger than 0\n'))

    res_file = open(config.write_res_prefix + config.mode + '/' + test_type + '_test_AU_pred_' + str(config.start_epoch) + '.txt', 'w')

    for epoch in range(config.start_epoch, config.n_epochs + 1):

        base_net.load_state_dict(
            torch.load(config.write_path_prefix + config.mode + '/base_net_' + str(config.start_epoch) + '.pth'))
        land_enc.load_state_dict(
            torch.load(config.write_path_prefix + config.mode + '/land_enc_' + str(config.start_epoch) + '.pth'))
        au_enc.load_state_dict(
            torch.load(config.write_path_prefix + config.mode + '/au_enc_' + str(config.start_epoch) + '.pth'))
        invar_shape_enc.load_state_dict(
            torch.load(config.write_path_prefix + config.mode + '/invar_shape_enc_' + str(config.start_epoch) + '.pth'))
        feat_gen.load_state_dict(
            torch.load(config.write_path_prefix + config.mode + '/feat_gen_' + str(config.start_epoch) + '.pth'))

        if test_type == 'source':
            f1score_arr, acc_arr = AU_detection_eval_src(dset_loaders[test_type]['test'], base_net, au_enc, use_gpu=use_gpu)
        else:
            f1score_arr, acc_arr = AU_detection_eval_tgt(dset_loaders[test_type]['test'], base_net, land_enc, au_enc,
                                                         invar_shape_enc, feat_gen, use_gpu=use_gpu)

        print('epoch =%d, f1 score mean=%f, accuracy mean=%f' %(epoch, f1score_arr.mean(), acc_arr.mean()))
        print>> res_file, '%d\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean())

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--output_size', type=int, default=44, help='size for landmark response map')
    parser.add_argument('--au_num', type=int, default=6, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='mini-batch size for evaluation')
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of total epochs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='weak', choices=['weak', 'full'])
    parser.add_argument('--base_net', type=str, default='Feat_Enc')
    parser.add_argument('--land_enc', type=str, default='Land_Detect')
    parser.add_argument('--au_enc', type=str, default='AU_Detect')
    parser.add_argument('--invar_shape_enc', type=str, default='Texture_Enc')
    parser.add_argument('--feat_gen', type=str, default='Generator')

    # Directories.
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--src_test_path_prefix', type=str, default='data/list/BP4D_test')
    parser.add_argument('--tgt_test_path_prefix', type=str, default='data/list/emotioNet_test')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)