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
    au_weight_src = torch.from_numpy(np.loadtxt(config.src_train_path_prefix + '_weight.txt'))
    if use_gpu:
        au_weight_src = au_weight_src.float().cuda()
    else:
        au_weight_src = au_weight_src.float()

    au_class_criterion = nn.BCEWithLogitsLoss(au_weight_src)
    land_predict_criterion = land_softmax_loss
    discriminator_criterion = nn.MSELoss()
    reconstruct_criterion = nn.L1Loss()
    land_discriminator_criterion = land_discriminator_loss
    land_adaptation_criterion = land_adaptation_loss

    ## prepare data
    dsets = {}
    dset_loaders = {}
    dsets['source'] = {}
    dset_loaders['source'] = {}

    dsets['source']['train'] = ImageList_land_au(config.crop_size, config.src_train_path_prefix,
                                                 transform=prep.image_train(crop_size=config.crop_size),
                                                 target_transform=prep.land_transform(output_size=config.output_size,
                                                                                      scale=config.crop_size / config.output_size,
                                                                                      flip_reflect=np.loadtxt(
                                                                                          config.flip_reflect)))

    dset_loaders['source']['train'] = util_data.DataLoader(dsets['source']['train'], batch_size=config.train_batch_size,
                                                           shuffle=True, num_workers=config.num_workers)

    dsets['source']['val'] = ImageList_au(config.src_val_path_prefix,
                                          transform=prep.image_test(crop_size=config.crop_size))

    dset_loaders['source']['val'] = util_data.DataLoader(dsets['source']['val'], batch_size=config.eval_batch_size,
                                                         shuffle=False, num_workers=config.num_workers)


    dsets['target'] = {}
    dset_loaders['target'] = {}

    dsets['target']['train'] = ImageList_land_au(config.crop_size, config.tgt_train_path_prefix,
                                                 transform=prep.image_train(crop_size=config.crop_size),
                                                 target_transform=prep.land_transform(output_size=config.output_size,
                                                                                      scale=config.crop_size / config.output_size,
                                                                                      flip_reflect=np.loadtxt(
                                                                                          config.flip_reflect)))

    dset_loaders['target']['train'] = util_data.DataLoader(dsets['target']['train'], batch_size=config.train_batch_size,
                                                           shuffle=True, num_workers=config.num_workers)

    dsets['target']['val'] = ImageList_au(config.tgt_val_path_prefix,
                                          transform=prep.image_test(crop_size=config.crop_size))

    dset_loaders['target']['val'] = util_data.DataLoader(dsets['target']['val'], batch_size=config.eval_batch_size,
                                                         shuffle=False, num_workers=config.num_workers)


    ## set network modules
    base_net = network.network_dict[config.base_net]()
    land_enc = network.network_dict[config.land_enc](land_num=config.land_num)
    land_enc_store = network.network_dict[config.land_enc](land_num=config.land_num)
    au_enc = network.network_dict[config.au_enc](au_num=config.au_num)
    invar_shape_enc = network.network_dict[config.invar_shape_enc]()
    feat_gen = network.network_dict[config.feat_gen]()
    invar_shape_disc = network.network_dict[config.invar_shape_disc](land_num=config.land_num)
    feat_gen_disc_src = network.network_dict[config.feat_gen_disc]()
    feat_gen_disc_tgt = network.network_dict[config.feat_gen_disc]()

    if config.start_epoch > 0:
        base_net.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/base_net_' + str(config.start_epoch) + '.pth'))
        land_enc.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/land_enc_' + str(config.start_epoch) + '.pth'))
        au_enc.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/au_enc_' + str(config.start_epoch) + '.pth'))
        invar_shape_enc.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/invar_shape_enc_' + str(config.start_epoch) + '.pth'))
        feat_gen.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/feat_gen_' + str(config.start_epoch) + '.pth'))
        invar_shape_disc.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/invar_shape_disc_' + str(config.start_epoch) + '.pth'))
        feat_gen_disc_src.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/feat_gen_disc_src_' + str(config.start_epoch) + '.pth'))
        feat_gen_disc_tgt.load_state_dict(torch.load(config.write_path_prefix + config.mode + '/feat_gen_disc_tgt_' + str(config.start_epoch) + '.pth'))

    if use_gpu:
        base_net = base_net.cuda()
        land_enc = land_enc.cuda()
        land_enc_store = land_enc_store.cuda()
        au_enc = au_enc.cuda()
        invar_shape_enc = invar_shape_enc.cuda()
        feat_gen = feat_gen.cuda()
        invar_shape_disc = invar_shape_disc.cuda()
        feat_gen_disc_src = feat_gen_disc_src.cuda()
        feat_gen_disc_tgt = feat_gen_disc_tgt.cuda()

    ## collect parameters
    base_net_parameter_list = [{'params': filter(lambda p: p.requires_grad, base_net.parameters()), 'lr': 1}]
    land_enc_parameter_list = [{'params': filter(lambda p: p.requires_grad, land_enc.parameters()), 'lr': 1}]
    au_enc_parameter_list = [{'params': filter(lambda p: p.requires_grad, au_enc.parameters()), 'lr': 1}]
    invar_shape_enc_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, invar_shape_enc.parameters()), 'lr': 1}]
    feat_gen_parameter_list = [{'params': filter(lambda p: p.requires_grad, feat_gen.parameters()), 'lr': 1}]
    invar_shape_disc_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, invar_shape_disc.parameters()), 'lr': 1}]
    feat_gen_disc_src_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, feat_gen_disc_src.parameters()), 'lr': 1}]
    feat_gen_disc_tgt_parameter_list = [
        {'params': filter(lambda p: p.requires_grad, feat_gen_disc_tgt.parameters()), 'lr': 1}]

    ## set optimizer
    Gen_optimizer = optim_dict[config.gen_optimizer_type](
        itertools.chain(invar_shape_enc_parameter_list, feat_gen_parameter_list),
        1.0, [config.gen_beta1, config.gen_beta2])
    Task_optimizer = optim_dict[config.task_optimizer_type](
        itertools.chain(base_net_parameter_list, land_enc_parameter_list, au_enc_parameter_list),
        1.0, [config.task_beta1, config.task_beta2])
    Disc_optimizer = optim_dict[config.gen_optimizer_type](
        itertools.chain(invar_shape_disc_parameter_list, feat_gen_disc_src_parameter_list,
                        feat_gen_disc_tgt_parameter_list), 1.0, [config.gen_beta1, config.gen_beta2])

    Gen_param_lr = []
    for param_group in Gen_optimizer.param_groups:
        Gen_param_lr.append(param_group['lr'])

    Task_param_lr = []
    for param_group in Task_optimizer.param_groups:
        Task_param_lr.append(param_group['lr'])

    Disc_param_lr = []
    for param_group in Disc_optimizer.param_groups:
        Disc_param_lr.append(param_group['lr'])


    Gen_lr_scheduler = lr_schedule.schedule_dict[config.gen_lr_type]
    Task_lr_scheduler = lr_schedule.schedule_dict[config.task_lr_type]
    Disc_lr_scheduler = lr_schedule.schedule_dict[config.gen_lr_type]

    print(base_net, land_enc, au_enc, invar_shape_enc, feat_gen)
    print(invar_shape_disc, feat_gen_disc_src, feat_gen_disc_tgt)

    if not os.path.exists(config.write_path_prefix + config.mode):
        os.makedirs(config.write_path_prefix + config.mode)
    if not os.path.exists(config.write_res_prefix + config.mode):
        os.makedirs(config.write_res_prefix + config.mode)

    val_type = 'target'  # 'source'
    res_file = open(config.write_res_prefix + config.mode + '/' + val_type + '_AU_pred_' + str(config.start_epoch) + '.txt', 'w')

    ## train
    len_train_tgt = len(dset_loaders['target']['train'])
    count = 0

    for epoch in range(config.start_epoch, config.n_epochs + 1):
        # eval in the train
        if epoch >= config.start_epoch:
            base_net.train(False)
            land_enc.train(False)
            au_enc.train(False)
            invar_shape_enc.train(False)
            feat_gen.train(False)
            if val_type == 'source':
                f1score_arr, acc_arr = AU_detection_eval_src(dset_loaders[val_type]['val'], base_net, au_enc, use_gpu=use_gpu)
            else:
                f1score_arr, acc_arr = AU_detection_eval_tgt(dset_loaders[val_type]['val'], base_net, land_enc, au_enc,
                                                             invar_shape_enc, feat_gen, use_gpu=use_gpu)

            print('epoch =%d, f1 score mean=%f, accuracy mean=%f' %(epoch, f1score_arr.mean(), acc_arr.mean()))
            print>> res_file, '%d\t%f\t%f' % (epoch, f1score_arr.mean(), acc_arr.mean())
            base_net.train(True)
            land_enc.train(True)
            au_enc.train(True)
            invar_shape_enc.train(True)
            feat_gen.train(True)

        if epoch > config.start_epoch:
            print('taking snapshot ...')
            torch.save(base_net.state_dict(), config.write_path_prefix + config.mode + '/base_net_' + str(epoch) + '.pth')
            torch.save(land_enc.state_dict(), config.write_path_prefix + config.mode + '/land_enc_' + str(epoch) + '.pth')
            torch.save(au_enc.state_dict(), config.write_path_prefix + config.mode + '/au_enc_' + str(epoch) + '.pth')
            torch.save(invar_shape_enc.state_dict(), config.write_path_prefix + config.mode + '/invar_shape_enc_' + str(epoch) + '.pth')
            torch.save(feat_gen.state_dict(), config.write_path_prefix + config.mode + '/feat_gen_' + str(epoch) + '.pth')
            torch.save(invar_shape_disc.state_dict(), config.write_path_prefix + config.mode + '/invar_shape_disc_' + str(epoch) + '.pth')
            torch.save(feat_gen_disc_src.state_dict(), config.write_path_prefix + config.mode + '/feat_gen_disc_src_' + str(epoch) + '.pth')
            torch.save(feat_gen_disc_tgt.state_dict(), config.write_path_prefix + config.mode + '/feat_gen_disc_tgt_' + str(epoch) + '.pth')

        if epoch >= config.n_epochs:
            break

        for i, batch_src in enumerate(dset_loaders['source']['train']):
            if i % config.display == 0 and count > 0:
                print(
                            '[epoch = %d][iter = %d][loss_disc = %f][loss_invar_shape_disc = %f][loss_gen_disc = %f][total_loss = %f][loss_invar_shape_adaptation = %f][loss_gen_adaptation = %f][loss_self_recons = %f][loss_gen_cycle = %f][loss_au = %f][loss_land = %f]' % (
                        epoch, i, loss_disc.data.cpu().numpy(), loss_invar_shape_disc.data.cpu().numpy(),
                        loss_gen_disc.data.cpu().numpy(), total_loss.data.cpu().numpy(),
                        loss_invar_shape_adaptation.data.cpu().numpy(), loss_gen_adaptation.data.cpu().numpy(),
                        loss_self_recons.data.cpu().numpy(), loss_gen_cycle.data.cpu().numpy(),
                        loss_au.data.cpu().numpy(), loss_land.data.cpu().numpy()))

                print('learning rate = %f, %f, %f' % (Disc_optimizer.param_groups[0]['lr'], Gen_optimizer.param_groups[0]['lr'], Task_optimizer.param_groups[0]['lr']))
                print('the number of training iterations is %d' % (count))

            input_src, land_src, au_src = batch_src
            if count % len_train_tgt == 0:
                if count > 0:
                    dset_loaders['target']['train'] = util_data.DataLoader(dsets['target']['train'], batch_size=config.train_batch_size,
                                                                           shuffle=True, num_workers=config.num_workers)
                iter_data_tgt = iter(dset_loaders['target']['train'])
            input_tgt, land_tgt, au_tgt = iter_data_tgt.next()

            if input_tgt.size(0) > input_src.size(0):
                input_tgt, land_tgt, au_tgt = input_tgt[0:input_src.size(0), :, :, :], land_tgt[0:input_src.size(0),
                                                                                       :], au_tgt[
                                                                                           0:input_src.size(0)]
            elif input_tgt.size(0) < input_src.size(0):
                input_src, land_src, au_src = input_src[0:input_tgt.size(0), :, :, :], land_src[0:input_tgt.size(0),
                                                                                       :], au_src[
                                                                                           0:input_tgt.size(0)]

            if use_gpu:
                input_src, land_src, au_src, input_tgt, land_tgt, au_tgt = \
                    input_src.cuda(), land_src.long().cuda(), au_src.float().cuda(), \
                    input_tgt.cuda(), land_tgt.long().cuda(), au_tgt.float().cuda()
            else:
                land_src, au_src, land_tgt, au_tgt = \
                    land_src.long(), au_src.float(), land_tgt.long(), au_tgt.float()

            land_enc_store.load_state_dict(land_enc.state_dict())

            base_feat_src = base_net(input_src)
            align_attention_src, align_feat_src, align_output_src = land_enc(base_feat_src)
            au_feat_src, au_output_src = au_enc(base_feat_src)

            base_feat_tgt = base_net(input_tgt)
            align_attention_tgt, align_feat_tgt, align_output_tgt = land_enc(base_feat_tgt)
            au_feat_tgt, au_output_tgt = au_enc(base_feat_tgt)

            invar_shape_output_src = invar_shape_enc(base_feat_src.detach())
            invar_shape_output_tgt = invar_shape_enc(base_feat_tgt.detach())

            # new_gen
            new_gen_tgt = feat_gen(align_attention_src.detach(), invar_shape_output_tgt)
            new_gen_src = feat_gen(align_attention_tgt.detach(), invar_shape_output_src)

            # recons_gen
            recons_gen_src = feat_gen(align_attention_src.detach(), invar_shape_output_src)
            recons_gen_tgt = feat_gen(align_attention_tgt.detach(), invar_shape_output_tgt)

            # new2_gen
            new_gen_invar_shape_output_src = invar_shape_enc(new_gen_src.detach())
            new_gen_invar_shape_output_tgt = invar_shape_enc(new_gen_tgt.detach())
            new_gen_align_attention_src, new_gen_align_feat_src, new_gen_align_output_src = land_enc_store(new_gen_src)
            new_gen_align_attention_tgt, new_gen_align_feat_tgt, new_gen_align_output_tgt = land_enc_store(new_gen_tgt)
            new2_gen_tgt = feat_gen(new_gen_align_attention_src.detach(), new_gen_invar_shape_output_tgt)
            new2_gen_src = feat_gen(new_gen_align_attention_tgt.detach(), new_gen_invar_shape_output_src)

            ############################
            # 1. train discriminator #
            ############################
            Disc_optimizer = Disc_lr_scheduler(Disc_param_lr, Disc_optimizer, epoch, config.n_epochs,
                                               1, config.decay_start_epoch, config.gen_lr)
            Disc_optimizer.zero_grad()

            align_output_invar_shape_src = invar_shape_disc(
                invar_shape_output_src.detach())
            align_output_invar_shape_tgt = invar_shape_disc(
                invar_shape_output_tgt.detach())

            # loss_invar_shape_disc
            loss_base_invar_shape_disc_src = land_discriminator_criterion(align_output_invar_shape_src, land_src)
            loss_base_invar_shape_disc_tgt = land_discriminator_criterion(align_output_invar_shape_tgt, land_tgt)
            loss_invar_shape_disc = (loss_base_invar_shape_disc_src + loss_base_invar_shape_disc_tgt) * 0.5

            base_gen_src_pred = feat_gen_disc_src(base_feat_src.detach())
            new_gen_src_pred = feat_gen_disc_src(new_gen_src.detach())

            real_label = torch.ones((base_feat_src.size(0), 1))
            fake_label = torch.zeros((base_feat_src.size(0), 1))
            if use_gpu:
                real_label, fake_label = real_label.cuda(), fake_label.cuda()
            # loss_gen_disc_src
            loss_base_gen_src = discriminator_criterion(base_gen_src_pred, real_label)
            loss_new_gen_src = discriminator_criterion(new_gen_src_pred, fake_label)
            loss_gen_disc_src = (loss_base_gen_src + loss_new_gen_src) * 0.5

            base_gen_tgt_pred = feat_gen_disc_tgt(base_feat_tgt.detach())
            new_gen_tgt_pred = feat_gen_disc_tgt(new_gen_tgt.detach())

            # loss_gen_disc_tgt
            loss_base_gen_tgt = discriminator_criterion(base_gen_tgt_pred, real_label)
            loss_new_gen_tgt = discriminator_criterion(new_gen_tgt_pred, fake_label)
            loss_gen_disc_tgt = (loss_base_gen_tgt + loss_new_gen_tgt) * 0.5
            # loss_gen_disc
            loss_gen_disc = (loss_gen_disc_src + loss_gen_disc_tgt) * 0.5

            loss_disc = loss_invar_shape_disc + loss_gen_disc

            loss_disc.backward()

            # optimize discriminator
            Disc_optimizer.step()

            ############################
            # 2. train base network #
            ############################
            Gen_optimizer = Gen_lr_scheduler(Gen_param_lr, Gen_optimizer, epoch, config.n_epochs,
                                             1, config.decay_start_epoch, config.gen_lr)
            Gen_optimizer.zero_grad()
            Task_optimizer = Task_lr_scheduler(Task_param_lr, Task_optimizer, epoch, config.n_epochs,
                                               1, config.decay_start_epoch, config.task_lr)
            Task_optimizer.zero_grad()

            align_output_invar_shape_src = invar_shape_disc(invar_shape_output_src)
            align_output_invar_shape_tgt = invar_shape_disc(invar_shape_output_tgt)

            # loss_invar_shape_adaptation
            loss_base_invar_shape_adaptation_src = land_adaptation_criterion(align_output_invar_shape_src)
            loss_base_invar_shape_adaptation_tgt = land_adaptation_criterion(align_output_invar_shape_tgt)
            loss_invar_shape_adaptation = (
                                                  loss_base_invar_shape_adaptation_src + loss_base_invar_shape_adaptation_tgt) * 0.5

            new_gen_src_pred = feat_gen_disc_src(new_gen_src)
            loss_gen_adaptation_src = discriminator_criterion(new_gen_src_pred, real_label)

            new_gen_tgt_pred = feat_gen_disc_tgt(new_gen_tgt)
            loss_gen_adaptation_tgt = discriminator_criterion(new_gen_tgt_pred, real_label)
            # loss_gen_adaptation
            loss_gen_adaptation = (loss_gen_adaptation_src + loss_gen_adaptation_tgt) * 0.5

            loss_gen_cycle_src = reconstruct_criterion(new2_gen_src, base_feat_src.detach())
            loss_gen_cycle_tgt = reconstruct_criterion(new2_gen_tgt, base_feat_tgt.detach())
            # loss_gen_cycle
            loss_gen_cycle = (loss_gen_cycle_src + loss_gen_cycle_tgt) * 0.5

            loss_self_recons_src = reconstruct_criterion(recons_gen_src, base_feat_src.detach())
            loss_self_recons_tgt = reconstruct_criterion(recons_gen_tgt, base_feat_tgt.detach())
            # loss_self_recons
            loss_self_recons = (loss_self_recons_src + loss_self_recons_tgt) * 0.5

            loss_base_gen_au_src = au_class_criterion(au_output_src, au_src)
            loss_base_gen_au_tgt = au_class_criterion(au_output_tgt, au_tgt)
            loss_base_gen_land_src = land_predict_criterion(align_output_src, land_src)
            loss_base_gen_land_tgt = land_predict_criterion(align_output_tgt, land_tgt)

            new_gen_au_feat_src, new_gen_au_output_src = au_enc(new_gen_src)
            new_gen_au_feat_tgt, new_gen_au_output_tgt = au_enc(new_gen_tgt)
            loss_new_gen_au_src = au_class_criterion(new_gen_au_output_src, au_tgt)
            loss_new_gen_au_tgt = au_class_criterion(new_gen_au_output_tgt, au_src)
            loss_new_gen_land_src = land_predict_criterion(new_gen_align_output_src, land_tgt)
            loss_new_gen_land_tgt = land_predict_criterion(new_gen_align_output_tgt, land_src)

            # loss_land
            loss_land = (
                                    loss_base_gen_land_src + loss_base_gen_land_tgt + loss_new_gen_land_src + loss_new_gen_land_tgt) * 0.5
            # loss_au
            if config.mode == 'weak':
                loss_au = (loss_base_gen_au_src + loss_new_gen_au_tgt) * 0.5
            else:
                loss_au = (loss_base_gen_au_src + loss_base_gen_au_tgt + loss_new_gen_au_src + loss_new_gen_au_tgt) * 0.25

            total_loss = config.lambda_land_adv * loss_invar_shape_adaptation + \
                         config.lambda_feat_adv * loss_gen_adaptation + \
                         config.lambda_cross_cycle * loss_gen_cycle + config.lambda_self_recons * loss_self_recons + \
                         config.lambda_au * loss_au + config.lambda_land * loss_land

            total_loss.backward()
            Gen_optimizer.step()
            Task_optimizer.step()

            count = count + 1

    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--crop_size', type=int, default=176, help='crop size for images')
    parser.add_argument('--output_size', type=int, default=44, help='size for landmark response map')
    parser.add_argument('--au_num', type=int, default=6, help='number of AUs')
    parser.add_argument('--land_num', type=int, default=49, help='number of landmarks')
    parser.add_argument('--train_batch_size', type=int, default=16, help='mini-batch size for training')
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
    parser.add_argument('--invar_shape_disc', type=str, default='Land_Disc')
    parser.add_argument('--feat_gen_disc', type=str, default='Discriminator')

    # Training configuration.
    parser.add_argument('--lambda_au', type=float, default=1, help='weight for AU detection loss')
    parser.add_argument('--lambda_land', type=float, default=0.6, help='weight for landmark detection loss')
    parser.add_argument('--lambda_land_adv', type=float, default=400, help='weight for landmark adversarial loss')
    parser.add_argument('--lambda_feat_adv', type=float, default=1.2, help='weight for feature adversarial loss')
    parser.add_argument('--lambda_cross_cycle', type=float, default=40, help='weight for cross-cycle consistency loss')
    parser.add_argument('--lambda_self_recons', type=float, default=3, help='weight for self-reconstruction loss')
    parser.add_argument('--display', type=int, default=100, help='iteration gaps for displaying')
    parser.add_argument('--gen_optimizer_type', type=str, default='Adam')
    parser.add_argument('--gen_beta1', type=float, default=0.5, help='beta1 for Adam optimizer of generation')
    parser.add_argument('--gen_beta2', type=float, default=0.9, help='beta2 for Adam optimizer of generation')
    parser.add_argument('--gen_lr_type', type=str, default='lambda')
    parser.add_argument('--gen_lr', type=float, default=5e-5, help='learning rate for generation')
    parser.add_argument('--task_optimizer_type', type=str, default='Adam')
    parser.add_argument('--task_beta1', type=float, default=0.95, help='beta1 for Adam optimizer of task')
    parser.add_argument('--task_beta2', type=float, default=0.999, help='beta2 for Adam optimizer of task')
    parser.add_argument('--task_lr_type', type=str, default='lambda')
    parser.add_argument('--task_lr', type=float, default=1e-4, help='learning rate for task')
    parser.add_argument('--decay_start_epoch', type=int, default=5, help='epoch for decaying lr')

    # Directories.
    parser.add_argument('--write_path_prefix', type=str, default='data/snapshots/')
    parser.add_argument('--write_res_prefix', type=str, default='data/res/')
    parser.add_argument('--flip_reflect', type=str, default='data/list/reflect_49.txt')
    parser.add_argument('--src_train_path_prefix', type=str, default='data/list/BP4D_train')
    parser.add_argument('--src_val_path_prefix', type=str, default='data/list/BP4D_val')
    parser.add_argument('--src_test_path_prefix', type=str, default='data/list/BP4D_test')
    parser.add_argument('--tgt_train_path_prefix', type=str, default='data/list/emotioNet_train')
    parser.add_argument('--tgt_val_path_prefix', type=str, default='data/list/emotioNet_val')
    parser.add_argument('--tgt_test_path_prefix', type=str, default='data/list/emotioNet_test')

    config = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id

    print(config)
    main(config)