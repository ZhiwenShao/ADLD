import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score

def AU_detection_eval_src(loader, base_net, au_enc, use_gpu=True):
    missing_label = 999
    for i, batch in enumerate(loader):
        input, label = batch
        if use_gpu:
            input, label = input.cuda(), label.cuda()

        base_feat = base_net(input)
        au_feat, au_output = au_enc(base_feat)
        au_output = F.sigmoid(au_output)
        if i == 0:
            all_output = au_output.data.cpu().float()
            all_label = label.data.cpu().float()
        else:
            all_output = torch.cat((all_output, au_output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_label.data.numpy()

    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr


def AU_detection_eval_tgt(loader, base_net, land_enc, au_enc, invar_shape_enc, feat_gen, use_gpu=True):
    missing_label = 999
    for i, batch in enumerate(loader):
        input, label = batch
        if use_gpu:
            input, label = input.cuda(), label.cuda()

        base_feat = base_net(input)
        align_attention, align_feat, align_output = land_enc(base_feat)
        invar_shape_output = invar_shape_enc(base_feat)
        new_gen = feat_gen(align_attention, invar_shape_output)
        new_gen_au_feat, new_gen_au_output = au_enc(new_gen)
        au_output = F.sigmoid(new_gen_au_output)
        if i == 0:
            all_output = au_output.data.cpu().float()
            all_label = label.data.cpu().float()
        else:
            all_output = torch.cat((all_output, au_output.data.cpu().float()), 0)
            all_label = torch.cat((all_label, label.data.cpu().float()), 0)

    AUoccur_pred_prob = all_output.data.numpy()
    AUoccur_actual = all_label.data.numpy()

    AUoccur_pred = np.zeros(AUoccur_pred_prob.shape)
    AUoccur_pred[AUoccur_pred_prob < 0.5] = 0
    AUoccur_pred[AUoccur_pred_prob >= 0.5] = 1

    AUoccur_actual = AUoccur_actual.transpose((1, 0))
    AUoccur_pred = AUoccur_pred.transpose((1, 0))

    f1score_arr = np.zeros(AUoccur_actual.shape[0])
    acc_arr = np.zeros(AUoccur_actual.shape[0])
    for i in range(AUoccur_actual.shape[0]):
        curr_actual = AUoccur_actual[i]
        curr_pred = AUoccur_pred[i]

        new_curr_actual = curr_actual[curr_actual != missing_label]
        new_curr_pred = curr_pred[curr_actual != missing_label]

        f1score_arr[i] = f1_score(new_curr_actual, new_curr_pred)
        acc_arr[i] = accuracy_score(new_curr_actual, new_curr_pred)

    return f1score_arr, acc_arr


def land_softmax_loss(input, target, weight=None, size_average=True, reduce=True):
    classify_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, reduce=reduce)

    for i in range(input.size(1)):
        t_input = input[:, i, :, :]
        t_input = t_input.view(t_input.size(0), -1)
        t_target = target[:, i]
        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def land_adaptation_loss(input, size_average=True, reduce=True):
    classify_loss = nn.MSELoss(size_average=size_average, reduce=reduce)
    use_gpu = torch.cuda.is_available()

    for i in range(input.size(1)):
        t_input = input[:, i, :, :]
        t_input = t_input.view(t_input.size(0), -1)
        t_target = torch.ones(t_input.size()) * 1.0 / t_input.size(1)
        if use_gpu:
            t_target = t_target.cuda()

        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()


def land_discriminator_loss(input, target, size_average=True, reduce=True):
    classify_loss = nn.MSELoss(size_average=size_average, reduce=reduce)
    use_gpu = torch.cuda.is_available()
    for i in range(input.size(1)):
        t_input = input[:, i, :, :]
        t_input = t_input.view(t_input.size(0), -1)

        t_target = torch.zeros(t_input.size())
        if use_gpu:
            t_target = t_target.cuda()
        t_true_target = target[:, i]

        for j in range(t_true_target.size(0)):
            t_target[j, t_true_target[j]] = 1

        t_loss = classify_loss(t_input, t_target)
        t_loss = torch.unsqueeze(t_loss, 0)
        if i == 0:
            loss = t_loss
        else:
            loss = torch.cat((loss, t_loss), 0)

    if size_average:
        return loss.mean()
    else:
        return loss.sum()