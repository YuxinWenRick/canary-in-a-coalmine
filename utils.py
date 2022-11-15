# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import copy
import scipy

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torchvision


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def imshow(img):
    img = img.detach().cpu()
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()


# naive test
def test_canary(curr_canary, args, return_logits=False):
    with torch.no_grad():
        logits = args.target_model(curr_canary)
        predicted_class = logits.argmax(1).item()
    
    # 1 means predicted as in data; 0 means predicted as out data
    if return_logits:
        return int(predicted_class == args.target_img_class), logits
    else:
        return int(predicted_class == args.target_img_class)


def get_logits(curr_canary, model, keep_tensor=False):
    with torch.no_grad():
        logits = model(curr_canary)
    
    if not keep_tensor:
        logits = logits.detach().cpu().tolist()

    return logits


# canary losses
def _label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.shape[0], num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cw_loss(outputs, intended_classes, confidence=0, clamp=-5):
    """CW variant 2. This is assert-level equivalent."""
    one_hot_labels = _label_to_onehot(intended_classes, num_classes=outputs.shape[1])
    target_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).max(dim=1)
    cw_indiv = torch.clamp(second_logit - target_logit + confidence, min=clamp)
    return cw_indiv.mean()

def cw_loss_reverse(outputs, intended_classes, confidence=0, clamp=-5):
    """CW variant 2. This is assert-level equivalent."""
    outputs = -outputs
    one_hot_labels = _label_to_onehot(intended_classes, num_classes=outputs.shape[1])
    target_logit = (outputs * one_hot_labels).sum(dim=1)
    second_logit, _ = (outputs - outputs * one_hot_labels).min(dim=1)
    cw_indiv = torch.clamp(second_logit - target_logit + confidence, min=clamp)
    return cw_indiv.mean()

def reverse_ce(outputs, labels):
    outputs = nn.Softmax(dim=1)(outputs)
    outputs = 1 - outputs + 0.000000001
    outputs = torch.log(outputs)
    return nn.NLLLoss()(outputs, labels)

def dummy_loss(outputs, labels):
    return torch.tensor([0.0], requires_grad=True).cuda()

def get_attack_loss(loss):
    if loss == 'ce':
        return nn.CrossEntropyLoss()
    elif loss == 'cw':
        return cw_loss
    elif loss == 'reverse_cw':
        return cw_loss_reverse
    elif loss == 'reverse_ce':
        return reverse_ce
    elif loss is None or loss == 'None':
        return dummy_loss
    elif loss == 'target_logits':
        return nn.MSELoss()
    elif loss == 'target_logits_softmax':
        return nn.MSELoss()
    elif loss == 'target_logits_log':
        return nn.MSELoss()
    else:
        raise NotImplementedError()


def normalize_logits(logits):
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    logits = np.array(np.exp(logits), dtype=np.float64)
    logits = logits / np.sum(logits,axis=-1,keepdims=True)

    return logits


def get_pure_logits(pred_logits, class_labels):
    pred_logits = copy.deepcopy(pred_logits)

    scores = []
    for pred_logits_i in pred_logits:
        score = copy.deepcopy(pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels])
        
        scores.append(score)
        
    scores = np.array(scores)

    return scores


def get_normal_logits(pred_logits, class_labels):
    pred_logits = copy.deepcopy(pred_logits)

    scores = []
    for pred_logits_i in pred_logits:
        pred_logits_i = normalize_logits(pred_logits_i)

        score = copy.deepcopy(pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels])
        
        scores.append(score)
        
    scores = np.array(scores)

    return scores


def get_log_logits(pred_logits, class_labels):
    pred_logits = copy.deepcopy(pred_logits)

    scores = []
    for pred_logits_i in pred_logits:
        pred_logits_i = normalize_logits(pred_logits_i)

        y_true = copy.deepcopy(pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels])
        pred_logits_i[np.arange(len(pred_logits_i)), :, class_labels] = 0
        y_wrong = np.sum(pred_logits_i, axis=2)
        score = (np.log(y_true+1e-45) - np.log(y_wrong+1e-45))
        
        scores.append(score)
        
    scores = np.array(scores)

    return scores


def calibrate_logits(pred_logits, class_labels, logits_strategy):
    if logits_strategy == 'pure_logits':
        scores = get_pure_logits(pred_logits, class_labels)
    elif logits_strategy == 'log_logits':
        scores = get_log_logits(pred_logits, class_labels)
    elif logits_strategy == 'normal_logits':
        scores = get_normal_logits(pred_logits, class_labels)
    else:
        raise NotImplementedError()

    return scores

'''
    Implemtation from:
    https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021
'''

def lira_online(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, fix_variance=False):
    dat_in = []
    dat_out = []

    for j in range(shadow_scores.shape[1]):
        dat_in.append(shadow_scores[shadow_in_out_labels[:, j], j, :])
        dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])
        
    in_size = min(map(len,dat_in))
    out_size = min(map(len,dat_out))

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_out)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    final_preds = []
    true_labels = []

    for ans, sc in zip(target_in_out_labels, target_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in+1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)
        score = pr_in-pr_out

        final_preds.extend(score.mean(1))
        true_labels.extend(ans)

    final_preds = np.array(final_preds)
    true_labels = np.array(true_labels)

    return -final_preds, true_labels


def lira_offline(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, fix_variance=False):
    dat_out = []

    for j in range(shadow_scores.shape[1]):
        dat_out.append(shadow_scores[~shadow_in_out_labels[:, j], j, :])
        
    out_size = min(map(len,dat_out))

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    final_preds = []
    true_labels = []

    for ans, sc in zip(target_in_out_labels, target_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out+1e-30)

        final_preds.extend(score.mean(1))
        true_labels.extend(ans)

    final_preds = np.array(final_preds)
    true_labels = np.array(true_labels)

    return -final_preds, true_labels


def cal_stats(final_preds, true_labels):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, final_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    return fpr, tpr, auc, acc, low


def cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=1, logits_strategy=None):
    some_stats = {}
    
    final_preds, true_labels = lira_online(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels,
                                        fix_variance=True)
    fpr, tpr, auc, acc, low = cal_stats(logits_mul * final_preds, true_labels)
    some_stats['fix_auc'] = auc
    some_stats['fix_acc'] = acc
    some_stats['fix_TPR@0.01FPR'] = low

    final_preds, true_labels = lira_offline(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels,
                                        fix_variance=True)
    fpr, tpr, auc, acc, low = cal_stats(logits_mul * final_preds, true_labels)
    some_stats['fix_off_auc'] = auc
    some_stats['fix_off_acc'] = acc
    some_stats['fix_off_TPR@0.01FPR'] = low

    return some_stats


def get_dataset(args):
    if args.dataset == 'cifar10':
        args.data_mean = (0.4914, 0.4822, 0.4465)
        args.data_std = (0.2023, 0.1994, 0.2010)
        args.num_classes = 10

        return torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar100':
        args.data_mean = (0.5071, 0.4867, 0.4408)
        args.data_std = (0.2675, 0.2565, 0.2761)
        args.num_classes = 100

        return torchvision.datasets.CIFAR100
    elif args.dataset == 'mnist':
        args.data_mean = (0.1307,)
        args.data_std = (0.3081,)
        args.num_classes = 10

        return torchvision.datasets.MNIST
    else:
        raise NotImplementedError()


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=1.0, inner_exp=2, outer_exp=0.5, double_opponents=True, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

        grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], **setup).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        self.groups = 6 if self.double_opponents else 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)

        self.register_buffer("weight", grad_weight)

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        """Use a convolution-based approach."""
        if self.double_opponents:
            tensor = torch.cat(
                [
                    tensor,
                    tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                    tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                    tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :],
                ],
                dim=1,
            )
        diffs = torch.nn.functional.conv2d(
            tensor, self.weight, None, stride=1, padding=1, dilation=1, groups=self.groups
        )
        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale


def calulate_reg(x, y, shadow_models, args):
    if args.regularization.lower() == 'l1':
        norm = x.abs().mean()
    elif args.regularization.lower() == 'l2':
        norm = x.pow(2.0).mean()
    elif args.regularization.lower() == 'l2_tv':
        bs_img, c_img, h_img, w_img = x.size()
        tv_h = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2).sum()
        tv_w = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2).sum()
        norm = (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)
    elif args.regularization.lower() == 'l1_tv':
        bs_img, c_img, h_img, w_img = x.size()
        tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
        tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
        norm = (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)
    elif args.regularization.lower() == 'color_tv':
        norm = TotalVariation({'device': args.device, 'dtype': x.dtype})(x)
    elif args.regularization.lower() == 'perturb_l1':
        x = x - args.fixed_target_img
        norm = x.abs().mean()
    elif args.regularization.lower() == 'perturb_l2':
        x = x - args.fixed_target_img
        norm = x.pow(2.0).mean()
    else:
        raise NotImplementedError()

    return norm


def generate_aug_imgs(args):
    canaries = []
    
    counter = args.num_aug

    for i in range(counter):
        x = args.aug_trainset[args.target_img_id][0]
        x = x.unsqueeze(0)

        canaries.append(x)
    
    return canaries


def get_log_logits_torch(logits, y):
    logits = logits - torch.max(logits, dim=-1, keepdims=True)[0]
    logits = torch.exp(logits)
    logits = logits / torch.sum(logits, dim=-1, keepdims=True)

    y_true = logits[:, y]
    num_class = logits.shape[-1]
    wrong_indx = [i for i in range(num_class) if i != y]
    y_wrong = torch.sum(logits[:, wrong_indx], dim=-1)
    logits = (torch.log(y_true+1e-45) - torch.log(y_wrong+1e-45))

    return logits


def calculate_loss(x, y, shadow_models, args):
    in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)

    losses = 0
    in_loss = 0
    out_loss = 0

    if args.out_model_loss == 'cw' or args.out_model_loss == 'ce':
        y_out = copy.deepcopy(y)
        if args.out_target_class is None:
            y_out = (y_out + 1) % args.num_classes
        else:
            y_out *= 0
            y_out += args.out_target_class
    elif args.out_target_class is not None:
            y_out = copy.deepcopy(y)
            y_out *= 0
            y_out += args.out_target_class
    else:
        y_out = y

    if 'bce' in args.in_model_loss:
        y = nn.functional.one_hot(y, args.num_classes).float()

    if 'bce' in args.out_model_loss:
        y_out = nn.functional.one_hot(y_out, args.num_classes).float()

    if 'target_logits' in args.in_model_loss:
        with torch.no_grad():
            tmp_outputs = out_models[0](x)
        y = torch.zeros(tmp_outputs.shape, device=args.device, dtype=tmp_outputs.dtype)
        y[:, args.target_img_class] += args.target_logits[0]
        y = y[:, args.target_img_class]

    if 'target_logits' in args.out_model_loss:
        with torch.no_grad():
            tmp_outputs = out_models[0](x)
        y_out = torch.zeros(tmp_outputs.shape, device=args.device, dtype=tmp_outputs.dtype)
        y_out[:, args.target_img_class] += args.target_logits[1]
        y_out = y_out[:, args.target_img_class]

    args.y_out = y_out

    in_outputs = []
    out_outputs = []
    in_models_counts = 0
    out_models_counts = 0
    for curr_model in in_models:
        outputs = curr_model(x)

        if 'kl' in args.out_model_loss or args.in_model_loss == 'kl':
            if args.in_model_loss == 'target_kl':
                in_outputs.append(F.log_softmax(-outputs)[:, y])
            elif 'target' in args.out_model_loss:
                # only works when -outputs, need to double check
                in_outputs.append(F.log_softmax(-outputs)[:, y])
            else:
                in_outputs.append(F.log_softmax(outputs))

            if 'kl' in args.in_model_loss:
                continue
        elif args.out_model_loss == 'mse':
            in_outputs.append(outputs[0, y])

            if args.in_model_loss == 'mse':
                continue
        
        if args.in_model_loss == 'target_logits':
            outputs = outputs[:, args.target_img_class]
        elif args.in_model_loss == 'target_logits_softmax':
            outputs = F.softmax(outputs)
            outputs = outputs[:, args.target_img_class]
        elif args.in_model_loss == 'target_logits_log':
            outputs = get_log_logits_torch(outputs, args.target_img_class)

        curr_loss = args.in_criterion(outputs, y)

        if args.in_stop_loss:
            if curr_loss > args.in_stop_loss:
                in_loss += curr_loss
                in_models_counts += 0
        else:
            in_loss += curr_loss
            in_models_counts += 0

    for curr_model in out_models:
        outputs = curr_model(x)

        if args.out_model_loss == 'kl' or args.in_model_loss == 'kl':
            out_outputs.append(F.log_softmax(outputs))
        elif 'target_kl' in args.out_model_loss:
            out_outputs.append(F.log_softmax(outputs)[:, y])
        elif args.out_model_loss == 'mse':
            out_outputs.append(outputs[0, y])
        else:
            if args.out_model_loss == 'target_logits':
                outputs = outputs[:, args.target_img_class]
            elif args.out_model_loss == 'target_logits_softmax':
                outputs = F.softmax(outputs)
                outputs = outputs[:, args.target_img_class]
            elif args.out_model_loss == 'target_logits_log':
                outputs = get_log_logits_torch(outputs, args.target_img_class)

            curr_loss = args.out_criterion(outputs, y_out)
            if args.out_stop_loss:
                if curr_loss > args.out_stop_loss:
                    out_loss += curr_loss
                    out_models_counts += 0
            else:
                out_loss += curr_loss
                out_models_counts += 0

    random.shuffle(in_outputs)
    random.shuffle(out_outputs)

    if 'kl' in args.out_model_loss or 'kl' in args.in_model_loss:
        min_len = min(len(in_outputs), len(out_outputs))
        in_outputs = in_outputs[:min_len]
        out_outputs = out_outputs[:min_len]

        if 'target' in args.out_model_loss:
            in_outputs = torch.cat(in_outputs).squeeze().unsqueeze(-1)
            out_outputs = torch.cat(out_outputs).squeeze().unsqueeze(-1)
        else:
            in_outputs = torch.cat(in_outputs)
            out_outputs = torch.cat(out_outputs)

        if 'kl' in args.out_model_loss:
            out_loss += args.out_criterion(out_outputs, in_outputs)

        if 'kl' in args.in_model_loss:
            in_loss += args.in_criterion(in_outputs, out_outputs)

    elif args.out_model_loss == 'mse' or args.in_model_loss == 'mse':
        min_len = min(len(in_outputs), len(out_outputs))
        in_outputs = in_outputs[:min_len]
        out_outputs = out_outputs[:min_len]
        in_outputs = torch.cat(in_outputs).squeeze().unsqueeze(-1)
        out_outputs = torch.cat(out_outputs).squeeze().unsqueeze(-1)

        if args.out_model_loss == 'mse':
            out_loss += args.out_criterion(in_outputs, out_outputs)

        if args.in_model_loss == 'mse':
            in_loss += args.in_criterion(out_outputs, in_outputs)
    
    if in_models_counts != 0:
        in_loss /= in_models_counts
    if out_models_counts != 0:
        out_loss /= out_models_counts

    if in_loss == 0 and out_loss == 0:
        losses = torch.tensor([0.0], requires_grad=True).to(args.device)
    else:
        losses = args.in_model_loss_weight * in_loss + args.out_model_loss_weight * out_loss

    if args.regularization is not None:
        reg_norm = calulate_reg(x, y, shadow_models, args)
        losses += args.reg_lambda * reg_norm
    else:
        reg_norm = 0

    return losses, in_loss, out_loss, reg_norm


def split_shadow_models(shadow_models, target_img_id):
    in_models = []
    out_models = []

    for curr_model in shadow_models:
        if target_img_id in curr_model.in_data:
            curr_model.is_in_model = True
            in_models.append(curr_model)
        else:
            curr_model.is_in_model = False
            out_models.append(curr_model)
    
    return in_models, out_models


def get_curr_shadow_models(shadow_models, x, args):
    if args.offline:
        if args.stochastic_k is None:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            return out_models
        else:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            curr_shadow_models = random.sample(out_models, args.stochastic_k)
            return curr_shadow_models

    if args.stochastic_k is None:
        curr_shadow_models = shadow_models
    else:
        if args.stochastic_k > 1:
            # more balanced for kl loss
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)

            num_in = int(args.stochastic_k / 2)
            num_out = args.stochastic_k - num_in
            
            curr_shadow_models = random.sample(in_models, num_in)
            curr_shadow_models += random.sample(out_models, num_out)
        elif args.balance_shadow:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            min_len = min(len(in_models), len(out_models))

            curr_shadow_models = random.sample(in_models, min_len)
            curr_shadow_models += random.sample(out_models, min_len)
        else:
            curr_shadow_models = random.sample(shadow_models, args.stochastic_k)


    return curr_shadow_models
