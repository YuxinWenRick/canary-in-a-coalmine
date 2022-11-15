import torch
import torch.nn as nn
import numpy as np

import torchvision.transforms as transforms

import os
import argparse
import copy
import wandb
import random
from pynvml import *

from utils import *
from models.inferencemodel import *


def generate_class_dict(args):
    dataset_class_dict = [[] for _ in range(args.num_classes)]
    for i in range(len(args.aug_trainset)):
        _, tmp_class = args.aug_trainset[i]
        dataset_class_dict[tmp_class].append(i)

    return dataset_class_dict


def generate_close_imgs(args):
    canaries = []
    target_class_list = args.dataset_class_dict[args.target_img_class]

    if args.aug_strategy and 'same_class_imgs' in args.aug_strategy:
        # assume always use the target img
        canaries = [args.aug_trainset[args.target_img_id][0].unsqueeze(0)]
        for i in range(args.num_gen - 1):
            img_id = random.sample(target_class_list, 1)[0]
            x = args.aug_trainset[img_id][0]
            x = x.unsqueeze(0)

            canaries.append(x)
    elif args.aug_strategy and 'nearest_imgs' in args.aug_strategy:
        similarities = []
        target_img = args.aug_trainset[args.target_img_id][0]
        canaries = []
        for i in target_class_list:
            similarities.append(torch.abs(target_img - args.aug_trainset[i][0]).sum())
        
        top_k_indx = np.argsort(similarities)[:(args.num_gen)]
        target_class_list = np.array(target_class_list)
        final_list = target_class_list[top_k_indx]

        for i in final_list:
            canaries.append(args.aug_trainset[i][0].unsqueeze(0))
    
    return canaries


def initialize_poison(args):
    """Initialize according to args.init.
    Propagate initialization in distributed settings.
    """
    if args.aug_strategy and ('same_class_imgs' in args.aug_strategy or 'nearest_imgs' in args.aug_strategy):
        if 'dataset_class_dict' not in args:
            args.dataset_class_dict = generate_class_dict(args)

        fixed_target_img = generate_close_imgs(args)
        args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)
    else:
        fixed_target_img = generate_aug_imgs(args)
        args.fixed_target_img = torch.cat(fixed_target_img, dim=0).to(args.device)

    # ds has to be placed on the default (cpu) device, not like self.ds
    dm = torch.tensor(args.data_mean)[None, :, None, None]
    ds = torch.tensor(args.data_std)[None, :, None, None]
    if args.init == 'zero':
        init = torch.zeros(args.num_gen, *args.canary_shape)
    elif args.init == 'rand':
        init = (torch.rand(args.num_gen, *args.canary_shape) - 0.5) * 2
        init *= 1 / ds
    elif args.init == 'randn':
        init = torch.randn(args.num_gen, *args.canary_shape)
        init *= 1 / ds
    elif args.init == 'normal':
        init = torch.randn(args.num_gen, *args.canary_shape)
    elif args.init == 'target_img':
        # init = torch.zeros(args.num_gen, *args.canary_shape).to(args.device)
        # init.data[:] = copy.deepcopy(args.canary_trainset[args.target_img_id][0])
        init = copy.deepcopy(args.fixed_target_img)
        init.requires_grad = True
        return init
    else:
        raise NotImplementedError()

    init = init.to(args.device)
    dm = dm.to(args.device)
    ds = ds.to(args.device)

    if args.epsilon:
        x_diff = init.data - args.fixed_target_img.data
        x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
                                                ds / 255), -args.epsilon / ds / 255)
        x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
                                                args.fixed_target_img), -dm / ds - args.fixed_target_img)
        init.data = args.fixed_target_img.data + x_diff.data
    else:
        init = torch.max(torch.min(init, (1 - dm) / ds), -dm / ds)

    init.requires_grad = True

    return init


def generate_canary_one_shot(shadow_models, args, return_loss=False):
    target_img_class = args.target_img_class

    # get loss functions
    args.in_criterion = get_attack_loss(args.in_model_loss)
    args.out_criterion = get_attack_loss(args.out_model_loss)

    # initialize patch
    x = initialize_poison(args)
    y = torch.tensor([target_img_class] * args.num_gen).to(args.device)

    dm = torch.tensor(args.data_mean)[None, :, None, None].to(args.device)
    ds = torch.tensor(args.data_std)[None, :, None, None].to(args.device)
    
    # initialize optimizer
    if args.opt.lower() in ['adam', 'signadam']:
        optimizer = torch.optim.Adam([x], lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() in ['sgd', 'signsgd']:
        optimizer = torch.optim.SGD([x], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.opt.lower() in ['adamw']:
        optimizer = torch.optim.AdamW([x], lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduling:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.iter // 2.667, args.iter // 1.6,
                                                                                    args.iter // 1.142], gamma=0.1)
    else:
        scheduler = None
        
    # generate canary
    last_loss = 100000000000
    trigger_times = 0
    loss = torch.tensor([0.0], requires_grad=True)

    for step in range(args.iter):
        # choose shadow models
        curr_shadow_models = get_curr_shadow_models(shadow_models, x, args)

        for _ in range(args.inner_iter):
            loss, in_loss, out_loss, reg_norm = calculate_loss(x, y, curr_shadow_models, args)
            optimizer.zero_grad()
            # loss.backward()
            if loss != 0:
                x.grad,  = torch.autograd.grad(loss, [x])
            if args.opt.lower() in ['signsgd', 'signadam'] and x.grad is not None:
                x.grad.sign_()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
            
            # projection
            with torch.no_grad():
                if args.epsilon:
                    x_diff = x.data - args.fixed_target_img.data
                    x_diff.data = torch.max(torch.min(x_diff, args.epsilon /
                                                            ds / 255), -args.epsilon / ds / 255)
                    x_diff.data = torch.max(torch.min(x_diff, (1 - dm) / ds -
                                                            args.fixed_target_img), -dm / ds - args.fixed_target_img)
                    x.data = args.fixed_target_img.data + x_diff.data
                else:
                    x.data = torch.max(torch.min(x, (1 - dm) / ds), -dm / ds)
            
            if args.print_step:
                print(f'step: {step}, ' + 'loss: %.3f, in_loss: %.3f, out_loss: %.3f, reg_loss: %.3f' % (loss, in_loss, out_loss, reg_norm))

        if args.stop_loss is not None and loss <= args.stop_loss:
            break

        if args.early_stop and loss > last_loss:
            trigger_times += 1

            if trigger_times >= args.patience:
                break
        else:
            trigger_times = 0

            # if loss == 0:
            #     break

        last_loss = loss.item()

    if return_loss:
        return x.detach(), loss.item()
    else:
        return x.detach()


def generate_canary(shadow_models, args):
    canaries = []

    if args.aug_strategy is not None:
        rand_start = random.randrange(args.num_classes)

        for out_target_class in range(1000): # need to be simplified later
            if args.canary_aug:
                args.target_img, args.target_img_class = args.aug_trainset[args.target_img_id]
                args.target_img = args.target_img.unsqueeze(0).to(args.device)

            if 'try_all_out_class' in args.aug_strategy:
                out_target_class = (rand_start + out_target_class) % args.num_classes
            elif 'try_random_out_class' in args.aug_strategy:
                out_target_class = random.randrange(args.num_classes)
            elif 'try_random_diff_class' in args.aug_strategy:
                pass
            else:
                raise NotImplementedError()

            if out_target_class != args.target_img_class:
                if args.print_step:
                    print(f'Try class: {out_target_class}')

                if 'try_random_diff_class' in args.aug_strategy:
                    out_target_class = []
                    for _ in range(args.num_gen):
                        a = random.randrange(args.num_classes)
                        while a == args.target_img_class:
                            a = random.randrange(args.num_classes)

                        out_target_class.append(a)
                    
                    args.out_target_class = torch.tensor(out_target_class).to(args.device)
                else:
                    args.out_target_class = out_target_class

                x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)

                canaries.append(x)
                args.canary_losses[-1].append(loss)

            if sum([len(canary) for canary in canaries]) >= args.num_aug:
                break
    else:
        x, loss = generate_canary_one_shot(shadow_models, args, return_loss=True)
        canaries.append(x)
        args.canary_losses[-1].append(loss)
        
    return canaries


def main(args):
    usewandb = not args.nowandb
    if usewandb:
        wandb.init(project='canary_generation',name=args.save_name)
        wandb.config.update(args)

    print(args)
    
    # set random seed
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    tv_dataset = get_dataset(args)
    
    # load shadow and target models
    shadow_models = []
    for i in range(args.num_shadow):
        curr_model = InferenceModel(i, args).to(args.device)
        shadow_models.append(curr_model)

    target_model = InferenceModel(-1, args).to(args.device)

    args.target_model = target_model
    args.shadow_models = shadow_models

    # dataset
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        transform_train = transforms.Compose([
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
        ])
    trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_train)

    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        transform_train = transforms.Compose([
        transforms.Resize(args.canary_size),
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
        ])
    args.canary_trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_train)

    if args.dataset == 'mnist':
        transform_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
        ])
    else:
        if args.no_dataset_aug:
            transform_aug = transforms.Compose([
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize(args.data_mean, args.data_std),
            ])
        else:
            transform_aug = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(args.data_mean, args.data_std),
            ])
    args.aug_trainset = tv_dataset(root='./data', train=True, download=True, transform=transform_aug)
    args.aug_testset = tv_dataset(root='./data', train=False, download=True, transform=transform_aug)

    args.img_shape = trainset[0][0].shape
    args.canary_shape = args.canary_trainset[0][0].shape

    args.pred_logits = [] # N x (num of shadow + 1) x num_trials x num_class (target at -1)
    args.in_out_labels = [] # N x (num of shadow + 1)
    args.canary_losses = [] # N x num_trials
    args.class_labels = []  # N
    args.img_id = [] # N

    for i in range(args.start, args.end):
        args.target_img_id = i

        args.target_img, args.target_img_class = trainset[args.target_img_id]
        args.target_img = args.target_img.unsqueeze(0).to(args.device)

        args.in_out_labels.append([])
        args.canary_losses.append([])
        args.pred_logits.append([])

        if args.num_val:
            in_models, out_models = split_shadow_models(shadow_models, args.target_img_id)
            num_in = min(int(args.num_val / 2), len(in_models))
            num_out = args.num_val - num_in

            train_shadow_models = random.sample(in_models, num_in)
            train_shadow_models += random.sample(out_models, num_out)

            val_shadow_models = train_shadow_models
        else:
            train_shadow_models = shadow_models
            val_shadow_models = shadow_models

        if args.aug_strategy and 'baseline' in args.aug_strategy:
            curr_canaries = generate_aug_imgs(args)
        else:
            curr_canaries = generate_canary(train_shadow_models, args)

        # get logits
        curr_canaries = torch.cat(curr_canaries, dim=0).to(args.device)
        for curr_model in val_shadow_models:
            args.pred_logits[-1].append(get_logits(curr_canaries, curr_model))
            args.in_out_labels[-1].append(int(args.target_img_id in curr_model.in_data))

        args.pred_logits[-1].append(get_logits(curr_canaries, target_model))
        args.in_out_labels[-1].append(int(args.target_img_id in target_model.in_data))

        args.img_id.append(args.target_img_id)
        args.class_labels.append(args.target_img_class)

        progress_bar(i, args.end - args.start)


    # accumulate results
    pred_logits = np.array(args.pred_logits)
    in_out_labels = np.array(args.in_out_labels)
    canary_losses = np.array(args.canary_losses)
    class_labels = np.array(args.class_labels)
    img_id = np.array(args.img_id)

    # save predictions
    os.makedirs(f'saved_predictions/{args.name}/', exist_ok=True)
    np.savez(f'saved_predictions/{args.name}/{args.save_name}.npz', pred_logits=pred_logits, in_out_labels=in_out_labels, canary_losses=canary_losses, class_labels=class_labels,
            img_id=img_id)


    ### dummy calculatiton of auc and acc
    ### to be simplified
    pred = np.load(f'saved_predictions/{args.name}/{args.save_name}.npz')

    pred_logits = pred['pred_logits']
    in_out_labels = pred['in_out_labels']
    canary_losses = pred['canary_losses']
    class_labels = pred['class_labels']
    img_id = pred['img_id']

    in_out_labels = np.swapaxes(in_out_labels, 0, 1).astype(bool)
    pred_logits = np.swapaxes(pred_logits, 0, 1)

    scores = calibrate_logits(pred_logits, class_labels, args.logits_strategy)

    shadow_scores = scores[:-1]
    target_scores = scores[-1:]
    shadow_in_out_labels = in_out_labels[:-1]
    target_in_out_labels = in_out_labels[-1:]

    some_stats = cal_results(shadow_scores, shadow_in_out_labels, target_scores, target_in_out_labels, logits_mul=args.logits_mul)

    print(some_stats)

    if usewandb:
        wandb.log(some_stats)

    if not args.save_preds:
        os.remove(f'saved_predictions/{args.name}/{args.save_name}.npz')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Gen Canary')
    parser.add_argument('--bs', default=512, type=int)
    parser.add_argument('--size', default=32, type=int)
    parser.add_argument('--canary_size', default=32, type=int)
    parser.add_argument('--name', default='test')
    parser.add_argument('--save_name', default='test')
    parser.add_argument('--num_shadow', default=None, type=int, required=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--net', default='res18')
    parser.add_argument('--patch', default=4, type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default=512, type=int)
    parser.add_argument('--convkernel', default=8, type=int, help="parameter for convmixer")
    parser.add_argument('--init', default='rand')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--iter', default=100, type=int)
    parser.add_argument('--scheduling', action='store_true')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=50000, type=int)
    parser.add_argument('--in_model_loss', default='ce', type=str)
    parser.add_argument('--out_model_loss', default='ce', type=str)
    parser.add_argument('--stop_loss', default=None, type=float)
    parser.add_argument('--print_step', action='store_true')
    parser.add_argument('--out_target_class', default=None, type=int)
    parser.add_argument('--aug_strategy', default=None, nargs='+')
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--num_aug', default=1, type=int)
    parser.add_argument('--logits_mul', default=1, type=int)
    parser.add_argument('--logits_strategy', default='log_logits')
    parser.add_argument('--in_model_loss_weight', default=1, type=float)
    parser.add_argument('--out_model_loss_weight', default=1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--reg_lambda', default=0.001, type=float)
    parser.add_argument('--regularization', default=None)
    parser.add_argument('--stochastic_k', default=None, type=int)
    parser.add_argument('--in_stop_loss', default=None, type=float)
    parser.add_argument('--out_stop_loss', default=None, type=float)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--inner_iter', default=1, type=int)
    parser.add_argument('--canary_aug', action='store_true')
    parser.add_argument('--num_val', default=None, type=int)
    parser.add_argument('--num_gen', default=1, type=int) # number of canaries generated during opt
    parser.add_argument('--epsilon', default=1, type=float)
    parser.add_argument('--no_dataset_aug', action='store_true')
    parser.add_argument('--balance_shadow', action='store_true')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--target_logits', default=None, nargs='+', type=float)
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--offline', action='store_true')

    args = parser.parse_args()
    
    main(args)
