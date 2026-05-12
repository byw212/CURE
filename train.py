import argparse
import torch.nn.functional as F
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, Adam
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.GGCD_utils import get_feature
from torch.utils.data import Subset
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator, \
    get_params_groups, vit_base
from util.GGCD_utils import get_feature, prepare_training

torch.manual_seed(0)

def Pretrain(student, init_loader, finetune_loader, args):
    for name, m in student[0].named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block_FT:
                m.requires_grad = True
    regularized = []
    not_regularized = []
    for name, param in student[0].named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    params_groups1 =  [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    regularized1 = []
    not_regularized1 = []
    for name, param in student[1].named_parameters():
        if not param.requires_grad:

            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:

            not_regularized1.append(param)
        else:

            regularized1.append(param)
    params_groups2 =  [{'params': regularized1}, {'params': not_regularized1, 'weight_decay': 0.}]
    lr1 = 0.01
    lr2 = 0.02

    optimizer1 = SGD(params_groups1, lr=lr1, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = SGD(params_groups2, lr=lr2, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler1 = lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=args.pratrain_epochs,
        eta_min=args.lr* 1e-3,
    )
    exp_lr_scheduler2 = lr_scheduler.CosineAnnealingLR(
        optimizer2,
        T_max=args.pratrain_epochs,
        eta_min=args.lr* 1e-3,
    )
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.pratrain_epochs):
        loss_record = AverageMeter()
        student.train()

        for batch_idx, batch in enumerate(finetune_loader):

            images, class_labels, uq_idxs = batch
            #images = torch.cat(images, dim=0).to(device)
            images = images.to(device)

            class_labels = class_labels.to(device)
            #class_labels = torch.cat([class_labels,class_labels], dim=0).to(device)
            with (torch.cuda.amp.autocast(fp16_scaler is not None)):
                proj_pre, x, logits = student(images.to(device))
                W = student[1].last_layer.weight
                mlp_prototype = student[1].mlp(W[class_labels])
                student_proj = torch.cat((proj_pre, mlp_prototype), dim=0)
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,device=device)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([proj_pre.unsqueeze(1),mlp_prototype.unsqueeze(1)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels)

                ce_loss = nn.CrossEntropyLoss()(logits/0.1,class_labels)
                entropy_loss = semantic_exploration_energy_loss(W)

                pstr = ''
                pstr += f'cls_loss: {ce_loss.item():.4f} '
                pstr += f'entropy_loss: {entropy_loss.item():.4f} '
                pstr += f'\n'
                loss = ce_loss + entropy_loss
                loss +=0.35*sup_con_loss + 0.65*contrastive_loss



            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if fp16_scaler is None:
                loss.backward()
                optimizer1.step()
                optimizer2.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer1)
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
        exp_lr_scheduler1.step()
        exp_lr_scheduler2.step()
        save_dict = {
            'model': student.state_dict()
        }
        torch.save(save_dict, args.model_path)
    return student


def semantic_exploration_energy_loss(prototypes, epsilon=0.9, lambda_=1):
    # prototypes: [num_classes, dim]
    norm_prototypes = F.normalize(prototypes, dim=1)
    sim_matrix = torch.matmul(norm_prototypes, norm_prototypes.T)  # [num_classes, num_classes]

    # mask self-similarity
    mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=prototypes.device)
    sims = sim_matrix[mask]  # (N*(N-1)) values

    # energy = sum_{i<j} (1 - cos)^2
    energy = torch.mean((1 - sims)**2)

    # soft hinge loss version
    energy_loss = torch.log(1 + torch.exp(lambda_ * (energy - epsilon)))
    return energy_loss



def train(teacher, train_loader, init_loader, unlabelled_train_loader, args):
    params_groups1 = get_params_groups(teacher)

    optimizer1 = SGD(params_groups1, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler1 = lr_scheduler.CosineAnnealingLR(
        optimizer1,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    # optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
    )

    # # inductive
    best_test_acc_lab = 0

    for epoch in range(args.epochs):

        num = 0
        n = 0
        information = get_feature(teacher, init_loader, device)
        loss_record = AverageMeter()
        if epoch % 2 == 0:
            weights, category_counts = prepare_training(information, args, epoch, mode='ward')
            print(weights.shape)
            weights = torch.tensor(weights).to(device)
            category_counts1 = torch.tensor(category_counts).to(device)
            log_prior1 = torch.log(category_counts1)
            print(weights.shape)
            weights = torch.tensor(weights).to(device)

            weights.requires_grad = True
            optimizer_w1 = SGD([weights], lr=0.01,
                               momentum=args.momentum, weight_decay=args.weight_decay)

        teacher.train()

        for batch_idx, batch in enumerate(train_loader):

            n += 1
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            mask = torch.cat([mask_lab, mask_lab], dim=0)
            images = torch.cat(images, dim=0).to(device)

            with ((torch.cuda.amp.autocast(fp16_scaler is not None))):
                norm_w = torch.nn.functional.normalize(weights, dim=-1)
                student_proj, teacher_feat, logits_L = teacher(images)

                logits1 = teacher_feat @ weights.T / 0.1
                w_loss = (norm_w @ norm_w.T).mean()
                softmax1 = logits1.softmax(dim=1)
                aug1, aug2 = [f for f in softmax1.chunk(2)]
                KL_loss = F.kl_div(aug1.log(), aug2, reduction='batchmean')
                KL_loss += F.kl_div(aug2.log(), aug1, reduction='batchmean')
                teacher_out1 = logits1.detach()

                # clustering, sup
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                sup_logits1 = torch.cat([f[mask_lab] for f in logits1.chunk(2)], dim=0)

                cls_loss = nn.CrossEntropyLoss()(sup_logits1, sup_labels)
                cluster_loss = cluster_criterion(logits1[~mask], teacher_out1[~mask],log_prior1,epoch)

                P1 = 1.2

                R1 = torch.cat([P1 * torch.ones(args.num_labeled_classes), torch.ones(args.num_unlabeled_classes)]).to(
                    device)

                avg_probs1 = (logits1*R1).softmax(dim=1).mean(dim=0)
                me_max_loss = -torch.sum(torch.log(avg_probs1 ** (-avg_probs1))) + math.log(float(len(avg_probs1)))
                cluster_loss += args.memax_weight * me_max_loss
                loss = 0
                entropy_loss = semantic_exploration_energy_loss(weights)

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'w_loss: {w_loss.item():.4f} '
                pstr += '\n'
                pstr += f'KL_loss: {KL_loss.item():.4f} '
                pstr += f'entropy_loss: {entropy_loss:.4f} '

                loss += 0.5 * cluster_loss + 0.5 * cls_loss
                loss += 0.5* KL_loss + entropy_loss*args.lambdas

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer1.zero_grad()
            optimizer_w1.zero_grad()

            if fp16_scaler is None:
                loss.backward()
                optimizer1.step()
                optimizer_w1.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer1)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                print(num / n)
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                                 .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(teacher, weights, unlabelled_train_loader, epoch=epoch,
                                         save_name='Train ACC Unlabelled', args=args)
        # args.logger.info('Testing on disjoint test set...')
        # all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        # args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler1.step()

        save_dict = {
            'model': teacher.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }

        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        if old_acc > best_test_acc_lab:
            # args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            args.logger.info(
                'Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))

            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))

            # inductive
            best_test_acc_lab = old_acc
            # transductive
            best_train_acc_lab = old_acc
            best_train_acc_ubl = new_acc
            best_train_acc_all = all_acc

        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(
            f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')


def test(model, weight, test_loader, epoch, save_name, args):
    teacher = model

    teacher.eval()

    preds, targets = [], []
    mask = np.array([])
    result = []
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.to(device)
        with (torch.no_grad()):
            _, t_feat, _ = teacher(images)
            logits = t_feat @ weight.T
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask,
                             np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    targets = np.concatenate(targets)

    preds = np.concatenate(preds)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='cub',
                        help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--grad_from_block_FT', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--lambdas', type=float, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--pratrain_epochs', type=int, default=30)

    parser.add_argument('--memax_weight', type=float, default=1)
    parser.add_argument('--warmup_teacher_temp', default=0.7, type=float,
                        help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.4, type=float,
                        help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='GGCD', type=str)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args.device = device
    args = get_class_splits(args)

    args.num_labeled_classes: int = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[args.exp_name])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')

    torch.backends.cudnn.benchmark = True

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    pretrain_path = dino_pretrain_path
    model = vit_base()
    state_dict = torch.load(pretrain_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in model.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    init_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=init_transform, n_views=args.n_views)
    FT_transform  = ContrastiveLearningViewGenerator(base_transform=init_transform , n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else 1 for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)

    length = len(train_dataset)
    k = 200
    num_samples = args.num_labeled_classes * k if args.num_labeled_classes * k < length else length
    print(num_samples,length)
    samples = np.random.choice(list(range(length)), size=num_samples, replace=False)
    init_dataset = deepcopy(train_dataset)
    init_dataset.unlabelled_dataset.transform = init_transform
    init_dataset.labelled_dataset.transform = init_transform
    prototype_G_dataset = Subset(init_dataset, samples)
    init_loader = DataLoader(prototype_G_dataset, num_workers=args.num_workers, batch_size=512, shuffle=False,
                             drop_last=False)
    finetune_dataset = deepcopy(train_dataset.labelled_dataset)
    finetune_dataset.transform = test_transform
    finetune_loader = DataLoader(finetune_dataset, num_workers=args.num_workers,
                                 batch_size=128, shuffle=True, pin_memory=False)
    # test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------

    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.num_labeled_classes, nlayers=1)
    model = nn.Sequential(model, projector).to(device)
    model = model.to(device)

    # ----------------------
    # TRAIN
    # ----------------------

    model = Pretrain(model, finetune_loader, args)
    for m in model[0].parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model[0].named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    train(model, train_loader, init_loader, test_loader_unlabelled, args)