import os
import torch
import torch.nn.functional as F
import inspect
from statistics import mean
from sklearn.decomposition import PCA
import numpy as np
from torch.optim import SGD, lr_scheduler,Adam
from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
from util.cluster_and_log_utils import log_accs_from_preds
import matplotlib.pyplot as plt
import torch.autograd as autograd

def get_feature(model,dataset,device,mode='training'):
    print('Prepare for extracting features')
    length = len(dataset)
    data_iter = iter(dataset)
    model.eval()
    with torch.no_grad():
        if mode == 'FT':
            x, y, idx = next(data_iter)
            #x = torch.cat(x,dim=0)
        else:
            x, y, idx, z = next(data_iter)
            z = z[:, 0]
            mask = z.to(device).bool()
        uq_idx = idx.to(device)
        label = y.to(device)

        _, features, _ = model(x.to(device))
        #features = torch.nn.functional.normalize(features, dim=-1)
    for i in range(1, length):
        with torch.no_grad():
            if mode == 'FT':
                x, y, idx = next(data_iter)
            else:
                x, y, idx, z = next(data_iter)
                z = z[:, 0].bool()
                z = z.to(device)
            y = y.to(device)
            _,feature,_ = model(x.to(device))
            #feature = torch.nn.functional.normalize(feature, dim=-1)
            features = torch.cat([features, feature], dim=0)
            label = torch.cat([label, y], dim=0)
            if mode != 'FT':
                mask = torch.cat([mask, z], dim=0)

    #features[features.abs()<0.5]=0
    #print(features.shape)
    if mode == 'FT':
        information = (features,label,None)
    else:
        information = (features, label,mask)
    print('Extract features successfully')

    return information

def prepare_training(information,args,epoch,mode='ward'):
    def transform(data):
        try:
            data = data.detach().cpu().numpy()
            return data
        except:
            return data
    feats, targets, mask = (transform(x) for x in information)

    print('start clustering')
    if mode == 'FT':
        K = args.num_labeled_classes
    else:
        K = args.mlp_out_dim

    #hierarchy
    linked = linkage(feats, method=mode)
    dist = linked[:, 2]#[:-args.num_labeled_classes]
    d = dist[-K]
    preds = fcluster(linked, t=d, criterion='distance')

    #kmeans
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=K, random_state=42, init='k-means++')
    # kmeans.fit(feats)
    #
    # # 3. 获取聚类标签与中心点
    # preds = kmeans.labels_+1

    #GaussianMixture
    # from sklearn.mixture import GaussianMixture
    # gmm = GaussianMixture(n_components=K, random_state=42)
    # gmm.fit(feats)
    # preds = gmm.predict(feats)+1  # 每个样本的聚类标签

    #SpectralClustering
    # from sklearn.cluster import SpectralClustering
    # sc = SpectralClustering(
    #     n_clusters=K,
    #     affinity='rbf',
    #     n_neighbors=10,
    #     assign_labels='kmeans',
    #     random_state=42
    # )
    # preds = sc.fit_predict(feats)+1




    best_acc_k = max(preds)
    class_feat = [[] for _ in range(best_acc_k)]
    reorder_w = [0 for _ in range(best_acc_k)]

    preds = preds.astype(int)
    targets = targets.astype(int)
    for num, i in enumerate(preds):
        class_feat[i - 1].append(feats[num].reshape(1, -1))
    class_feat = [np.concatenate(x) for x in class_feat]

    pseudo_w = [x.mean(axis=0).reshape(1, -1) for x in class_feat]
    D = max(preds.max(), args.num_labeled_classes) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        if mask[i]:
            w[preds[i], targets[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    flag = []
    for i in range(args.num_labeled_classes):
        reorder_w[i] = pseudo_w[ind_map[i] - 1].reshape(1,-1)
        flag.append(ind_map[i] - 1)
    flag.sort()
    for i in reversed(flag):
        pseudo_w.pop(i)
    for i in range(args.num_labeled_classes, best_acc_k):
        reorder_w[i] = pseudo_w[i-args.num_labeled_classes].reshape(1,-1)

    w = np.concatenate(reorder_w)

    f2 = np.linalg.norm(w,ord=2, axis=-1).reshape(-1,1)
    labeled_target = targets[mask]
    w = w/f2

    p_seen = np.bincount(labeled_target)/len(labeled_target)
    p_unseen = np.ones(args.num_unlabeled_classes)*np.mean(p_seen)
    category_counts = np.concatenate([p_seen,p_unseen],axis=0)

    return w,category_counts

