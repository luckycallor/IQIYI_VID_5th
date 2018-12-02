import time
import random
import pickle
import numpy as np
from sklearn.cluster import DBSCAN


def load_labels(fn):
    label_dict = {}
    with open(fn) as f:
        for l in f.readlines():
            l = l.strip().split(' ')
            label_dict[l[0]] = int(l[1])
    return label_dict


def feat_weighted_mean(feats, weight_idx=3):
    fs = []
    ws = []
    for f in feats:
        fs.append(f[-1]*f[weight_idx])
        ws.append(f[weight_idx])
    fs = np.array(fs)
    ws = np.array(ws)
    ws = ws/np.sum(ws)
    f = np.sum(fs*ws[:, None], axis=0)
    return f


def load_noise(feat_path, qua_thred=0.0, det_thred=0.0, class_num=4934):
    feat_dict = pickle.load(open(feat_path, 'rb'), encoding='bytes')
    vns = []
    feats = []
    labels = []
    for vn in feat_dict.keys():
        fs = feat_dict[vn]
        if(len(fs) == 0):
            continue
        filtered_fs = [f for f in fs if f[2]>=det_thred and f[3]>=qua_thred]
        if(len(filtered_fs) == 0):
            feats.append(feat_weighted_mean(fs))
        else:
            feats.append(feat_weighted_mean(filtered_fs))
        vns.append(vn)
        labels.append(random.randint(class_num, class_num+class_num-1))
    return vns, feats, labels


if __name__ == '__main__':
    noise_paths = ['/data/hhd/iqiyi/feats_val_noise.pickle', '/data/hhd/iqiyi/feats_noise_aug_100.pickle']
    eps = 1.2
    min_samples = 2
    n_jobs=16
    save_path = '/data/hhd/iqiyi/noise_labels_100.txt'

    print('load data...')
    start_time = time.time()
    vns = []
    feats = []
    for p in noise_paths:
        cur_vns, cur_feats, _ = load_noise(p, 0.0, 0.0)
        vns += cur_vns
        feats += cur_feats
    print('done![time: %f]' % (time.time()-start_time))
    print('noise num: %d' % len(vns))

    print('cluster...')
    start_time = time.time()
    m = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    cluster_labels = m.fit_predict(np.array(feats))
    print('done![time: %f]' % (time.time()-start_time))

    print('save result...')
    start_time = time.time()
    with open(save_path, 'w') as f:
        for i in range(len(vns)):
            f.write(vns[i]+' '+str(int(cluster_labels[i]))+'\n')
    print('done![time: %f]' % (time.time()-start_time))
    l_list = list(set(cluster_labels))
    print('===============================')
    print('label len: %d' % len(l_list))
    print('label max: %d' % max(l_list))
    print('label min: %d' % min(l_list))
    print('noise rate: %f' % np.mean(cluster_labels == -1))
    print('===============================')
    











