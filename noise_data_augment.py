import pickle
import random
import numpy as np
from itertools import product
from datetime import datetime


def combine_to_augment(vns, num, data):
    augmented_data = []
    ori_vns = list(vns)
    for i in range(int(num/len(vns))):
        random.shuffle(vns)
        augmented_data += [data[ori_vns[j]]+data[vns[j]] for j in range(len(vns))]
    left = int(num%len(vns))
    if left > 0:
        ori_vns = random.sample(ori_vns, left)
        vns = random.sample(vns, left)
        augmented_data += [data[ori_vns[j]]+data[vns[j]] for j in range(len(vns))]
    return augmented_data




def random_split(feats):
    start = random.randint(0, len(feats)-1)
    end = random.randint(start+1, len(feats))
    return feats[start: end]


def split_to_augment(vns, num, data):
    augmented_data = []
    for i in range(int(num/len(vns))):
        augmented_data += [random_split(data[vn]) for vn in vns]
    left_vns = random.sample(vns, int(num%len(vns)))
    augmented_data += [random_split(data[vn]) for vn in left_vns]
    return augmented_data


# augment for val: 63884
# augment for train: 159135


augment_num = 500000
noise_data_path = '/data/hhd/iqiyi/feats_val_noise.pickle'
save_path = '/data/hhd/iqiyi/feats_noise_aug_100.pickle'

noise_data = pickle.load(open(noise_data_path, 'rb'), encoding='bytes')
noise_vns = list(noise_data.keys())
noise_feat_lens = [len(noise_data[vn]) for vn in noise_vns]
noise_vns = np.array(noise_vns)
noise_feat_lens = np.array(noise_feat_lens)

short_noise_vns = list(noise_vns[np.where(noise_feat_lens < 50)])
long_noise_vns = list(noise_vns[np.where(noise_feat_lens >= 50)])

augmented_data = combine_to_augment(short_noise_vns, int(augment_num/2), noise_data)+split_to_augment(long_noise_vns, augment_num-int(augment_num/2), noise_data)

augmented_data_dict = {}
time_id = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
for i in range(augment_num):
    cur_vn = 'AUG_VID_NOISE_%s_%07d' % (time_id, i)
    augmented_data_dict[cur_vn] = augmented_data[i]

pickle.dump(augmented_data_dict, open(save_path, 'wb'))
























