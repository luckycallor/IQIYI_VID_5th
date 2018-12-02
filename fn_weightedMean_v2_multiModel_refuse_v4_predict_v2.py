import os
import sys
import time
import pickle
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from datetime import datetime
from sklearn.model_selection import StratifiedKFold

from ArcFace_losses import arcface_loss
from evaluation_map_v2 import calculate_map


def inference(inputs, hidden_size, embd_size):
    net = inputs
    if hidden_size == None:
        return inputs
    for i in range(len(hidden_size)):
        net = slim.fully_connected(inputs, hidden_size[i], activation_fn=tf.nn.relu, scope='fn_'+str(i))
    embd = slim.fully_connected(net, embd_size, activation_fn=None, scope='fn_embd')
    return embd


def save_pred(vns, probs, path, class_num=4934):
    ids = list(range(class_num))
    vns = np.array(vns)
    probs = np.array(probs)
    with open(path, 'w') as f:
        for i in ids:
            cur_arr = probs[:,i]
            idx_100 = np.argpartition(cur_arr, -100)[-100:]
            cur_vns = vns[idx_100]
            cur_arr = cur_arr[idx_100]
            idx_sort = np.argsort(-cur_arr)
            cur_vns = cur_vns[idx_sort]
            f.write(str(i+1)+' ')
            f.write(' '.join(cur_vns))
            f.write('\n')


# def save_pred(vns, preds, probs, path, class_num=4934):
#     preds = [p+1 for p in preds]
#     ids = list(set(preds))
#     ids.sort()
#     df = pd.DataFrame({'vn': vns, 'pred': preds, 'prob': probs})
#     with open(path, 'w') as f:
#         for i in ids:
#             if i > class_num:
#                 continue
#             cur_df = df[df.pred == i].sort_values(by='prob', ascending=False)
#             cur_vns = cur_df['vn'].values[: min(len(cur_df), 100)]
#             f.write(str(i)+' ')
#             f.write(' '.join(cur_vns))
#             f.write('\n')


def feat2X(feats, max_len, embd_size):
    X = np.zeros([len(feats), max_len, embd_size], dtype=np.float32)
    for i in range(len(feats)):
        f = feats[i]
        X[i, :len(f), :] = f
    return X


# def run_val(val_X, val_vns, val_gt_path, val_pred_path, batch_size, sess, input_ph, label_ph, prob, pred, embd_size):
#     batch_num = int(len(val_X)/batch_size)
#     left = int(len(val_X)%batch_size)
#     pred_list = []
#     prob_list = []
#     batch_y = np.zeros([batch_size], dtype=np.int64)
#     for i in range(batch_num):
#         batch_X = np.array(val_X[i*batch_size: (i+1)*batch_size])
#         cur_prob, cur_pred = sess.run([prob, pred], feed_dict={input_ph: batch_X, label_ph:batch_y})
#         cur_prob = np.max(cur_prob, axis=1)
#         cur_pred = np.reshape(cur_pred, [-1])
#         pred_list += list(cur_pred)
#         prob_list += list(cur_prob)
#     if left > 0:
#         batch_X = np.zeros([batch_size, embd_size], dtype=np.float32)
#         batch_X[:left] = np.array(val_X[-left:])
#         cur_prob, cur_pred = sess.run([prob, pred], feed_dict={input_ph: batch_X, label_ph:batch_y})
#         cur_prob = np.max(cur_prob, axis=1)
#         cur_pred = np.reshape(cur_pred, [-1])
#         pred_list += list(cur_pred)[:left]
#         prob_list += list(cur_prob)[:left]
#     save_pred(val_vns, pred_list, prob_list, val_pred_path)
#     return calculate_map(val_gt_path, val_pred_path)


def run_test(test_X, test_vns, batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, embd_size):
    print('===================================================')
    batch_num = int(len(test_X)/batch_size)
    left = int(len(test_X)%batch_size)
    prob_list = []
    batch_y = np.zeros([batch_size], dtype=np.int64)
    batch_w = np.ones([batch_size], dtype=np.float32)
    for i in range(batch_num):
        batch_X = np.array(test_X[i*batch_size: (i+1)*batch_size])
        cur_prob = sess.run(prob, feed_dict={input_ph: batch_X, label_ph:batch_y, loss_weight_ph: batch_w})
        prob_list += list(cur_prob)
    if left > 0:
        batch_X = np.zeros([batch_size, embd_size], dtype=np.float32)
        batch_X[:left] = np.array(test_X[-left:])
        cur_prob = sess.run(prob, feed_dict={input_ph: batch_X, label_ph:batch_y, loss_weight_ph: batch_w})
        print(type(cur_prob))
        print(cur_prob.shape)
        prob_list += list(cur_prob)[:left]
    print(len(prob_list))
    print(type(prob_list[0]))
    print(len(prob_list[0]))
    print('===================================================')
    return pd.DataFrame({'video_name': test_vns, 'prob': prob_list})


def test(test_X, test_vns, hidden_size, embd_size, class_num, noise_num, epoch_num, batch_size, learning_rate, lr_decay, pretrained_model=''):
    lr = learning_rate
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 512], name='input')
    label_ph = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
    loss_weight_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='loss_weight')
    embds = inference(input_ph, hidden_size, embd_size)
    logits, embd_norm, w_norm = arcface_loss(embedding=embds, labels=label_ph, w_init=tf.contrib.layers.xavier_initializer(uniform=False), out_num=2*class_num+noise_num)
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)*loss_weight_ph)
    prob = tf.nn.softmax(logits)
    pred = tf.argmax(prob, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, label_ph), dtype=tf.float32))
    zero_pred = tf.reduce_mean(tf.cast(tf.greater(pred, class_num-1), dtype=tf.float32))
    total_loss = inference_loss
    train_op = slim.train.AdamOptimizer(lr).minimize(total_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=200)
        saver.restore(sess, pretrained_model)
        return run_test(test_X, test_vns, batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, 512)


def l2_normalize(feat):
    return feat/np.sqrt(np.sum(np.square(feat)))


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


def load_labels(fn):
    label_dict = {}
    with open(fn) as f:
        for l in f.readlines():
            l = l.strip().split(' ')
            label_dict[l[0]] = int(l[1])
    return label_dict


def frame_sorted_feat(fs):
    pass


def load_data(feat_path, folds=3, label_path=''):
    feat_dict = pickle.load(open(feat_path, 'rb'), encoding='bytes')
    if label_path != '':
        label_dict = load_labels(label_path)
    vns = []
    feats_1 = []
    feats_2 = []
    feats_3 = []
    feats_4 = []
    labels = []
    for vn in feat_dict.keys():
        fs = feat_dict[vn]
        if(len(fs) == 0):
            continue
        fold_idx = int(len(fs)/folds)
        fold_idx_2 = int(len(fs)/folds/2)
        feats_1.append(feat_weighted_mean(fs))
        if fold_idx > 0:
            feats_2.append(feat_weighted_mean(fs[fold_idx:]))
            feats_3.append(feat_weighted_mean(fs[:-fold_idx]))
        elif(len(fs) > 1):
            feats_2.append(feat_weighted_mean(fs[1:]))
            feats_3.append(feat_weighted_mean(fs[:-1]))
        else:
            feats_2.append(feat_weighted_mean(fs[0:]))
            feats_3.append(feat_weighted_mean(fs[:]))
        if fold_idx_2>0:
            feats_4.append(feat_weighted_mean(fs[fold_idx_2:-fold_idx_2]))
        elif len(fs) > 2:
            feats_4.append(feat_weighted_mean(fs[1:-1]))
        else:
            feats_4.append(feat_weighted_mean(fs[0:]))
        vns.append(vn)
        if label_path != '':
            labels.append(label_dict[vn]-1)
    feats = [feats_1, feats_2, feats_3, feats_4]
    return vns, feats, labels


# def feat2X(feats, max_len, embd_size):
#     X = np.zeros([len(feats), max_len, embd_size], dtype=np.float32)
#     seq_lens = []
#     for i in range(len(feats)):
#         f = feats[i]
#         X[i, -len(f):, :] = f
#         seq_lens.append(len(f))
#     return X, np.array(seq_lens)


def parse_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, help='learning rate', default=0.001)
    parser.add_argument('--lr_decay', type=float, help='learning rate decay', default=0.8)
    parser.add_argument('--folds', type=int, help='folds number', default=3)
    parser.add_argument('--pretrained_subdir', type=str, help='path of pretrained model', default='20181011-145103')
    parser.add_argument('--embd_size', type=int, help='embd size', default=512)

    return parser.parse_args(argv)



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    folds = args.folds
    subdir = args.pretrained_subdir
    embd_size = args.embd_size

    hidden_size = []
    class_num = 4934
    noise_num = 3691
    epoch_num = 100
    batch_size = 10240

    model_dir = '/data/hhd/iqiyi/model/fn_weightedMean_v2_multiModel_refuse_v4'
    val_dir = '/data/hhd/iqiyi/val_pred/fn_weightedMean_v2_multiModel_refuse_v4'
    result_dir = '/data/hhd/iqiyi/result/fn_weightedMean_v2_multiModel_refuse_v4_predict_v2'
    train_feat_path = '/data/hhd/iqiyi/feats_train_v2.pickle'
    train_label_path = '/data/hhd/iqiyi/gt_v2/train_v2.txt'
    val_labeled_feat_path = '/data/hhd/iqiyi/feats_val_labeled.pickle'
    val_label_path = '/data/hhd/iqiyi/gt_v2/val_label.txt'
    val_noise_path = '/data/hhd/iqiyi/feats_val_noise.pickle'
    test_feat_path = '/data/hhd/iqiyi/feats_test.pickle'

    print('data prepare...')
    start_time = time.time()
    test_vns, test_feats, _ = load_data(test_feat_path)
    print('done![time: %.3f]' % (time.time()-start_time))

    
    model_dir = os.path.join(model_dir, subdir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    result_dir = os.path.join(result_dir, subdir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    pretrained_models = pickle.load(open(os.path.join(model_dir, 'best_models.pkl'), 'rb'))
    best_maps = pickle.load(open(os.path.join(model_dir, 'best_maps.pkl'), 'rb'))
    assert len(pretrained_models) == len(best_maps)
    folds = len(pretrained_models)

    probs_df = pd.DataFrame({'video_name': test_vns})
    probs_arr = np.zeros([len(test_vns), 2*class_num+noise_num])
    weighted_probs_arr = np.zeros([len(test_vns), 2*class_num+noise_num])
    for i in range(folds):
        pretrained_model = pretrained_models[i]
        cur_prob = np.zeros([len(test_vns), 2*class_num+noise_num])
        for j in range(4):
            tmp_prob = test(test_feats[j], test_vns, hidden_size, embd_size, class_num, noise_num, epoch_num, batch_size, learning_rate, lr_decay, pretrained_model=pretrained_model)
            tf.reset_default_graph()
            probs_df = probs_df.merge(tmp_prob)
            tmp_prob = np.array(list(probs_df['prob'].values))
            cur_prob = np.maximum(cur_prob, tmp_prob)
            probs_df = probs_df.drop(['prob'], axis=1)
        probs_arr += cur_prob
        weighted_probs_arr += best_maps[i]*cur_prob
    # pred = list(np.argmax(probs_arr, axis=1))
    # prob = list(np.max(probs_arr, axis=1))
    save_pred(test_vns, probs_arr, os.path.join(result_dir, 'result_ensembleBySum.txt'))
    # pred = list(np.argmax(weighted_probs_arr, axis=1))
    # prob = list(np.max(weighted_probs_arr, axis=1))
    save_pred(test_vns, weighted_probs_arr, os.path.join(result_dir, 'result_ensembleByWeightedSum.txt'))




