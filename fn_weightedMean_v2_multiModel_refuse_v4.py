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


def save_gt(vns, preds, probs, path, class_num=4934):
    preds = [p+1 for p in preds]
    ids = list(set(preds))
    ids.sort()
    df = pd.DataFrame({'vn': vns, 'pred': preds, 'prob': probs})
    with open(path, 'w') as f:
        for i in ids:
            if i > class_num:
                continue
            cur_df = df[df.pred == i].sort_values(by='prob', ascending=False)
            cur_vns = cur_df['vn'].values[: min(len(cur_df), 100)]
            f.write(str(i)+' ')
            f.write(' '.join(cur_vns))
            f.write('\n')


def feat2X(feats, max_len, embd_size):
    X = np.zeros([len(feats), max_len, embd_size], dtype=np.float32)
    for i in range(len(feats)):
        f = feats[i]
        X[i, :len(f), :] = f
    return X


def run_val(val_X, val_vns, val_gt_path, val_pred_path, batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, pred, embd_size, val=True):
    batch_num = int(len(val_X)/batch_size)
    left = int(len(val_X)%batch_size)
    pred_list = []
    prob_list = []
    batch_y = np.zeros([batch_size], dtype=np.int64)
    batch_w = np.ones([batch_size], dtype=np.float32)
    for i in range(batch_num):
        batch_X = np.array(val_X[i*batch_size: (i+1)*batch_size])
        cur_prob, cur_pred = sess.run([prob, pred], feed_dict={input_ph: batch_X, label_ph:batch_y, loss_weight_ph: batch_w})
        cur_pred = np.reshape(cur_pred, [-1])
        pred_list += list(cur_pred)
        prob_list += list(cur_prob)
    if left > 0:
        batch_X = np.zeros([batch_size, embd_size], dtype=np.float32)
        batch_X[:left] = np.array(val_X[-left:])
        cur_prob, cur_pred = sess.run([prob, pred], feed_dict={input_ph: batch_X, label_ph:batch_y, loss_weight_ph: batch_w})
        cur_pred = np.reshape(cur_pred, [-1])
        pred_list += list(cur_pred)[:left]
        prob_list += list(cur_prob)[:left]
    if val:
        save_pred(val_vns, prob_list, val_pred_path)
        return calculate_map(val_gt_path, val_pred_path)
    else:
        return pred_list


def run_test(test_X, test_vns, result_path, batch_size, sess, input_ph, label_ph, prob, embd_size):
    batch_num = int(len(test_X)/batch_size)
    left = int(len(test_X)%batch_size)
    prob_list = []
    batch_y = np.zeros([batch_size], dtype=np.int64)
    for i in range(batch_num):
        batch_X = np.array(test_X[i*batch_size: (i+1)*batch_size])
        cur_prob = sess.run(prob, feed_dict={input_ph: batch_X, label_ph:batch_y})
        prob_list += list(cur_prob)
    if left > 0:
        batch_X = np.zeros([batch_size, embd_size], dtype=np.float32)
        batch_X[:left] = np.array(test_X[-left:])
        cur_prob = sess.run(prob, feed_dict={input_ph: batch_X, label_ph:batch_y})
        cur_prob = np.max(cur_prob, axis=1)
        prob_list += list(cur_prob)[:left]
    pd.DataFrame({'video_name': test_vns, 'prob': prob_list}).to_csv(result_path, index=False)


def train(train_X_labeled, train_X_noise, train_y_labeled, train_y_noise, val_X, val_vns, val_gt_path, hidden_size, embd_size, class_num, noise_num, epoch_num, batch_size, learning_rate, lr_decay, weight_decay, noise_weight_list, model_dir, val_dir, pretrained_model=''):
    lr = learning_rate
    base_num = class_num+noise_num
    log_dir = os.path.join(model_dir, 'log')
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)
    log_path = os.path.join(log_dir, 'log.log')
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 512], name='input')
    label_ph = tf.placeholder(dtype=tf.int64, shape=[None], name='label')
    loss_weight_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='loss_weight')
    embds = inference(input_ph, hidden_size, embd_size)
    logits, embd_norm, w_norm = arcface_loss(embedding=embds, labels=label_ph, w_init=tf.contrib.layers.xavier_initializer(uniform=False), out_num=2*class_num+noise_num)
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_ph)*loss_weight_ph)
    var_to_decay = [v for v in tf.trainable_variables() if 'weight' in v.name]
    wd_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(weight_decay), var_to_decay)
    prob = tf.nn.softmax(logits)
    pred = tf.argmax(prob, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, label_ph), dtype=tf.float32))
    zero_pred = tf.reduce_mean(tf.cast(tf.greater(pred, class_num-1), dtype=tf.float32))
    noBase_pred = tf.reduce_mean(tf.cast(tf.greater(pred, base_num-1), dtype=tf.float32))
    total_loss = inference_loss+wd_loss
    train_op = slim.train.AdamOptimizer(lr).minimize(total_loss)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    w_update_freq = int((epoch_num-1)/len(noise_weight_list))+1
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(max_to_keep=5)
        if pretrained_model != '':
            saver.restore(sess, pretrained_model)
            noise_pred = run_val(train_X_noise, val_vns, val_gt_path, '', batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, pred, 512, val=False)
            for i in range(len(noise_pred)):
                if noise_pred[i] < class_num:
                    noise_pred[i] += base_num
            train_y_noise = noise_pred
        best_map = 0
        best_model = ''
        best_epoch = 0
        start_time = time.time()
        step = 0
        for i in range(epoch_num):
            # idxes = np.arange(0, len(train_X))
            # np.random.shuffle(idxes)
            # train_X = train_X[idxes]
            # train_y = train_y[idxes]
            # train_seq_len = train_seq_len[idxes]
            train_X = train_X_labeled+train_X_noise
            train_y = train_y_labeled+train_y_noise
            tmp = list(zip(train_X, train_y))
            random.shuffle(tmp)
            train_X[:], train_y[:] = zip(*tmp)
            epoch_size = int(len(train_X)/batch_size)
            noise_weight = noise_weight_list[int(i/w_update_freq)]
            for j in range(epoch_size):
                batch_X = np.array(train_X[j*batch_size: (j+1)*batch_size])
                batch_y = np.array(train_y[j*batch_size: (j+1)*batch_size], dtype=np.int64)
                batch_loss_weight = np.ones_like(batch_y, dtype=np.float32)
                batch_loss_weight[np.where(batch_y >= base_num)] = noise_weight
                n_e, n_w, _, l, a, z_p, nb_p = sess.run([embd_norm, w_norm, train_op, total_loss, acc, zero_pred, noBase_pred], feed_dict={input_ph: batch_X, label_ph: batch_y, loss_weight_ph: batch_loss_weight})
                print('[%d/%d][%d/%d]-loss: %.4f-acc: %.4f (zero pred: %.4f, noBase pred: %.4f)-time: %.3f' % (i, epoch_num, j, epoch_size, l, a, z_p, nb_p, (time.time()-start_time)))
                if l != l:
                    print('any zero: [embd %s][weight: %s]' % (str(np.any(n_e == 0.0)), str(np.any(n_w == 0.0))))
                step += 1
                start_time = time.time()
            val_pred_path = os.path.join(val_dir, 'epoch-%d.txt' % (i+1))
            cur_map = run_val(val_X, val_vns, val_gt_path, val_pred_path, batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, pred, 512)
            with open(log_path, 'a') as f:
                f.write('epoch: %d(noise weight: %f), val_mAP: %f\n' % ((i+1), noise_weight, cur_map))
            if cur_map > best_map:
                best_map = cur_map
                best_epoch = i+1
                best_model = os.path.join(model_dir, 'm-%d' % (i+1))+'-'+str(step)
                saver.save(sess, os.path.join(model_dir, 'm-%d' % (i+1)), global_step=step)
            noise_pred = run_val(train_X_noise, val_vns, val_gt_path, val_pred_path, batch_size, sess, input_ph, label_ph, loss_weight_ph, prob, pred, 512, val=False)
            for i in range(len(noise_pred)):
                if noise_pred[i] < class_num:
                    noise_pred[i] += base_num
            train_y_noise = noise_pred
        with open(log_path, 'a') as f:
            f.write('best epoch: %d, best mAP: %f\n' % (best_epoch, best_map))
        return best_map, best_model


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


def load_data(feat_path, qua_thred=0.0, det_thred=0.0, label_path=''):
    feat_dict = pickle.load(open(feat_path, 'rb'), encoding='bytes')
    if label_path != '':
        label_dict = load_labels(label_path)
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
        if label_path != '':
            labels.append(label_dict[vn]-1)
    return vns, feats, labels


def load_noise(feat_path, label_path, qua_thred=0.0, det_thred=0.0, class_num=4934, noise_num=0):
    base_class = class_num+noise_num
    feat_dict = pickle.load(open(feat_path, 'rb'), encoding='bytes')
    label_dict = load_labels(label_path)
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
        if label_dict[vn] >= 0:
            labels.append(class_num+label_dict[vn])
        else:
            labels.append(random.randint(base_class, base_class+class_num-1))
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
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.0)
    parser.add_argument('--folds', type=int, help='folds number', default=10)
    parser.add_argument('--det_thred', type=float, help='det thred', default=0.0)
    parser.add_argument('--qua_thred', type=float, help='qua thred', default=0.0)
    parser.add_argument('--pretrained_subdir', type=str, help='path of pretrained models', default='')
    parser.add_argument('--embd_size', type=int, help='embd size', default=512)
    parser.add_argument('--batch_size', type=int, help='embd size', default=10240)

    return parser.parse_args(argv)



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay
    noise_weight = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    folds = args.folds
    det_thred = args.det_thred
    qua_thred = args.qua_thred
    pretrained_subdir = args.pretrained_subdir
    embd_size = args.embd_size
    batch_size = args.batch_size

    hidden_size = []
    class_num = 4934
    noise_num = 3691
    epoch_num = 200

    model_dir = '/data/hhd/iqiyi/model/fn_weightedMean_v2_multiModel_refuse_v4'
    val_dir = '/data/hhd/iqiyi/val_pred/fn_weightedMean_v2_multiModel_refuse_v4'
    train_feat_path = '/data/hhd/iqiyi/feats_train_v2.pickle'
    train_label_path = '/data/hhd/iqiyi/gt_v2/train_v2.txt'
    val_labeled_feat_path = '/data/hhd/iqiyi/feats_val_labeled.pickle'
    val_label_path = '/data/hhd/iqiyi/gt_v2/val_label.txt'
    # val_noise_path = '/data/hhd/iqiyi/feats_val_noise.pickle'
    train_aug_path = '/data/hhd/iqiyi/feats_noise_aug_train.pickle'
    val_aug_path = '/data/hhd/iqiyi/feats_noise_aug_val.pickle'
    train_noise_path = '/data/hhd/iqiyi/feats_noise_train_split.pickle'
    val_noise_path = '/data/hhd/iqiyi/feats_noise_val_split.pickle'
    noise_label_path = '/data/hhd/iqiyi/noise_labels_1.2.txt'

    print('data prepare...')
    start_time = time.time()
    train_vns, train_feats, train_labels = load_data(train_feat_path, qua_thred=qua_thred, det_thred=det_thred, label_path=train_label_path)
    val_labeled_vns, val_labeled_feats, val_labels = load_data(val_labeled_feat_path, qua_thred=qua_thred, det_thred=det_thred, label_path=val_label_path)
    train_aug_vns, train_aug_feats, train_aug_labels = load_noise(train_aug_path, noise_label_path, qua_thred=qua_thred, det_thred=det_thred, class_num=class_num, noise_num=noise_num)
    train_noise_vns, train_noise_feats, train_noise_labels = load_noise(train_noise_path, noise_label_path, qua_thred=qua_thred, det_thred=det_thred, class_num=class_num, noise_num=noise_num)
    val_aug_vns, val_aug_feats, val_aug_labels = load_noise(val_aug_path, noise_label_path, qua_thred=qua_thred, det_thred=det_thred, class_num=class_num, noise_num=noise_num)
    val_noise_vns, val_noise_feats, val_noise_labels = load_noise(val_noise_path, noise_label_path, qua_thred=qua_thred, det_thred=det_thred, class_num=class_num, noise_num=noise_num)
    data_vns = train_vns + val_labeled_vns
    del train_vns
    del val_labeled_vns
    data_X = train_feats+val_labeled_feats
    del train_feats
    del val_labeled_feats
    data_y = train_labels+val_labels
    del train_labels
    del val_labels
    noise_vns = train_aug_vns+train_noise_vns+val_aug_vns+val_noise_vns
    noise_X = train_aug_feats+train_noise_feats+val_aug_feats+val_noise_feats
    noise_y = train_aug_labels+train_noise_labels+val_aug_labels+val_noise_labels
    print('========================================')
    print('data vns: %d' % len(data_vns))
    print('data X: %d' % len(data_X))
    print('data y: %d' % len(data_y))
    print('noise vns: %d' % len(noise_vns))
    print('noise X: %d' % len(noise_X))
    print('noise y: %d' % len(noise_y))
    print('========================================')
    del train_aug_feats
    del train_aug_labels
    del train_aug_vns
    del train_noise_feats
    del train_noise_labels
    del train_noise_vns
    del val_aug_feats
    del val_aug_labels
    del val_aug_vns
    del val_noise_feats
    del val_noise_labels
    del val_noise_vns
    # tmp = list(zip(data_vns, data_X, data_y))
    # random.shuffle(tmp)
    # data_vns[:], data_X[:], data_y[:]= zip(*tmp)
    # tmp = list(zip(noise_vns, noise_X, noise_y))
    # random.shuffle(tmp)
    # noise_vns[:], noise_X[:], noise_y[:]= zip(*tmp)
    data_vns = np.array(data_vns)
    data_X = np.array(data_X)
    data_y = np.array(data_y)
    noise_vns = np.array(noise_vns)
    noise_X = np.array(noise_X)
    noise_y = np.array(noise_y)
    print('done![time: %.3f]' % (time.time()-start_time))

    if pretrained_subdir != '':
        pre_model_dir = os.path.join(model_dir, pretrained_subdir)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_dir = os.path.join(model_dir, subdir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    val_dir = os.path.join(val_dir, subdir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    if pretrained_subdir != '':
        data_splits = pickle.load(open(os.path.join(pre_model_dir, 'data_splits.pkl'), 'rb'))
        noise_splits = pickle.load(open(os.path.join(pre_model_dir, 'noise_splits.pkl'), 'rb'))
        pretrained_models = pickle.load(open(os.path.join(pre_model_dir, 'best_models.pkl'), 'rb'))
        folds = len(pretrained_models)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        data_splits = list(skf.split(data_X, data_y))
        noise_splits = list(skf.split(noise_X, noise_y))
        pretrained_models = ['']*folds
    
    log_path = os.path.join(model_dir, 'log.log')
    with open(log_path, 'a') as f:
        f.write('===============================================================================\n')
        f.write('folds: '+str(folds)+'\n')
        f.write('qua_thred: '+str(qua_thred)+'\n')
        f.write('det_thred: '+str(det_thred)+'\n')
        f.write('hidden size: '+str(hidden_size)+'\n')
        f.write('embd size: '+str(embd_size)+'\n')
        f.write('batch size: '+str(batch_size)+'\n')
        f.write('epoch num: '+str(epoch_num)+'\n')
        f.write('learning rate: '+str(learning_rate)+'\n')
        f.write('lr decay: '+str(lr_decay)+'\n')
        f.write('weight decay: '+str(weight_decay)+'\n')
        f.write('noise_weight: '+str(noise_weight)+'\n')
        f.write('pretrained subdir: '+str(pretrained_subdir)+'\n')
        f.write('===============================================================================\n')

    pickle.dump(data_splits, open(os.path.join(model_dir, 'data_splits.pkl'), 'wb'))
    pickle.dump(noise_splits, open(os.path.join(model_dir, 'noise_splits.pkl'), 'wb'))

    best_maps = []
    best_models = []
    for i in range(folds):
        pretrained_model = pretrained_models[i]
        train_idxes, val_idxes = data_splits[i]
        train_idxes_n, val_idxes_n = noise_splits[i]
        print('fold %d...' % i)
        train_feats = list(data_X[train_idxes])
        train_labels = list(data_y[train_idxes])
        val_feats = list(data_X[val_idxes])
        val_labels = list(data_y[val_idxes])
        val_vns = list(data_vns[val_idxes])
        train_feats_noise = list(noise_X[train_idxes])
        train_labels_noise = list(noise_y[train_idxes])
        val_feats_noise = list(noise_X[val_idxes])
        val_labels_noise = list(noise_y[val_idxes])
        val_vns_noise = list(noise_vns[val_idxes])
        cur_model_dir = os.path.join(model_dir, 'fold_'+str(i))
        if not os.path.exists(cur_model_dir):
            os.makedirs(cur_model_dir)
        cur_val_dir = os.path.join(val_dir, 'fold_'+str(i))
        if not os.path.exists(cur_val_dir):
            os.makedirs(cur_val_dir)
        val_gt_path = os.path.join(cur_val_dir, 'val_gt_fold_%d.txt' % i)
        save_gt(val_vns, val_labels, probs=list(np.ones([len(val_vns)])), path=val_gt_path, class_num=class_num)
        val_vns = val_vns + val_vns_noise
        val_feats = val_feats + val_feats_noise
        print('train...')
        best_map, best_model = train(train_feats, train_feats_noise, train_labels, train_labels_noise, val_feats, val_vns, val_gt_path, hidden_size, embd_size, class_num, noise_num, epoch_num, batch_size, learning_rate, lr_decay, weight_decay, noise_weight, cur_model_dir, cur_val_dir, pretrained_model=pretrained_model)
        tf.reset_default_graph()
        best_maps.append(best_map)
        best_models.append(best_model)
        with open(log_path, 'a') as f:
            f.write('best mAP of fold %d: %f\n' % (i, best_map))
    pickle.dump(best_maps, open(os.path.join(model_dir, 'best_maps.pkl'), 'wb'))
    pickle.dump(best_models, open(os.path.join(model_dir, 'best_models.pkl'), 'wb'))
    with open(log_path, 'a') as f:
        f.write('best mAP mean: %f\n' % (np.mean(np.array(best_maps))))
    print('done!')



    



