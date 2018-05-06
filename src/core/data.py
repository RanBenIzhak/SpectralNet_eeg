'''
data.py: contains all data generating code for datasets used in the script
'''

import os, sys
import h5py

import numpy as np
from sklearn import preprocessing

from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json

from core import pairs
from core.takens_embed import get_delayed_manifold as takens


def get_data(params, data=None):
    '''
    Convenience function: preprocesses all data in the manner specified in params, and returns it
    as a nested dict with the following keys:
    the permutations (if any) used to shuffle the training and validation sets
    'p_train'                           - p_train
    'p_val'                             - p_val
    the data used for spectral net
    'spectral'
        'train_and_test'                - (x_train, y_train, x_val, y_val, x_test, y_test)
        'train_unlabeled_and_labeled'   - (x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled)
        'val_unlabeled_and_labeled'     - (x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled)
    the data used for siamese net, if the architecture uses the siamese net
    'siamese'
        'train_and_test'                - (pairs_train, dist_train, pairs_val, dist_val)
        'train_unlabeled_and_labeled'   - (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        'val_unlabeled_and_labeled'     - (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)
    '''
    ret = {}

    # get data if not provided
    if data is None:
        x_train, x_test, y_train, y_test = load_data(params)
    else:
        print("WARNING: Using data provided in arguments. Must be tuple or array of format (x_train, x_test, y_train, y_test)")
        x_train, x_test, y_train, y_test = data

    ret['spectral'] = {}
    if params.get('use_all_data'):
        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = np.concatenate((y_train, y_test), axis=0)
        x_test = np.zeros((0,) + x_train.shape[1:])
        y_test = np.zeros((0,))

    # split x training, validation, and test subsets
    if 'val_set_fraction' not in params:
        print("NOTE: Validation set required, setting val_set_fraction to 0.1")
        train_val_split = (.9, .1)
    elif params['val_set_fraction'] > 0 and params['val_set_fraction'] <= 1:
        train_val_split = (1 - params['val_set_fraction'], params['val_set_fraction'])
    else:
        raise ValueError("val_set_fraction is invalid! must be in range (0, 1]")

    # shuffle training and test data separately into themselves and concatenate
    if 'bci' in params['dset']:
        (x_train, y_train, p_train), (x_val, y_val, p_val) = split_data(x_train, y_train, train_val_split)
    else:
        p = np.concatenate([np.random.permutation(len(x_train)), len(x_train) + np.random.permutation(len(x_test))], axis=0)
        (x_train, y_train, p_train), (x_val, y_val, p_val) = split_data(x_train, y_train, train_val_split, permute=p[:len(x_train)])

    # further split each training and validation subset into its supervised and unsupervised sub-subsets
    if params.get('train_labeled_fraction'):
        train_split = (1 - params['train_labeled_fraction'], params['train_labeled_fraction'])
    else:
        train_split = (1, 0)
    (x_train_unlabeled, y_train_unlabeled, p_train_unlabeled), (x_train_labeled, y_train_labeled, _) = split_data(x_train, y_train, train_split)

    if params.get('val_labeled_fraction'):
        val_split = (1 - params['val_labeled_fraction'], params['val_labeled_fraction'])
    else:
        val_split = (1, 0)
    (x_val_unlabeled, y_val_unlabeled, p_val_unlabeled), (x_val_labeled, y_val_labeled, _) = split_data(x_val, y_val, val_split)


    # embed data in code space, if necessary
    if params.get('use_code_space'):
        all_data = [x_train, x_val, x_test, x_train_unlabeled, x_train_labeled, x_val_unlabeled, x_val_labeled]
        for i, d in enumerate(all_data):
            all_data[i] = embed_data(d, dset=params['dset'])
        x_train, x_val, x_test, x_train_unlabeled, x_train_labeled, x_val_unlabeled, x_val_labeled = all_data

    # collect everything into a dictionary
    ret['spectral']['train_and_test'] = (x_train, y_train, x_val, y_val, x_test, y_test)
    ret['spectral']['train_unlabeled_and_labeled'] = (x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled)
    ret['spectral']['val_unlabeled_and_labeled'] = (x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled)

    ret['p_train'] = p_train
    ret['p_val'] = p_val

    # get siamese data if necessary
    if 'siamese' in params['affinity']:
        ret['siamese'] = {}

        if params.get('precomputedKNNPath'):
            # if we use precomputed knn, we cannot shuffle the data; instead
            # we pass the permuted index array and the full matrix so that
            # create_pairs_from_unlabeled data can keep track of the indices
            p_train_unlabeled = p_train[:len(x_train_unlabeled)]
            train_path = params.get('precomputedKNNPath', '')
            if train_val_split[1] < 0.09 or params['siam_k'] > 100:
                # if the validation set is very small, the benefit of
                # the precomputation is small, and there will be a high miss
                # rate in the precomputed neighbors (neighbors that are not
                # in the validation set) so we just recomputed neighbors
                p_val_unlabeled = None
                val_path = ''
            else:
                p_val_unlabeled = p_val[:len(x_val_unlabeled)]
                val_path = params.get('precomputedKNNPath', '')
        else:
            # if we do not use precomputed knn, then this does not matter
            p_train_unlabeled = None
            train_path = params.get('precomputedKNNPath', '')
            p_val_unlabeled = None
            val_path = params.get('precomputedKNNPath', '')

        pairs_train_unlabeled, dist_train_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=x_train_unlabeled,
            p=p_train_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=train_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )
        pairs_val_unlabeled, dist_val_unlabeled = pairs.create_pairs_from_unlabeled_data(
            x1=x_val_unlabeled,
            p=p_val_unlabeled,
            k=params.get('siam_k'),
            tot_pairs=params.get('siamese_tot_pairs'),
            precomputed_knn_path=val_path,
            use_approx=params.get('use_approx', False),
            pre_shuffled=True,
        )

        #get pairs for labeled data
        class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
        pairs_train_labeled, dist_train_labeled = pairs.create_pairs_from_labeled_data(x_train_labeled, class_indices)
        class_indices = [np.where(y_train_labeled == i)[0] for i in range(params['n_clusters'])]
        pairs_val_labeled, dist_val_labeled = pairs.create_pairs_from_labeled_data(x_train_labeled, class_indices)

        ret['siamese']['train_unlabeled_and_labeled'] = (pairs_train_unlabeled, dist_train_unlabeled, pairs_train_labeled, dist_train_labeled)
        ret['siamese']['val_unlabeled_and_labeled'] = (pairs_val_unlabeled, dist_val_unlabeled, pairs_val_labeled, dist_val_labeled)

        #combine labeled and unlabeled pairs for training the siamese
        pairs_train = np.concatenate((pairs_train_unlabeled, pairs_train_labeled), axis=0)
        dist_train = np.concatenate((dist_train_unlabeled, dist_train_labeled), axis=0)
        pairs_val = np.concatenate((pairs_val_unlabeled, pairs_val_labeled), axis=0)
        dist_val = np.concatenate((dist_val_unlabeled, dist_val_labeled), axis=0)

        ret['siamese']['train_and_test'] = (pairs_train, dist_train, pairs_val, dist_val)

    return ret

def load_data(params):
    '''
    Convenience function: reads from disk, downloads, or generates the data specified in params
    '''
    if params['dset'] == 'reuters':
        with h5py.File('../../data/reuters/reutersidf_total.h5', 'r') as f:
            x = np.asarray(f.get('data'), dtype='float32')
            y = np.asarray(f.get('labels'), dtype='float32')

            n_train = int(0.9 * len(x))
            x_train, x_test = x[:n_train], x[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]
    elif params['dset'] == 'mnist':
        x_train, x_test, y_train, y_test = get_mnist()
    elif params['dset'] == 'cc':
        x_train, x_test, y_train, y_test = generate_cc(params.get('n'), params.get('noise_sig'), params.get('train_set_fraction'))
        x_train, x_test = pre_process(x_train, x_test, params.get('standardize'))
    else:
        raise ValueError('Dataset provided ({}) is invalid!'.format(params['dset']))

    return x_train, x_test, y_train, y_test

def embed_data(x, dset):
    '''
    Convenience function: embeds x into the code space using the corresponding
    autoencoder (specified by dset).
    '''
    if not len(x):
        return np.zeros(shape=(0, 10))
    if dset == 'reuters':
        dset = 'reuters10k'

    json_path = '../pretrain_weights/ae_{}.json'.format(dset)
    weights_path = '../pretrain_weights/ae_{}_weights.h5'.format(dset)

    with open(json_path) as f:
        pt_ae = model_from_json(f.read())
    pt_ae.load_weights(weights_path)

    x = x.reshape(-1, np.prod(x.shape[1:]))

    get_embeddings = K.function([pt_ae.input],
                                  [pt_ae.layers[3].output])

    get_reconstruction = K.function([pt_ae.layers[4].input],
                                  [pt_ae.output])
    x_embedded = predict_with_K_fn(get_embeddings, x)[0]
    x_recon = predict_with_K_fn(get_reconstruction, x_embedded)[0]
    reconstruction_mse = np.mean(np.square(x - x_recon))
    print("using pretrained embeddings; sanity check, total reconstruction error:", np.mean(reconstruction_mse))

    del pt_ae

    return x_embedded

def predict_with_K_fn(K_fn, x, bs=1000):
    '''
    Convenience function: evaluates x by K_fn(x), where K_fn is
    a Keras function, by batches of size 1000.
    '''
    if not isinstance(x, list):
        x = [x]
    num_outs = len(K_fn.outputs)
    y = [np.empty((len(x[0]), output_.get_shape()[1])) for output_ in K_fn.outputs]
    recon_means = []
    for i in range(int(x[0].shape[0]/bs + 1)):
        x_batch = []
        for x_ in x:
            x_batch.append(x_[i*bs:(i+1)*bs])
        temp = K_fn(x_batch)
        for j in range(num_outs):
            y[j][i*bs:(i+1)*bs] = temp[j]

    return y

def split_data(x, y, split, permute=None):
    '''
    Splits arrays x and y, of dimensionality n x d1 and n x d2, into
    k pairs of arrays (x1, y1), (x2, y2), ..., (xk, yk), where both
    arrays in the ith pair is of shape split[i-1]*n x (d1, d2)
    x, y:       two matrices of shape n x d1 and n x d2
    split:      a list of floats of length k (e.g. [a1, a2,..., ak])
                where a, b > 0, a, b < 1, and a + b == 1
    permute:    a list or array of length n that can be used to
                shuffle x and y identically before splitting it
    returns:    a tuple of tuples, where the outer tuple is of length k
                and each of the k inner tuples are of length 3, of
                the format (x_i, y_i, p_i) for the corresponding elements
                from x, y, and the permutation used to shuffle them
                (in the case permute == None, p_i would simply be
                range(split[0]+...+split[i-1], split[0]+...+split[i]),
                i.e. a list of consecutive numbers corresponding to the
                indices of x_i, y_i in x, y respectively)
    '''
    n = len(x)
    if permute is not None:
        if not isinstance(permute, np.ndarray):
            raise ValueError("Provided permute array should be an np.ndarray, not {}!".format(type(permute)))
        if len(permute.shape) != 1:
            raise ValueError("Provided permute array should be of dimension 1, not {}".format(len(permute.shape)))
        if len(permute) != len(x):
            raise ValueError("Provided permute should be the same length as x! (len(permute) = {}, len(x) = {}".format(len(permute), len(x)))
    else:
        permute = np.arange(len(x))

    if np.sum(split) != 1:
        raise ValueError("Split elements must sum to 1!")

    ret_x_y_p = []
    prev_idx = 0
    for s in split:
        idx = prev_idx + np.round(s * n).astype(np.int)
        p_ = permute[prev_idx:idx]
        x_ = x[p_]
        y_ = y[p_]
        prev_idx = idx
        ret_x_y_p.append((x_, y_, p_))

    return tuple(ret_x_y_p)

def generate_cc(n=1200, noise_sigma=0.1, train_set_fraction=1.):
    '''
    Generates and returns the nested 'C' example dataset (as seen in the leftmost
    graph in Fig. 1)
    '''
    pts_per_cluster = int(n / 2)
    r = 1

    # generate clusters
    theta1 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)
    theta2 = (np.random.uniform(0, 1, pts_per_cluster) * r * np.pi - np.pi / 2).reshape(pts_per_cluster, 1)

    cluster1 = np.concatenate((np.cos(theta1) * r, np.sin(theta1) * r), axis=1)
    cluster2 = np.concatenate((np.cos(theta2) * r, np.sin(theta2) * r), axis=1)

    # shift and reverse cluster 2
    cluster2[:, 0] = -cluster2[:, 0] + 0.5
    cluster2[:, 1] = -cluster2[:, 1] - 1

    # combine clusters
    x = np.concatenate((cluster1, cluster2), axis=0)

    # add noise to x
    x = x + np.random.randn(x.shape[0], 2) * noise_sigma

    # generate labels
    y = np.concatenate((np.zeros(shape=(pts_per_cluster, 1)), np.ones(shape=(pts_per_cluster, 1))), axis=0)

    # shuffle
    p = np.random.permutation(n)
    y = y[p]
    x = x[p]

    # make train and test splits
    n_train = int(n * train_set_fraction)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train].flatten(), y[n_train:].flatten()

    return x_train, x_test, y_train, y_test

def get_mnist():
    '''
    Returns the train and test splits of the MNIST digits dataset,
    where x_train and x_test are shaped into the tensorflow image data
    shape and normalized to fit in the range [0, 1]
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # reshape and standardize x arrays
    x_train = np.expand_dims(x_train, -1) / 255
    x_test = np.expand_dims(x_test, -1) / 255
    return x_train, x_test, y_train, y_test

def pre_process(x_train, x_test, standardize):
    '''
    Convenience function: uses the sklearn StandardScaler on x_train
    and x_test if standardize == True
    '''
    #if we are going to standardize
    if standardize:
        #standardize the train data set
        preprocessor = preprocessing.StandardScaler().fit(x_train)
        x_train = preprocessor.transform(x_train)
        #if we have test data
        if x_test.shape[0] > 0:
            #standardize the test data set
            preprocessor = preprocessing.StandardScaler().fit(x_test)
            x_test = preprocessor.transform(x_test)
    return x_train, x_test

def eeg_preprocess(data, method, metadata=None, params=None):
    x_train, x_test, y_train, y_test = data
    if method == 'takens':
        tau, ndelay = params['tau'], params['ndelay']
        data_processed = takens(x_train, tau=tau, ndelay=ndelay)
        data_processed = np.transpose(data_processed, (1, 0, 2))
        train_data_processed = data_processed.reshape((data_processed.shape[0], -1))
        data_processed = takens(x_test, tau=tau, ndelay=ndelay)
        data_processed = np.transpose(data_processed, (1, 0, 2))
        test_data_processed = data_processed.reshape((data_processed.shape[0], -1))
        train_y_processed = y_train[tau * ndelay:]
        test_y_processed = y_test[tau * ndelay:]
        examples = []
        return (train_data_processed, test_data_processed, train_y_processed, test_y_processed), examples

    if method =='welch':
        data_processed = []
        labels_processed = []
        examples = []
        examples_counter = 0
        # Need to split the data into chunks on which we will average. chose 4 seconds chunks
        train_welch_x, train_welch_y = get_welch(x_train, y_train, params['tmi'])
        test_welch_x, test_welch_y = get_welch(x_test, y_test, params['emi'])
        # for i in range(500, x_train.shape[0] - 1000, 250):
        #     cur_segment = x_train[i-500:i+500]
        #     fsp, cur_welch = welch(cur_segment, fs=250, axis=0)
        #     data_processed.append(cur_welch)
        #     labels_processed.append(y_train[i])

        return tuple([np.asarray(x) for x in [train_welch_x, test_welch_x, train_welch_y, test_welch_y]]),  examples

    if method == 'welch_diffrential':

        welch_array = []
        psd_dif = []
        labels_processed = []
        examples = []
        examples_counter = 0
        cur_label_pos = 0  # for finding labels

        # Calculating changes in Power Density function over time every 0.5 seconds, by averaging welch of the last 2 seconds every time.
        fs = 250   # fs of the data
        inner_time = 0.2  # [seconds]
        inner_samp = int(inner_time * fs)  # check every X samples the difference in averages
        welch_range = 250 * 4  # average over last 4 seconds
        memory_length = 10  # compare to last 2 seconds  - 10 * inner_time
        assert(data.shape[0] > 10000)
        points = range(welch_range, data.shape[0])[::inner_samp]
        for j, point in enumerate(points):
            # finding the correct label of the sample at time "point"
            while point > metadata[cur_label_pos][1]:  # update location of current label if needed
                if cur_label_pos + 1 == len(metadata):
                    break
                cur_label_pos += 1
            if cur_label_pos == len(metadata):
                continue
            # else - we take last known event as label of the last data points.
            cur_label = metadata[cur_label_pos-1][0]

            # cutting the current
            cur_data = data[(point - welch_range):point]
            fsp, cur_welch =welch(cur_data, fs=250, nperseg=250, axis=0)
            cur_welch =  np.asarray(cur_welch)
            welch_array.append(cur_welch)

            if len(welch_array) < memory_length + 1:   # need to fill start buffer for compares
                continue

            # compute welch differences
            cur_welch_diff = [cur_welch - x for x in welch_array[-5:-1]]

            # append values
            psd_dif.append(cur_welch_diff)
            labels_processed.append(cur_label)

            # saving examples
            if j in [251, 478, 1032, 1065, 7651, 7971, 8521, 8983]:
                examples.append((fsp, cur_welch_diff, cur_label))

        psd_dif = np.asarray(psd_dif).reshape((len(psd_dif),-1)) # flattening the results for network
        labels_processed = np.asarray(labels_processed)
        # keep only samples with class label 1-4
        valid_labels = [769, 770, 771, 772]
        bool_arrays = [labels_processed == x for x in valid_labels]
        args = np.nonzero(np.logical_or(np.logical_or(bool_arrays[0], bool_arrays[1]),
                                        np.logical_or(bool_arrays[2], bool_arrays[3])))
        labels_clean = labels_processed[args] - 769
        psd_clean = psd_dif[args]

        return psd_clean, labels_clean, examples

    if method == None:
        examples=[]
        data_processed, labels_processed = [[], []], [[], []]
        if params['tmi']:
            for i in params['tmi']:
                cur_segment = x_train[i - 500:i + 500]
                data_processed[0].append(cur_segment)  # Note(!) we take only 20 first bins of the FFT (~0-20 Hz)
                labels_processed[0].append(y_train[i])
        if params['emi']:
            for i in params['emi']:
                cur_segment = x_test[i - 500:i + 500]
                data_processed[1].append(cur_segment)  # Note(!) we take only 20 first bins of the FFT (~0-20 Hz)
                labels_processed[1].append(y_test[i])
        return tuple([np.asarray(x) for x in [data_processed[0], data_processed[1], labels_processed[0], labels_processed[1]]]),\
               examples

def get_welch(data, labels, valid_indices=None):
    data_processed, labels_processed = [], []
    if valid_indices:
        for i in valid_indices:
            cur_segment = data[i - 500:i + 500]
            fsp, cur_welch = welch(cur_segment, nperseg=250, fs=250, axis=0)
            data_processed.append(cur_welch[:20].flatten())  # Note(!) we take only 20 first bins of the FFT (~0-20 Hz)
            labels_processed.append(labels[i])
    else:
        for i in range(500, data.shape[0] - 500, 250):
            cur_segment = data[i - 500:i + 500]
            fsp, cur_welch = welch(cur_segment, nperseg=250, fs=250, axis=0)
            data_processed.append(cur_welch[:20].flatten())
            labels_processed.append(labels[i])
    return data_processed, labels_processed