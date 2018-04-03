'''
data.py: contains all data generating code for datasets used in the script
'''

import os, sys
import h5py
import pickle

import numpy as np
import core.util as util
from sklearn import preprocessing
from scipy.signal import welch

from keras import backend as K
from keras.datasets import mnist
from keras.models import model_from_json
from core.takens_embed import get_delayed_manifold as takens
from core import pairs




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

    # preprocessed data
    data_path = os.path.join(os.path.dirname(sys.argv[0]), 'data', params['dset'] + '_preprocessed')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    data_file = os.path.join(data_path,params['mode'] + '_' + params['exp_num'] + '_' + params['preprocess'] +
                             '_preprocessed.pickle')
    data_file_spectral = os.path.join(data_path, params['mode'] + '_' + params['exp_num'] + '_' + params['preprocess'] +
                             '_spectral_preprocessed.pickle')
    data_file_rest = os.path.join(data_path, params['mode'] + '_' + params['exp_num'] + '_' + params['preprocess'] +
                                      '_rest_preprocessed.pickle')

    # if already calculated preprocess - since it takes long
    # if os.path.exists(data_file):
    try:
        with open(data_file, 'rb') as handle:
            ret = pickle.load(handle)
        print("Loaded pickle file")
        return ret
    except:
        print("Did not load general pickle file")
    try:
        with open(data_file_spectral, 'rb') as handle:
            ret_spectral = pickle.load(handle)
        with open(data_file_rest, 'rb') as handle:
            ret_rest = pickle.load(handle)
        ret = ret_rest
        ret['spectral'] = ret_spectral
        return ret
    except:
        ("Failed loading splitted pickle files")



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
            verbose=True
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

        try:
            with open(data_file, 'wb') as handle:
                pickle.dump(ret, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("finished saving data to pickle file \n")
            return ret
        except MemoryError:
            os.remove(data_file)
            print("Memory error - did not save training pairs for siamese network")
            print("might result in long run-times if re-running program")
        try:
            with open(data_file_spectral, 'wb') as handle:
                pickle.dump(ret['spectral'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            ret_rest = {x: ret[x] for x in ret if x not in ["spectral"]}
            with open(data_file_rest, 'wb') as handle:
                pickle.dump(ret_rest, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            os.remove(data_file_spectral)
            os.remove(data_file_rest)
            print("failed saving any pickle")

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
    elif params['dset'] == 'bci_iv_1':
        x_train, x_test, y_train, y_test = get_bci_iv_1(params)
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

def get_bci_iv_1(params):
    # ==== paths filenames ==== #
    dp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/BCI_IV_1')
    exp_files = [x for x in os.listdir(dp) if params['exp_num'] in x]
    exp_files = [os.path.join(dp, x) for x in exp_files if params['mode'] in x]
    x_file = [x for x in exp_files if 'cnt' in x][0]
    if params['mode'] == 'calib':
        y_file =  [x for x in exp_files if 'mrk' in x][0]
    if params['mode'] == 'eval':
        y_file = [x for x in exp_files if 'true_y' in x][0]

    # x_file = os.path.join(dp, 'BCICIV_calib_' + params['exp_num'] + '_cnt.txt')
    # y_file = os.path.join(dp, 'BCICIV_calib_' + params['exp_num'] + '_mrk.txt')

    # ==== loading in base format ==== #

    if not os.path.exists(x_file):
        raise()
    with open(x_file, 'r') as f:
        x = np.asarray([[int(z) for z in y.split('\t')] for y in f.readlines()])
    with open(y_file, 'r') as f:
        if params['mode'] == 'calib':
            y = np.asarray([[int(float(z)) for z in x.split('\t')] for x in f.readlines()])
            y_vec = vectorize_bci_labels(y, x.shape[0])
        elif params['mode'] == 'eval':
            y = np.asarray([[float(z) for z in x.split('\t')] for x in f.readlines()])
            y = y[::10]  # take every 10th label
            y_vec = y[:x.shape[0], 0]  # cut if there is 1 extra label - since y is not divided by 10
            assert x.shape[0] == y_vec.shape[0]   #  IMPORTANT!


    # x = x[first_ind:]
    # for data integrity, splitting at the start of each visual cue session - it matters where we cut!

    # changing -1 in y array to 2, since algorithm demands classes to be natural numbers
    y_vec[y_vec == -1] = 2
    # y_vec[np.argwhere(np.isnan(y_vec))] = params['nan']

    # # since x is uint64, with numbers between -32000 to 16000 we normalize it (stability for siamese net issue?
    x = (x - np.mean(x)) / np.max([np.abs(np.min(x)), np.abs(np.max(x))])

    # util.run_and_save_fft_examples(x, y_vec, params)

    # PREPROCESSING X
    if params['preprocess']:
        x_processed = eeg_preprocess(x, params.get('preprocess'))
    if params['mode'] == 'calib':
        slice_ind = y[int(len(y) * 0.8), 0]
    else:
        slice_ind = int(len(y_vec) * 0.8)
    x_train, x_test = x_processed[:slice_ind], x_processed[slice_ind:]
    y_train, y_test = y_vec[:slice_ind], y_vec[slice_ind:x_processed.shape[0]]

    return x_train, x_test, y_train, y_test

def vectorize_bci_labels(y, x_length):
    '''
    change indexed labels to [-1, 0, 1] labels in the original length of x
    :param y:
    :return:
    '''
    y_vec = []
    for i in range(y.shape[0] - 1):
        y_vec = np.concatenate((y_vec, np.multiply(np.ones(400), y[i, 1])))
        y_vec = np.concatenate((y_vec, np.zeros(y[i+1, 0] - y[i, 0] - 400)))
    # add last action task
    y_vec = np.concatenate((y_vec, np.multiply(np.ones(400), y[-1, 1])))
    # pad with 0's to X data length
    y_vec = np.concatenate((y_vec, np.zeros(x_length - len(y_vec))))
    return y_vec



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



def eeg_preprocess(data, method):
    if method == 'takens':
        data_processed = takens(data, tau=10, ndelay=7)
        data_processed = np.transpose(data_processed, (1, 0, 2))
        data_processed = data_processed.reshape((data_processed.shape[0], -1))
    if method =='welch':
        f_sample_points, data_processed = welch(data, fs=100)

    return data_processed

