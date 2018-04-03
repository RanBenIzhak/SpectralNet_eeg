'''
util.py: contains various utility functions used in the models
'''

from contextlib import contextmanager
import os, sys
from mpl_toolkits.mplot3d import Axes3D

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
import numpy as np
from scipy.stats import norm
import sklearn.metrics
from scipy import stats

from keras import backend as K
from keras.callbacks import Callback
from keras.engine.training import _make_batches
import tensorflow as tf

from core import costs as cf
from munkres import Munkres
import matplotlib.pyplot as plt

def train_gen(pairs_train, dist_train, batch_size):
    '''
    Generator used for training the siamese net with keras

    pairs_train:    training pairs
    dist_train:     training labels

    returns:        generator instance
    '''
    batches = _make_batches(len(pairs_train), batch_size)
    while 1:
        random_idx = np.random.permutation(len(pairs_train))
        for batch_start, batch_end in batches:
            p_ = random_idx[batch_start:batch_end]
            x1, x2 = pairs_train[p_, 0], pairs_train[p_, 1]
            y = dist_train[p_]
            yield([x1, x2], y)

def make_layer_list(arch, network_type=None, reg=None, dropout=0):
    '''
    Generates the list of layers specified by arch, to be stacked
    by stack_layers (defined in src/core/layer.py)

    arch:           list of dicts, where each dict contains the arguments
                    to the corresponding layer function in stack_layers

    network_type:   siamese or spectral net. used only to name layers

    reg:            L2 regularization (if any)
    dropout:        dropout (if any)

    returns:        appropriately formatted stack_layers dictionary
    '''
    layers = []
    if type(arch) == dict:
        arch = arch[network_type]
    for i, a in enumerate(arch):
        layer = {'l2_reg': reg}
        layer.update(a)
        if network_type:
            layer['name'] = '{}_{}'.format(network_type, i)
        layers.append(layer)
        if a['type'] != 'Flatten' and dropout != 0:
            dropout_layer = {
                'type': 'Dropout',
                'rate': dropout,
                }
            if network_type:
                dropout_layer['name'] = '{}_dropout_{}'.format(network_type, i)
            layers.append(dropout_layer)
    return layers

class LearningHandler(Callback):
    '''
    Class for managing the learning rate scheduling and early stopping criteria

    Learning rate scheduling is implemented by multiplying the learning rate
    by 'drop' everytime the validation loss does not see any improvement
    for 'patience' training steps
    '''
    def __init__(self, lr, drop, lr_tensor, patience):
        '''
        lr:         initial learning rate
        drop:       factor by which learning rate is reduced by the
                    learning rate scheduler
        lr_tensor:  tensorflow (or keras) tensor for the learning rate
        patience:   patience of the learning rate scheduler
        '''
        super(LearningHandler, self).__init__()
        self.lr = lr
        self.drop = drop
        self.lr_tensor = lr_tensor
        self.patience = patience

    def on_train_begin(self, logs=None):
        '''
        Initialize the parameters at the start of training (this is so that
        the class may be reused for multiple training runs)
        '''
        self.assign_op = tf.no_op()
        self.scheduler_stage = 0
        self.best_loss = np.inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        '''
        Per epoch logic for managing learning rate and early stopping
        '''
        stop_training = False
        # check if we need to stop or increase scheduler stage
        if isinstance(logs, dict):
            loss = logs['val_loss']
        else:
            loss = logs
        if loss <= self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.scheduler_stage += 1
                self.wait = 0

        # calculate and set learning rate
        lr = self.lr * np.power(self.drop, self.scheduler_stage)
        K.set_value(self.lr_tensor, lr)

        # built in stopping if lr is way too small
        if lr <= 1e-7:
            stop_training = True

        # for keras
        if hasattr(self, 'model'):
            if hasattr(self.model, 'stop_trainning'):
                self.model.stop_training = stop_training

        return stop_training

def get_scale(x, batch_size, n_nbrs):
    '''
    Calculates the scale* based on the median distance of the kth
    neighbors of each point of x*, a m-sized sample of x, where
    k = n_nbrs and m = batch_size

    x:          data for which to compute scale
    batch_size: m in the aforementioned calculation. it is
                also the batch size of spectral net
    n_nbrs:     k in the aforementeiond calculation.

    returns:    the scale*

    *note:      the scale is the variance term of the gaussian
                affinity matrix used by spectral net
    '''
    n = len(x)

    # sample a random batch of size batch_size
    sample = x[np.random.randint(n, size=batch_size), :]
    # flatten it
    sample = sample.reshape((batch_size, np.prod(sample.shape[1:])))

    # compute distances of the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(sample)
    distances, _ = nbrs.kneighbors(sample)

    # return the median distance
    return np.median(distances[:, n_nbrs - 1])

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i,j]
            cost_matrix[j,i] = s-t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels

def get_accuracy(cluster_assignments, y_true, n_clusters):
    '''
    Computes the accuracy based on the provided kmeans cluster assignments
    and true labels, using the Munkres algorithm

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    y_pred, confusion_matrix = get_y_preds(cluster_assignments, y_true, n_clusters)
    # calculate the accuracy
    return np.mean(y_pred == y_true), confusion_matrix

def print_accuracy(cluster_assignments, y_true, n_clusters, params, nmi_score, extra_identifier=''):
    '''
    Convenience function: prints the accuracy
    '''
    # get accuracy
    accuracy, confusion_matrix = get_accuracy(cluster_assignments, y_true, n_clusters)
    # get the confusion matrix
    print('confusion matrix{}: '.format(extra_identifier))
    print(confusion_matrix)
    print('spectralNet{} accuracy: '.format(extra_identifier) + str(np.round(accuracy, 3)))
    with open(os.path.join(params['results_path'], 'NMI_acc_report.txt'), 'w') as f:
        print("spectralNet accuracy: {}".format(np.round(accuracy, 3)), file=f)
        print("spectralNet NMI: {}".format(str(np.round(nmi_score, 3))), file=f)

    return accuracy, nmi_score

def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    '''
    Using either a newly instantiated ClusterClass or a provided
    cluster_obj, generates cluster assignments based on input data

    x:              the points with which to perform clustering
    cluster_obj:    a pre-fitted instance of a clustering class
    ClusterClass:   a reference to the sklearn clustering class, necessary
                    if instantiating a new clustering class
    n_clusters:     number of clusters in the dataset, necessary
                    if instantiating new clustering class
    init_args:      any initialization arguments passed to ClusterClass

    returns:    a tuple containing the label assignments and the clustering object
    '''
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    cluster_assignments = None
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def get_y_preds(cluster_assignments, y_true, n_clusters):
    '''
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)

    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset

    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    '''
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred, confusion_matrix

def grassmann(A, B):
    '''
    Computes the Grassmann distance between matrices A and B

    A, B:       input matrices

    returns:    the grassmann distance between A and B
    '''
    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann

def spectral_clustering(x, scale, n_nbrs=None, affinity='full', W=None):
    '''
    Computes the eigenvectors of the graph Laplacian of x,
    using the full Gaussian affinity matrix (full), the
    symmetrized Gaussian affinity matrix with k nonzero
    affinities for each point (knn), or the Siamese affinity
    matrix (siamese)

    x:          input data
    n_nbrs:     number of neighbors used
    affinity:   the aforementeiond affinity mode

    returns:    the eigenvectors of the spectral clustering algorithm
    '''
    if affinity == 'full':
        W =  K.eval(cf.full_affinity(K.variable(x), scale))
    elif affinity == 'knn':
        if n_nbrs is None:
            raise ValueError('n_nbrs must be provided if affinity = knn!')
        W =  K.eval(cf.knn_affinity(K.variable(x), scale, n_nbrs))
    elif affinity == 'siamese':
        if W is None:
            print ('no affinity matrix supplied')
            return
    d = np.sum(W, axis=1)
    D = np.diag(d)
    # (unnormalized) graph laplacian for spectral clustering
    L = D - W
    Lambda, V = np.linalg.eigh(L)
    return(Lambda, V)

def run_and_save_embedding(epoch_num, embedded_data, labels, acc_array, params, loss=None, val_loss=None, mode='Spectral Net'):
    fn = os.path.join(params.get('results_path'), 'Embeddings')
    if not os.path.exists(fn):
        os.makedirs(fn)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for i, label in enumerate(np.unique(labels)):
        cur_data = embedded_data[labels == label, :]
        # Randomly show 100 points from each label, else plot is too crowded
        indices = np.random.randint(cur_data.shape[0], size=100)

        # using PCA for showing data in 3D
        pca = PCA(n_components=3)
        x_new = pca.fit_transform(cur_data[indices])
        ax.scatter(cur_data[indices, 0], cur_data[indices, 1], cur_data[indices, 2], c=colors[i])


    plt.title(mode + ' embedding (3 first dimensions) - Epoch ' + str(epoch_num))
    plt.savefig(os.path.join(fn, mode + ' embedding - Epoch ' + str(epoch_num)))
    if mode == 'Spectral Net':
        plt.figure()
        plt.plot(range(len(loss)), loss)
        plt.savefig(os.path.join(fn, 'Loss.png'))

        plt.figure()
        plt.plot(range(len(val_loss)), val_loss)
        plt.savefig(os.path.join(fn, 'Validation loss.png'))

        plt.figure()
        plt.plot(range(len(acc_array)), acc_array)
        plt.savefig(os.path.join(fn, 'Accuracy.png'))

    plt.close('all')

def run_and_save_fft_examples(data, labels, params, window_size=256, overlap=0.5):
    '''
    Saves FFT plots of window_size, non-averaged examples and averaged examples
    :param data: 2D array, [N_samples, N_channels]
    :param window_size: window size
    :return: None
    '''
    if not (np.log2(window_size) % 1) == 0:
        print('window size %d is not a power of 2, rounding to nearest power' % window_size)
        window_size = int(2 ** np.round(np.log2(window_size)))

    start_diff = int(window_size - window_size * overlap)
    dshape = data.shape
    N_windows = dshape[0] // window_size
    pad = N_windows % window_size
    data = np.pad(data, ((0, pad), (0, 0)), 'constant')
    data_fft = []
    avg_label = []
    for i in range(N_windows):
        window =data[i * start_diff:i * start_diff + window_size, :]
        fft = np.fft.fft(window, axis=1)
        fft_centered = np.fft.fft(window - np.mean(window, axis=0))
        label_array = np.asarray(labels[i * start_diff:i * start_diff + window_size])
        if not np.argwhere(np.isnan(label_array)).shape[0] == 0:
            continue
        data_fft.append(fft)
        label = np.argmax(np.bincount(label_array.astype('int')))
        avg_label.append(label)


    # after FFT calculated, randomly select from each label and plot & save
    data_fft = np.asarray(data_fft)
    avg_label = np.asarray(avg_label)
    channels = np.random.randint(59, size=10)
    for label in np.unique(avg_label):
        label_data = data_fft[avg_label == label]
        example_i = label_data[np.random.randint(label_data.shape[0])]
        plt.figure()
        plt.plot(np.arange(0, 100, 100 / 256), 20 * np.log10(2 * np.abs(example_i[:, channels]) / 256))
        plt.title('FFT of 10 channels - example label ' + str(int(label)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        fn = os.path.join(params.get('results_path'), 'examples')
        if not os.path.exists(fn):
            os.makedirs(fn)
        plt.savefig(os.path.join(fn, 'fft_example_label_' + str(int(label)) + '.png'))


    # Averaged FFT for each state
    for label in np.unique(avg_label):
        avg_per_label = np.mean(data_fft[avg_label == label], axis=0)
        plt.figure()
        plt.plot(np.arange(0, 100, 100 / 256), 20 * np.log10(2 * np.abs(avg_per_label[:, channels]) / 256))
        plt.title('FFT of 10 channels - average of label ' + str(int(label)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        plt.savefig(os.path.join(fn, 'fft_average_label_' + str(int(label)) + '.png'))

    plt.close('all')



def replace_nan(y):
    nan_indices = np.argwhere(np.isnan(y))
    valid_indices = np.argwhere(np.isnan(y) == False)
    NaN_label = np.max(y[valid_indices]) + 1.0
    y[nan_indices] = NaN_label
    return y

