import sys, os

# add directories in src/ to path
sys.path.insert(0, 'path_to_spectralnet/src/')

# import run_net and get_data
from spectralnet import run_net
from src.core.data import get_data

# define hyperparameters
params = {
    'dset': 'new_dataset',
    'val_set_fraction': ...,
    'siam_batch_size': ...,
    'n_clusters': ...,
    'affinity': ...,
    'n_nbrs': ...,
    'scale_nbrs': ...,
    'siam_k': ...,
    'siam_ne': ...,
    'spec_ne': ...,
    'siam_lr': ...,
    'spec_lr': ...,
    'siam_patience': ...,
    'spec_patience': ...,
    'siam_drop': ...,
    'spec_drop': ...,
    'batch_size': ...,
    'siam_reg': ...,
    'spec_reg': ...,
    'siam_n': ...,
    'siamese_tot_pairs': ...,
    'arch': [
        {'type': 'relu', 'size': ...},
        {'type': 'relu', 'size': ...},
        {'type': 'relu', 'size': ...},
    ],
    'use_approx': ...,
}

# load dataset
x_train, x_test, y_train, y_test = load_new_dataset_data()
new_dataset_data = (x_train, x_test, y_train, y_test)

# preprocess dataset
data = get_data(params, new_dataset_data)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)