import sys, os

# add directories in src/ to path
sys.path.insert(0, '/home/user/PycharmProjects/SpectralNet_eeg/src/')

# import run_net and get_data
from src.applications.spectralnet import run_net
from src.core.data import get_data

# define hyperparameters
params = {
    'exp_name': 'takens_embed',             # For naming output folders and tracking results
    'preprocess': 'takens',                     # Takens, Welch, None
    'collect_data_samples': True,               # weather to save some plots of the original data
    'dset': 'bci_iv_1',
    'exp_num': 'ds1b',
    'mode': 'calib',
    'retrain': True,                   # To add in code - ignore pre-trained weights and re-train the network
    'val_set_fraction': 0.1,
    # 'standardize': True,                # Standarize N(0,1) the input train data
    'siam_batch_size': 128,
    'n_clusters': 3,
    'affinity': 'siamese',
    'train_labeled_fraction': 0.5,
    'val_labeled_fraction': 0.5,
    'n_nbrs': 10,
    'scale_nbrs': 2,
    'scale_nbr': 2,
    'siam_k': 3,
    'siam_ne': 50,               # For Debug, change to higher number (~100) when running
    'spec_ne': 100,              # For Debug, change to higher number (~100) when running
    'siam_lr': 1e-3,
    'spec_lr': 5e-5,
    'siam_patience': 5,
    'spec_patience': 10,
    'siam_drop': 0.3,
    'spec_drop': 0.3,
    'batch_size': 1024,
    'batch_size_orthonorm': 512,
    'siam_reg': None,
    'spec_reg': None,
    'siam_n': None,
    'siamese_tot_pairs': 100000,
    'arch': {'siamese':
                [
                    {'type': 'relu', 'size': 1024},
                    {'type': 'relu', 'size': 1024},
                    {'type': 'relu', 'size': 512},
                    {'type': 'relu', 'size': 20},
                ],
            'spectral':
                [
                    {'type': 'relu', 'size': 4096},
                    {'type': 'relu', 'size': 2048},
                    {'type': 'relu', 'size': 2048},
                    {'type': 'relu', 'size': 1024},
                ]
            },
    'use_approx': False,
    'use_all_data': False,
    'nan': 999,             # switch NaN in labels to this value
}
# updating params according to other params
data_path = os.path.join(os.path.dirname(sys.argv[0]), params['mode'], 'data', params['dset'] + '_preprocessed')
results_path = os.path.join(os.path.dirname(sys.argv[0]), 'Results', params['mode'], params['exp_name'] + '_' + params['exp_num'])
logs_path = os.path.join(os.path.dirname(sys.argv[0]), 'Logs', params['mode'], params['exp_name'] + '_' + params['exp_num'])
params.update({'dpath' : data_path,
               'results_path': results_path,
               'logs_path': logs_path})

if not os.path.exists(results_path):
    os.makedirs(results_path)
if not os.path.exists(data_path):
    os.makedirs(data_path)



# load & preprocess dataset
data = get_data(params)
# new_dataset_data = (x_train, x_test, y_train, y_test)
#
# # preprocess dataset
# data = get_data(params, new_dataset_data)

# run spectral net
x_spectralnet, y_spectralnet = run_net(data, params)
print('stop')
