import scipy.io as sio
import sys, os
import numpy as np

# add directories in src/ to path
sys.path.insert(0, '/home/ran/PycharmProjects/SpectralNet_eeg/src/')

from core.data import get_data, pre_process, eeg_preprocess
# from core.data_backup import eeg_preprocess
from applications.spectralnet import run_net
from core import statistics

MOTORIC_IMAGERY_ONLY=True

# define hyperparameters
params = {
    'dset': 'bci_iv_2a',
    'mode': 'Takens_imagery_only',
    'retrain': True,                   # To add in code - ignore pre-trained weights and re-train the network
    'val_set_fraction': 0.1,
    'standardize': True,                # Standarize N(0,1) the input train data
    'siam_batch_size': 512,
    'n_clusters': 4,
    'affinity': 'siamese',
    'train_labeled_fraction': 0.5,
    'val_labeled_fraction': 0.5,
    'n_nbrs': 5,
    'scale_nbrs': 5,
    'scale_nbr': 5,
    'siam_k': 10,
    'siam_ne': 50,
    'spec_ne': 50,              # For Debug, change to higher number (~100) when running
    'siam_lr': 1e-3,
    'spec_lr': 1e-4,
    'siam_patience': 5,
    'spec_patience': 10,
    'siam_drop': 0.3,
    'spec_drop': 0.3,
    'batch_size': 1024,
    'batch_size_orthonorm': 512,
    'siam_reg': None,
    'spec_reg': None,
    'siam_n': None,
    'siamese_tot_pairs': 400000,
    'arch': {'siamese':
                [
                    {'type': 'relu', 'size': 1024},
                    {'type': 'relu', 'size': 512},
                    {'type': 'relu', 'size': 256},
                    {'type': 'relu', 'size': 20},
                ],
            'spectral':
                [
                    {'type': 'relu', 'size': 4096},
                    {'type': 'relu', 'size': 4096},
                    {'type': 'relu', 'size': 2048},
                    {'type': 'relu', 'size': 1024},
                ]
            },
    'use_approx': True,
    'use_all_data': False,
    'generalization_metrics': None,
    'nan': 999,             # switch NaN in labels to this value
}

classes_events = [769, 770, 771, 772] #, 783]
event_to_class = {769: 1,
                  770: 2,
                  771: 3,
                  772: 4}

exp_list = ['A0{}'.format(int(x)) for x in range(1,9)]
for exp in exp_list:
    # Update params for results saves
    data_path = os.path.join('/home/ran/Databases/BCICIV', params['dset'])
    results_path = os.path.join(os.path.dirname(sys.argv[0]), 'Results', params['mode'], exp)
    logs_path = os.path.join(os.path.dirname(sys.argv[0]), 'Logs', params['mode'], exp)
    params.update({'dpath' : data_path,
                   'results_path': results_path,
                   'logs_path': logs_path})

    # load current experiment data
    data_path = '/home/ran/Databases/BCICIV/bci_iv_2a'
    eval_x = sio.loadmat(os.path.join(data_path, exp + 'E_data.mat'))['s'][:, :-3]
    eval_metadata = sio.loadmat(os.path.join(data_path, exp + 'E_metadata.mat'))['meta_data']
    train_x = sio.loadmat(os.path.join(data_path, exp + 'T_data.mat'))['s'][:, :-3]
    train_metadata = sio.loadmat(os.path.join(data_path, exp + 'T_metadata.mat'))['meta_data']
    train_GT_y = sio.loadmat(os.path.join(data_path, 'y_labels', exp + 'T.mat'))['classlabel']
    eval_GT_y = sio.loadmat(os.path.join(data_path, 'y_labels', exp + 'E.mat'))['classlabel']

    # change GT labels to [0-3] range
    train_GT_y = [x[0] for x in train_GT_y]
    eval_GT_y = [x[0] for x in eval_GT_y]

    # extract labels from metadata files
    train_y = np.zeros(train_x.shape[0])
    train_motoric_indices = []
    for entry in train_metadata:
        if entry[0] in classes_events:
            train_y[entry[1] + 250:entry[1] + (250 * 4)] = event_to_class[entry[0]]   # according to experiment, this is the time where the motor imagery is happening
            train_motoric_indices.append(entry[1] + 250)

    eval_y = np.zeros(eval_x.shape[0])
    event_idx=0
    eval_motoric_indices = []
    for entry in eval_metadata:
        if entry[0] == 783:
            eval_y[entry[1] + 250:entry[1] + (250 * 4)] = eval_GT_y[event_idx] - 1
            eval_motoric_indices.append(entry[1] + 250)
            event_idx += 1

    # if Siamese net maps everything to zeros - try normalizing input data by the following line:
    # train_x, eval_x = pre_process(train_x, eval_x, True)  # Normalizing the train data

    new_dataset_data = (train_x / 1e3, eval_x / 1e3, train_y, eval_y) # returning to miliVolts

    # preprocess dataset
    preprocess_params = {'tau': 15,
                         'ndelay': 7,
                         'tmi': train_motoric_indices,
                         'emi': eval_motoric_indices
                         }

    # statistics.show_fft_examples(new_dataset_data, 250, train_motoric_indices[50:55])
    mode='takens'  # change between None\takens\welch
    preprocessed_data, examples = eeg_preprocess(new_dataset_data, mode, params=preprocess_params)
    statistics.distances(preprocessed_data, mode)
    if 'takens' in params['mode'].lower():
        delay = preprocess_params['tau'] * preprocess_params['ndelay']
    else:
        delay=0

    if MOTORIC_IMAGERY_ONLY:
        if 'takens' in params['mode'].lower():
            delay = preprocess_params['tau'] * preprocess_params['ndelay']
            train_x_motoric = np.concatenate(tuple([preprocessed_data[0][x-delay:x + 750 - delay] for x in train_motoric_indices]), axis=0)
            train_y_motoric = np.concatenate(tuple([preprocessed_data[2][x-delay:x + 750 - delay] for x in train_motoric_indices]), axis=0)
            eval_x_motoric = np.concatenate(tuple([preprocessed_data[1][x-delay:x + 750 - delay] for x in eval_motoric_indices]), axis=0)
            eval_y_motoric = np.concatenate(tuple([preprocessed_data[3][x-delay:x + 750 - delay] for x in eval_motoric_indices]), axis=0)
        else:
            train_x_motoric = preprocessed_data[0]
            train_y_motoric = preprocessed_data[2]
            eval_x_motoric = preprocessed_data[1]
            eval_y_motoric = preprocessed_data[3]
    statistics.distances((train_x_motoric, eval_x_motoric, train_y_motoric, eval_y_motoric), 'takens_MI_only')
    # Exploring data statistics

    # Process into spectral-net format, create pairs for siamese network
    data = get_data(params, (train_x_motoric, eval_x_motoric, train_y_motoric, eval_y_motoric))

    # run spectral net
    x_spectralnet, y_spectralnet = run_net(data, params)
