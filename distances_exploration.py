import scipy.io as sio
import sys, os
import numpy as np

# add directories in src/ to path
sys.path.insert(0, '/home/ran/PycharmProjects/SpectralNet_eeg/src/')

from sklearn.decomposition import FastICA

from core.data import get_data, pre_process
from core.data_backup import eeg_preprocess
from applications.spectralnet import run_net
from core import statistics
from mne.decoding import CSP

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
    tst_idx = 0
    for entry in train_metadata:
        if entry[0] in classes_events:
            train_y[entry[1] + 250:entry[1] + (250 * 4)] = event_to_class[entry[0]]   # according to experiment, this is the time where the motor imagery is happening
            assert event_to_class[entry[0]] == train_GT_y[tst_idx]
            tst_idx += 1
            train_motoric_indices.append(entry[1] + 250)

    eval_y = np.zeros(eval_x.shape[0])
    event_idx=0
    eval_motoric_indices = []
    for entry in eval_metadata:
        if entry[0] == 783:
            eval_y[entry[1] + 250:entry[1] + (250 * 4)] = eval_GT_y[event_idx]
            eval_motoric_indices.append(entry[1] + 250)
            event_idx += 1

    # Preprocess EEG data to contain only segments with action classification (since this is our ground truth)

    # train_x, eval_x = pre_process(train_x, eval_x, True)  # Normalizing the train data

    # First Try - insert data as-is to the model. see if it gives anything other than non-sense
    new_dataset_data = (train_x / 1e3, eval_x / 1e3, train_y, eval_y) # returning to miliVolts

    # csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)


    def unprocessed_and_takens(new_dataset_data):
        statistics.distances(new_dataset_data, exp + '_unprocessed_data')

        # Siamese net maps everything to zeros
        # # Second Try
        # tmp = len(train_x)
        # new_dataset_data = (train_x[:int(tmp * 0.9)], train_x[int(tmp * 0.9):],
        #                     train_y[:int(tmp * 0.9)], train_y[int(tmp * 0.9):])

        # preprocess dataset
        preprocess_params = {'tau': 15,
                             'ndelay': 7,
                             'tmi': train_motoric_indices,
                             'emi': eval_motoric_indices
                             }

        # statistics.show_fft_examples(new_dataset_data, 250, train_motoric_indices[50:55])
        welch_preprocessed_data, examples = eeg_preprocess(new_dataset_data, 'welch', params=preprocess_params)
        statistics.distances(welch_preprocessed_data, exp + '_welch_mi_only')


        takens_delay = preprocess_params['tau'] * preprocess_params['ndelay']

        #  =========== Takens preprocessing ============== #
        takens_preprocessed_data, examples = eeg_preprocess(new_dataset_data, 'takens', params=preprocess_params)

        train_x_motoric = np.concatenate(tuple([takens_preprocessed_data[0][x-takens_delay:x + 750 - takens_delay] for x in train_motoric_indices]), axis=0)
        train_y_motoric = np.concatenate(tuple([takens_preprocessed_data[2][x-takens_delay:x + 750 - takens_delay] for x in train_motoric_indices]), axis=0)
        eval_x_motoric = np.concatenate(tuple([takens_preprocessed_data[1][x-takens_delay:x + 750 - takens_delay] for x in eval_motoric_indices]), axis=0)
        eval_y_motoric = np.concatenate(tuple([takens_preprocessed_data[3][x-takens_delay:x + 750 - takens_delay] for x in eval_motoric_indices]), axis=0)
        takens_mi_only = (train_x_motoric, eval_x_motoric, train_y_motoric, eval_y_motoric)
        statistics.distances(takens_mi_only, exp + '_takens_MI_only')

        # ======================================== #

        # =============== MI_only_no_preprocessing ============== #
        train_x_motoric = np.concatenate(tuple([new_dataset_data[0][x:x + 750] for x in train_motoric_indices]), axis=0)
        train_y_motoric = np.concatenate(tuple([new_dataset_data[2][x:x + 750] for x in train_motoric_indices]), axis=0)
        eval_x_motoric = np.concatenate(tuple([new_dataset_data[1][x:x + 750] for x in eval_motoric_indices]), axis=0)
        eval_y_motoric = np.concatenate(tuple([new_dataset_data[3][x:x + 750] for x in eval_motoric_indices]), axis=0)
        statistics.distances((train_x_motoric, eval_x_motoric, train_y_motoric, eval_y_motoric), exp + '_unprocessed_MI_only')

    # ========================= self Welch implemenetation for test ============= #

    def test_welch(new_dataset_data):
        fft_train_x, fft_test_x = [], []
        fft_train_y, fft_test_y = [], []
        for x in range(0, new_dataset_data[0].shape[0], 250):
            fft_samp = np.abs(np.fft.fft(new_dataset_data[0][x:x+250])[5:50].flatten())
            fft_train_x.append(fft_samp)
            fft_label = np.argmax(np.bincount(new_dataset_data[2][x:x+250].astype('int')))
            fft_train_y.append(fft_label)
        for x in range(0, new_dataset_data[1].shape[0], 250):
            fft_samp = np.abs(np.fft.fft(new_dataset_data[1][x:x+250])[5:50].flatten())
            fft_test_x.append(fft_samp)
            fft_label = np.argmax(np.bincount(new_dataset_data[3][x:x+250].astype('int')))
            fft_test_y.append(fft_label)

        fft_data = [np.asarray(x[:-1]) for x in [fft_train_x, fft_test_x, fft_train_y, fft_test_y]]
        # fft_train_x = np.asarray(fft_train_x[:-1])
        # fft_train_y = np.asarray(fft_train_y[:-1])
        fft_avg = [[], [], [], []]
        # calculating mean over each 4 FFT windows (each 4 seconds)
        for i, x in enumerate(fft_data[0]):
            fft_avg[0].append(np.mean(fft_data[0][max(i-2, 0):min(fft_data[0].shape[0], i+2)], axis=0))
            fft_avg[2].append(fft_data[2][i])
        for i, x in enumerate(fft_data[1]):
            fft_avg[1].append(np.mean(fft_data[1][max(i - 2, 0):min(fft_data[1].shape[0], i + 2)], axis=0))
            fft_avg[3].append(fft_data[3][i])

        fft_avg_arr = [np.asarray(x) for x in fft_avg]
        # fft_avg_arr[2] = np.squeeze(fft_avg_arr[2])
        # fft_avg_arr[3] = np.squeeze(fft_avg_arr[3])
        statistics.distances(fft_avg_arr, exp + '_mean_of_windows_fft')
        return fft_avg_arr

    def test_ICA(new_dataset_data):
        ica = FastICA(n_components=22)
        S_train = ica.fit_transform(new_dataset_data[0])
        S_test = ica.fit_transform(new_dataset_data[1])
        statistics.distances((S_train, S_test, new_dataset_data[2], new_dataset_data[3]), exp + '_ICA_decomp')

    test_welch(new_dataset_data)
    print("stop point")


