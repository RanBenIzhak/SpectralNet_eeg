from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from core.costs import euclidean_distance

def show_fft_examples(new_dataset_data, fs, indices):
    '''
    Shows averaged FFT (welch-like) of 1000 samples around each of Indices

    :param new_dataset_data:
    :param fs:
    :param average:
    :param indices:
    :return:
    '''

    for i in indices:
        chunks = np.asarray([fft(new_dataset_data[0][i-500+x:i+x], axis=0)
                 for x in range(0, 750, 250)])
        label = new_dataset_data[2][i]

        avg_fft = np.mean(chunks, axis=0)
        avg_fft_pos = np.abs(avg_fft[:125])

        fig = plt.figure()
        plt.plot(avg_fft_pos[:,15:20])
        plt.show()



def distances(data, name=None):
    '''
    data[2] - train labels
    :param data:
    :return:
    '''
    n_points = 400
    classes = np.unique(data[2]).astype('int')
    class_indices = [np.random.permutation(np.argwhere(data[2] == x)) for x in classes]
    # select 100 / max number of points from each class
    ppc = np.asarray([min(n_points, len(x)) for x in class_indices])
    first_indices = [data[0][class_indices[x][:ppc[x]]] for x in range(len(class_indices))]
    data_shape = first_indices[0].shape
    if len(data_shape) > 2:
        first_indices = [first_indices[x].reshape((first_indices[x].shape[0], -1)) for x in range(len(class_indices))]

    # calculate distances and nearest neighbours between all pairs of point in all 4 classes
    concat_400_points = np.concatenate(first_indices, axis=0)
    n_points = concat_400_points.shape[0]
    result = np.empty((n_points, n_points))
    for i, point in enumerate(concat_400_points):
        point_tile = np.tile(point, (n_points, 1))
        distances = np.abs(np.sum(point_tile-concat_400_points, axis=1))
        result[i] = distances

    # Checking nearest neighbour for each point and statistics for each point
    nn_list = np.argsort(result, axis=1)[:, 1]
    ppc_cumsum = np.cumsum(ppc)
    nn_class = [int(np.argwhere(ppc_cumsum > x)[0]) for x in nn_list]
    nn_count = [np.bincount(nn_class[x*ppc[x]:x*ppc[x]+ppc[x]]) for x in range(len(class_indices))]

    save_name = '/home/ran/PycharmProjects/SpectralNet_eeg/Results/Statistics/Bin_count_nn_100_' + name + '.png'
    matrix_plot(np.asarray(nn_count), classes, save_name)
    print("Calculated Nearest neighbours heatmap - results at: " + save_name)

def matrix_plot(mat, cats, savepath=None):
    fig, ax = plt.subplots()
    ax.matshow(mat, cmap=plt.cm.Blues)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = mat[j, i]
            ax.text(i, j, str(c)[:4], va='center', ha='center')
    plt.xticks(range(mat.shape[1]), cats, rotation=45)
    plt.yticks(range(mat.shape[0]), cats, rotation=45)
    if savepath:
        plt.savefig(savepath)