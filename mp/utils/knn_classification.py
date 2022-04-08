import torch
import numpy as np

from pathlib import Path
from argparse import ArgumentParser
from skhubness.analysis import Hubness
from mp.utils.utils import get_kl_divs, get_mp_dists


def get_clean_labels(label_file):
    """ Prepare annotations (remove files w/o annotations, remove empty annotations). """
    with open(label_file, 'r') as fp:
        labels = [l.rstrip().split(',') for l in fp]

    labels = {int(l[0]): l[1:] for l in labels}
    clean_labels = {k: labels[k] for k in labels.keys() if len(labels[k][0]) > 0}
    clean_labels = {k: [i for i in clean_labels[k] if len(i) > 0] for k in clean_labels.keys()}

    return clean_labels


def get_knn_accuracies(save_path, label_file):
    """ Computes accuracies for KL/MP kNN-based leave one out classification. """
    # prepare distances, hubness computation
    device = torch.device('cpu')
    kl_divs = get_kl_divs(save_path, device)
    mp_dists = get_mp_dists(save_path, device)
    hub = Hubness(k=5, hub_size=5, return_value='k_neighbors',  metric='precomputed', store_k_neighbors=True)
    clean_labels = get_clean_labels(label_file)

    # get accuracy for KL divergences
    hub.fit(kl_divs.cpu().numpy())
    k_neighbours = hub.score(has_self_distances=True)
    kl_accuracies = get_knn_accuracy(k_neighbours, clean_labels)

    # get accuracy for MP distances
    hub.fit(mp_dists.cpu().numpy())
    k_neighbours = hub.score(has_self_distances=True)
    mp_accuracies = get_knn_accuracy(k_neighbours, clean_labels)

    print('kNN accuracy using KL: {} +/- {}'.format(np.mean(kl_accuracies), np.std(kl_accuracies)))
    print('kNN accuracy using MP: {} +/- {}'.format(np.mean(mp_accuracies), np.std(mp_accuracies)))

    return kl_accuracies, mp_accuracies


def get_knn_accuracy(neighbours, labels):
    """ Computes kNN classification accuracy given labels and neighbour indices. """
    nr_files = float(len(labels))
    acc = []
    unsuccessful = 0

    for fi, file_idx in enumerate(list(labels.keys())):
        print('File {}/{}\r'.format(fi, int(nr_files)), flush=True, end='')
        # get current (true) labels
        cur_labels = np.array(labels[file_idx])

        cur_acc = 0
        # get current neighbours
        cur_neigh = [n for n in neighbours[file_idx] if n in labels.keys()]
        for cn in cur_neigh:
            cur_neigh_labels = np.array(labels[cn])

            # compute how far labels are equal
            true = set(cur_labels)
            guess = set(cur_neigh_labels)
            cur_acc += len(true.intersection(guess)) / len(true.union(guess))
        if len(cur_neigh) > 0:
            cur_acc = cur_acc / len(cur_neigh)
        else:
            unsuccessful += 1
        acc.append(cur_acc)

    print('{} files had no neighbours at all'.format(unsuccessful))

    return np.array(acc)


def main():
    descr = 'Computes kNN accuracies before/after MP.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('--save_path', metavar='DIR', type=str,
                        help='Directory where gaussians and kl-divs were stored during preprocessing.')
    parser.add_argument('--label_file', metavar='FILE', type=str,
                        help='File containing labels of files, line-format: index, '
                             'labels (can be zero, one, multiple separated with a comma).')
    options = parser.parse_args()
    save_path = Path(options.save_path)
    label_file = Path(options.label_file)

    if not save_path.exists():
        raise NotADirectoryError('Please define valid save path!')
    if not label_file.exists():
        raise FileNotFoundError('Please define valid label file!')

    get_knn_accuracies(save_path, label_file)


if __name__ == '__main__':
    main()
