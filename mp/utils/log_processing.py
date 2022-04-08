import csv
import numpy as np

from pathlib import Path
from argparse import ArgumentParser


def read_log_file(log_file_path):
    """ Reads log files, returns column-names and entries. """
    with open(log_file_path) as csv_file:
        log_lines = [l for l in csv.reader(csv_file, delimiter=',')]
    cols = log_lines[0]
    log_entries = log_lines[1:]
    return np.array(cols), np.array(log_entries)


def read_log_files(log_files):
    """ Reads list of log files, concatenates results - assumes same columns! """
    cols, total_log_entries = read_log_file(log_files[0])
    for log_file_path in log_files[1:]:
        _, log_entries = read_log_file(log_file_path)
        total_log_entries = np.append(total_log_entries, log_entries, axis=0)

    return cols, total_log_entries


def get_nr_converged_files(log_file_path):
    """ Determines how much adversaries were found/ files were hubs in the first place. """
    cols, log_entries = read_log_files(log_file_path)
    conv_idx, = np.asarray(cols == 'conv').nonzero()
    conv_idx = conv_idx.item()
    conv_entries = log_entries[:, conv_idx]
    conv_options, counts = np.unique(conv_entries, return_counts=True)
    conv_options = np.append(conv_options, 'total')
    counts = np.append(counts, np.sum(counts))
    print(['{}: {}'.format(o, c) for o, c in zip(conv_options, counts)])
    return conv_options, counts


def get_snr(log_file_path):
    """ Returns average / std of SNR of adversaries (vs original files). """
    cols, log_entries = read_log_files(log_file_path)
    conv_idx, = np.asarray(cols == 'conv').nonzero()
    db_idx, = np.asarray(cols == 'db').nonzero()
    conv_idx, db_idx = conv_idx.item(), db_idx.item()
    db_entries = np.array(log_entries[log_entries[:, conv_idx] == 'yes', db_idx], dtype=np.float)
    print('SNR: {} +- {} ({} files)'.format(np.mean(db_entries), np.std(db_entries), len(db_entries)))
    return np.mean(db_entries), np.std(db_entries), db_entries


def get_k_occ(log_file_path):
    """ Returns average / std of k-occurrence of adversarial hubs. """
    cols, log_entries = read_log_files(log_file_path)
    conv_idx, = np.asarray(cols == 'conv').nonzero()
    k_occ_idx, = np.asarray(cols == 'kocc').nonzero()
    conv_idx, k_occ_idx = conv_idx.item(), k_occ_idx.item()
    k_occ_entries = np.array(log_entries[log_entries[:, conv_idx] == 'yes', k_occ_idx], dtype=np.int)
    print('K-occurrence: {} +- {} ({} files)'.format(np.mean(k_occ_entries), np.std(k_occ_entries), len(k_occ_entries)))
    return np.mean(k_occ_entries), np.std(k_occ_entries), k_occ_entries


def main():
    parser = ArgumentParser(description='Program to process log files of hubness adversaries')
    parser.add_argument('--log_files', nargs='+', required=True)
    args = parser.parse_args()
    print(args.log_files)

    log_files = [Path(log_file) for log_file in args.log_files]
    for log_file in log_files:
        if not log_file.exists():
            ValueError('Please define valid log-file path!')

    get_nr_converged_files(log_files)
    get_snr(log_files)
    get_k_occ(log_files)


if __name__ == '__main__':
    main()
