import numpy as np

from pathlib import Path
from argparse import ArgumentParser
from mp.utils.log_processing import read_log_files


def check_k_occs(kaft_idx, kbef_idx, res):
    """ Prints / plots comparison of k-occurrences before and after MP defence. """
    # get k-occurrences
    k_occs_bef = np.array(res[:, kbef_idx], dtype=np.int)
    k_occs_aft = np.array([l.strip('|').split('|')[0] for l in res[:, kaft_idx]], dtype=np.int)
    print('k-occurrences before: {} +/- {}'.format(np.mean(k_occs_bef), np.std(k_occs_bef)))
    print('k-occurrences after: {} +/- {}'.format(np.mean(k_occs_aft), np.std(k_occs_aft)))

    return k_occs_aft, k_occs_bef


def process_mp_log(log_files):
    """ Method to analyse mp-defense log. """
    column_headers, res = read_log_files(log_files)

    # get column indices
    aft_idx, = np.asarray(column_headers == 'after').nonzero()[0]
    kbef_idx, = np.asarray(column_headers == 'k-occ before').nonzero()[0]
    kaft_idx, = np.asarray(column_headers == 'k-occ after').nonzero()[0]

    k_occs_aft, k_occs_bef = check_k_occs(kaft_idx, kbef_idx, res)

    # check hubness
    after_hubs = np.array([l.strip('|').split('|') for l in res[:, aft_idx]])
    any_hub = np.array([True if np.any(l == 'True') else False for l in after_hubs])
    print('{}/{} defended against'.format(len(any_hub) - np.sum(any_hub), len(any_hub)))

    return k_occs_bef, k_occs_aft, any_hub


def main():
    parser = ArgumentParser(description='Program to process log files of mutual proximity defence')
    parser.add_argument('--log_files', nargs='+', required=True)
    args = parser.parse_args()
    print(args.log_files)

    log_files = [Path(log_file) for log_file in args.log_files]
    for log_file in log_files:
        if not log_file.exists():
            ValueError('Please define valid log-file path!')

    process_mp_log(log_files)


if __name__ == '__main__':
    main()
