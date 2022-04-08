import torch

from pathlib import Path
from argparse import ArgumentParser
from skhubness.analysis import Hubness
from mp.utils.utils import get_kl_divs, get_mp_dists


def analysis(save_path, hub_size):
    """ Perform hubness analysis on KL vs MP distances. """
    device = torch.device('cpu')
    hub = Hubness(k=5, return_value='all', hub_size=hub_size, metric='precomputed',
                  store_k_occurrence=True, verbose=True)

    # load pre-computed KL divergences and MP distances
    kl_divs = get_kl_divs(save_path, device)
    mp_dists = get_mp_dists(save_path, device)

    # perform analysis
    hub.fit(kl_divs)
    kl_scores = hub.score(has_self_distances=True)
    hub.fit(mp_dists)
    mp_scores = hub.score(has_self_distances=True)

    print('KL scores: {}'.format(kl_scores))
    print('MP scores: {}'.format(mp_scores))


def main():
    parser = ArgumentParser(description='Script that performs simple hubness analysis on KL vs MP.')
    parser.add_argument('--save_path', metavar='DIR', type=str, required=True,
                        help='Path pointing to the directory where Gaussians etc. are stored.')
    parser.add_argument('--hub_size', type=int, default=5,
                        help='Size of hubs for analysis.')
    options = parser.parse_args()

    save_path = Path(options.save_path)
    if not save_path.exists():
        raise NotADirectoryError('Please define valid save path!')

    analysis(save_path, options.hub_size)


if __name__ == '__main__':
    main()
