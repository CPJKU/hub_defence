import torch
import numpy as np

from pathlib import Path
from argparse import ArgumentParser
from skhubness.analysis import Hubness
from skhubness.reduction import MutualProximity

from mp.utils.io import read_config_file, Logger
from mp.soundpark.soundpark_model import SoundparkModel
from mp.soundpark.soundpark_data import get_raw_soundpark_loader
from mp.utils.utils import get_single_kl, get_gaussians, get_kl_divs


def opts_parser():
    descr = 'Performs defence against adversarial attack with mutual proximity.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('--adv_path', metavar='DIR', type=str,
                        help='Directory to adversaries that should be defended against.')
    parser.add_argument('--save_path', metavar='DIR', type=str,
                        help='Directory where gaussians and kl-divs were stored during preprocessing.')
    parser.add_argument('--start', default=0, type=int,
                        help='Index of start file.')
    parser.add_argument('--nr_files', type=int,
                        help='Total number of files that are defended against.')
    return parser


def do_defence(adv_data, clean_files, sp_model, hub, mp, gaussians, kl_divs, device, logger):
    """ Checks hubs before and after mutual proximity for each adversary, to see whether it worked as defence. """
    for idx, (_, data) in enumerate(adv_data):
        file = adv_data.dataset.files[idx]
        # copy distances for now
        cur_dists = kl_divs.clone().to(device)
        # get adv gaussian
        adv_gaussian, = sp_model(data.to(device))

        file_idx = [f.stem for f in clean_files].index(file.stem)
        print('Current file: {}, cur/original file idx: {}/{}'.format(file.stem, idx, file_idx))
        for j, j_name in enumerate(clean_files):
            cur_dists[file_idx, j] = cur_dists[j, file_idx] = get_single_kl(adv_gaussian, gaussians[j_name])

        # get hub scores before mp
        hub.fit(cur_dists.cpu())
        before_scores = hub.score(has_self_distances=True)
        res_vec = [file.stem, file_idx in before_scores['hubs'], before_scores['k_occurrence'][file_idx]]

        # perform mp
        neigh_ind = np.array([list(range(0, cur_dists.shape[1])) for _ in range(cur_dists.shape[0])])
        mp.fit(cur_dists.cpu().numpy(), neigh_ind)
        mp_dists, _ = mp.transform(cur_dists.cpu().numpy(), neigh_ind)

        # get hub scores after mp
        hub.fit(mp_dists)
        after_scores = hub.score(has_self_distances=True)
        after_log_var = file_idx in after_scores['hubs']
        after_k_log_var = after_scores['k_occurrence'][file_idx]
        after_hub_occ = after_scores['hub_occurrence']
        after_ahub_occ = after_scores['antihub_occurrence']

        # logging
        res_vec.extend([after_log_var, after_k_log_var, after_hub_occ, after_ahub_occ])
        logger.append(res_vec)


def prep_and_run(adv_path, start_idx, nr_files, save_path):
    """ Prepares everything for defence test. """
    # get soundpark model, original gaussians and KL divergences
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
    sp_model = SoundparkModel(device)
    orig_gaussians = get_gaussians(save_path, device)
    orig_kl_divs = get_kl_divs(save_path, torch.device('cpu'))

    # get right hub size, prepare hubness computation and mutual proximity
    orig_config = read_config_file(Path(adv_path) / 'config.vars')
    hub_size = orig_config['hub_size']
    hub = Hubness(k=5, return_value='all', hub_size=hub_size, metric='precomputed',
                  store_k_occurrence=True, verbose=True)
    mp = MutualProximity(method='empiric', verbose=True)

    # prepare data
    adv_files = list(Path(adv_path).rglob('*.wav'))
    all_files = list(orig_gaussians.keys())
    start_idx = 0 if start_idx < 0 else start_idx
    nr_files = len(adv_files) if not nr_files else nr_files
    nr_files = len(adv_files) - start_idx if start_idx + nr_files > len(adv_files) else nr_files
    print('looking at adv files {} to {} (#{})'.format(start_idx, start_idx + nr_files, nr_files))
    adv_data = get_raw_soundpark_loader(adv_files[start_idx:start_idx + nr_files], 1)

    assert(np.all(np.array([Path(f) for f in list(orig_gaussians.keys())]) == np.array(all_files)))

    # prepare logger
    log_file_name = str(adv_path / 'mp_log_{}_{}.csv'.format(start_idx, start_idx + nr_files))
    logger = Logger(log_file_name, ['file', 'before', 'k-occ before', 'after', 'k-occ after', 'hub_occ', 'ahub_occ'])

    # perform defence
    do_defence(adv_data, all_files, sp_model, hub, mp, orig_gaussians, orig_kl_divs, device, logger)


def main():
    # parse command line arguments
    parser = opts_parser()
    options = parser.parse_args()

    save_path = Path(options.save_path)
    if not save_path.exists():
        raise NotADirectoryError('Please define a valid save path for preprocessed files!')
    adv_path = Path(options.adv_path)
    if not adv_path.exists():
        raise NotADirectoryError('Please define a valid path to where adversaries are stored!')

    prep_and_run(adv_path, options.start, options.nr_files, save_path)


if __name__ == '__main__':
    main()
