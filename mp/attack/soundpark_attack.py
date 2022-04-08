import torch
import torch.optim as optim

from pathlib import Path
from argparse import ArgumentParser
from skhubness.analysis import Hubness

from mp.attack.kl_losses import kl_delta_loss
from mp.soundpark.soundpark_model import SoundparkModel
from mp.soundpark.soundpark_data import get_cached_subset_dataloader
from mp.utils.utils import get_single_kl, snr, get_gaussians, get_kl_divs
from mp.utils.io import read_config_file, save_config_file, Logger, save_adversary


def opts_parser():
    descr = 'Performs adversarial attack on Soundpark music recommender.'
    parser = ArgumentParser(description=descr)
    parser.add_argument('--adv_dir', metavar='DIR', type=str,
                        default=Path(__file__).parent.parent.parent / 'adversaries',
                        help='Parent directory to which adversaries should be saved to.')
    parser.add_argument('--cache_file', metavar='DIR', type=str, required=True,
                        help='Path pointing to data cache.')
    parser.add_argument('--save_path', metavar='DIR', type=str,
                        help='Directory where gaussians and kl-divs were stored during preprocessing.')
    parser.add_argument('--adv-vars', metavar='FILE', action='append', type=str,
                        default=[Path(__file__).parent / 'config.vars'],
                        help='Reads configuration variables from a FILE of KEY=VALUE '
                             'lines for adversarial attack.')
    parser.add_argument('--start', default=0, type=int,
                        help='Index of start file.')
    parser.add_argument('--nr_files', type=int, default=1000,
                        help='Total number of files that are attacked.')
    return parser


def prep_paths(init_adv_path, config, save_path):
    """ Prepares paths and files. """
    # prep paths
    adv_path = init_adv_path / config['experiment_name']
    if not adv_path.exists():
        adv_path.mkdir(parents=True)
    # prep gaussian/kl div path
    if not save_path.exists():
        raise NotADirectoryError('Please define a valid path for stored preprocessed files!')
    # save parameters
    save_config_file(adv_path / 'config.vars', config)
    return adv_path


def get_target_hubs(indices, target_hubs, kl_divs, config):
    """ Determines target hubs based on different strategies. """
    if config['choose_hub'] == 'closest':
        hub_indices = torch.argmin(torch.stack([kl_divs[indices, h] for h in target_hubs]), dim=0)
    elif config['choose_hub'] == 'random':
        hub_indices = torch.randint(len(target_hubs), (len(indices),))
    else:
        raise ValueError('Please define valid target-hub-method (closest, biggest or random)!')
    return hub_indices


def init_delta(clean_data):
    """ Initialises delta with zeros. """
    delta = torch.zeros_like(clean_data).to(clean_data.device)
    delta.requires_grad = True
    return delta


def check_for_hubs(new_gaussians, indices, gaussians, kl_divs, not_convs, old_k_occs, hub):
    """ Checks whether adversaries are already recognised as hubs, filter-and-refine approach. """
    # prepare all file-names for KL-div computation
    all_files = list(gaussians.keys())
    new_update_mask = []
    k_occs = []
    for cur in range(len(new_gaussians)):
        if not not_convs[cur]:
            new_update_mask.append(0.), k_occs.append(old_k_occs[cur])
            continue

        idx = indices[cur]
        # compute new KL divergences
        new_kl = torch.stack([get_single_kl(new_gaussians[cur], gaussians[file]) for file in all_files])
        new_kl[idx] = get_single_kl(new_gaussians[cur], new_gaussians[cur])
        # store old KL divergence
        old_kl = kl_divs[idx].clone()
        # insert new divergences in distance matrix
        kl_divs[idx, :] = kl_divs[:, idx] = new_kl
        # compute updated hubs, restore KL divergences
        hub.fit(kl_divs.cpu())
        updated_hub_score = hub.score(has_self_distances=True)
        new_hubs = torch.tensor(updated_hub_score['hubs']).to(indices.device)
        k_occ = updated_hub_score['k_occurrence'][idx]
        kl_divs[idx, :] = kl_divs[:, idx] = old_kl
        del old_kl
        new_update_mask.append(0. if idx in new_hubs else 1.), k_occs.append(k_occ)
    return torch.tensor(new_update_mask), torch.tensor(k_occs)


def compute_adversaries_batch(target_data, indices, middles, gaussians, kl_divs, init_hub_score, sp_model, config, hub):
    """ Compute adversarial perturbation for a batch. """
    cpu_model = SoundparkModel(torch.device('cpu'))
    init_hubs = init_hub_score['hubs']
    update_mask = torch.tensor([1. if idx not in init_hubs else 0. for idx in indices]).to(middles.device)
    k_occs = init_hub_score['k_occurrence'][indices]
    if torch.sum(update_mask) == 0:
        return [0] * len(indices), middles, k_occs

    # get target hub indices / data
    target_hub_indices = get_target_hubs(indices, init_hubs, kl_divs, config)
    target_hub_data = torch.stack([target_data[t.item()][-1] for t in target_hub_indices]).to(middles.device)

    # prepare delta, optimiser and loss function
    delta = init_delta(middles)
    optimiser = optim.Adam([delta], lr=config['lr'])

    for e in range(config['max_epochs']):
        optimiser.zero_grad()
        loss = kl_delta_loss(middles, delta, target_hub_data, sp_model, config['alpha'])
        print('Epoch {}/{}, loss: {}, updates: {}, k_occs: {}\r'.format(e + 1, config['max_epochs'], loss.item(),
                    torch.sum(update_mask).item(), k_occs), flush=True, end='')
        loss.backward()
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad) * update_mask.view(len(delta), 1, 1)
        optimiser.step()
        with torch.no_grad():
            delta.clamp_(min=-config['clip_eps'], max=config['clip_eps'])

        # check whether attack was successful
        if e % 10 == 0:
            update_mask, k_occs = check_for_hubs(cpu_model((middles + delta).detach().cpu()), indices,
                                                 gaussians, kl_divs, update_mask, k_occs, hub)
            update_mask = update_mask.to(middles.device)
        if torch.sum(update_mask) == 0:
            # all adversaries found, return
            return [not u for u in update_mask], delta.detach(), k_occs
    return [not u for u in update_mask], delta.detach(), k_occs


def compute_adversaries(adv_path, cache_path, config, save_path):
    """ Compute adversaries for prepared data. """
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
    sp_model = SoundparkModel(device)
    hub = Hubness(k=5, return_value='all', hub_size=config['hub_size'], metric='precomputed', store_k_occurrence=True)

    # get initial gaussians, KL divergences
    gaussians = get_gaussians(save_path, torch.device('cpu'))
    files = list(gaussians.keys())
    kl_divs = get_kl_divs(save_path, device)

    # prepare data
    first_file = config['start_file'] if config['start_file'] >= 0 else 0
    last_file = first_file + config['nr_files'] if first_file + config['nr_files'] <= len(files) else len(files)
    data = get_cached_subset_dataloader(cache_path, torch.arange(first_file, last_file), 1)
    print('checking file {} up to {} (# {})...'.format(first_file, last_file - 1,
                                                       len(torch.arange(first_file, last_file))))
    # prepare logger
    logger = Logger(str(adv_path / 'log_{}_{}.txt').format(first_file, last_file), columns=['file', 'db', 'conv', 'kocc'])

    # get initial hubs
    hub.fit(kl_divs.cpu())
    init_hub_score = hub.score(has_self_distances=True)
    init_hubs = torch.tensor(init_hub_score['hubs']).to(device)

    if len(init_hubs) == 0:
        raise ValueError('No hubs were found, try changing the hub-size in the configuration!')

    print('Initial / Target hubs: {}'.format(init_hubs))
    target_data = get_cached_subset_dataloader(cache_path, init_hubs, 1).dataset

    for b, (indices, middles) in enumerate(data):
        print('\nBatch {}/{}'.format(b + 1, len(data)))
        indices, middles = indices.to(device), middles.to(device)
        # compute adversaries batch-wise, store it if converged
        convs, perts, k_occs = compute_adversaries_batch(target_data, indices, middles, gaussians, kl_divs,
                                                         init_hub_score, sp_model, config, hub)
        check_convergence(convs, perts.cpu(), k_occs, indices, middles.cpu(), init_hubs, files, adv_path, logger)


def check_convergence(convs, perts, k_occs, indices, middles, init_hubs, files, adv_path, logger):
    """ Checks whether adversary was found / file was hub to begin with, and stores some adversaries. """
    for i in range(len(convs)):
        # check whether we found adversary
        idx = indices[i]
        file_name = Path(files[idx]).name
        if idx in init_hubs:
            print('File was already hub ({})'.format(file_name))
            logger.append([file_name, snr(middles[i], middles[i]).item(), 'hub', k_occs[i].item()])
        elif convs[i]:
            save_adversary((middles[i] + perts[i]).cpu(), adv_path / Path(files[idx]).with_suffix('.wav').name)
            logger.append([file_name, snr(middles[i], middles[i] + perts[i]).item(), 'yes', k_occs[i].item()])
        else:
            print('Could not find adversary for this file ({})'.format(file_name))
            logger.append([file_name, snr(middles[i], middles[i] + perts[i]).item(), 'no', k_occs[i].item()])


def main():
    # parse command line arguments
    parser = opts_parser()
    options = parser.parse_args()
    config = {}
    for fn in options.adv_vars:
        config.update(read_config_file(Path(fn)))
    config.update({'start_file': options.start, 'nr_files': options.nr_files})

    save_path = Path(options.save_path)
    adv_path = prep_paths(Path(options.adv_dir), config, save_path)
    cache_path = Path(options.cache_file)
    # start attack
    compute_adversaries(adv_path, cache_path, config, save_path)


if __name__ == '__main__':
    main()
