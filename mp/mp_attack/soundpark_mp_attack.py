import torch
import torch.optim as optim

from pathlib import Path
from argparse import ArgumentParser
from skhubness.analysis import Hubness
from skhubness.reduction import MutualProximity

from mp.utils.io import read_config_file, Logger
from mp.soundpark.soundpark_model import SoundparkModel
from mp.mp_attack.mp_losses import mp_delta_loss, kl_delta_loss
from mp.soundpark.soundpark_data import get_cached_subset_dataloader
from mp.utils.utils import get_single_kl, get_row_mp, get_gaussians, get_kl_divs, get_mp_dists
from mp.attack.soundpark_attack import prep_paths, get_target_hubs, init_delta, check_convergence


def opts_parser():
    descr = 'Performs adversarial attack on soundpark music recommender w/ mutual proximity.'
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


def check_for_hubs(new_gaussians, indices, gaussians, kl_divs, mp_dists, not_convs, old_k_occs, hub, mp):
    """ Checks whether adversaries are already considered hubs, despite MP (KL-attack). """
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
        # store old KL divergence
        old_kl = kl_divs[idx].clone()
        # insert new divergences in distance matrix
        new_kl[idx] = get_single_kl(new_gaussians[cur], new_gaussians[cur])
        kl_divs[idx, :] = kl_divs[:, idx] = new_kl

        # compute new mp
        new_mp = get_row_mp(kl_divs, idx)
        # copy old mp values for now
        old_mp = mp_dists[idx].clone()
        # insert new mp values
        mp_dists[idx, :] = mp_dists[:, idx] = new_mp
        # compute new hubs
        hub.fit(mp_dists.cpu())
        updated_hub_score = hub.score(has_self_distances=True)
        new_hubs = torch.tensor(updated_hub_score['hubs']).to(indices.device)
        k_occ = updated_hub_score['k_occurrence'][idx]

        # if successful according to approx, compute entire MP to make sure
        if idx in new_hubs:
            import numpy as np
            neigh_ind = np.array([list(range(0, kl_divs.shape[1])) for _ in range(kl_divs.shape[0])])
            mp.fit(kl_divs.cpu().numpy(), neigh_ind)
            tot_mp, _ = mp.transform(kl_divs.cpu().numpy(), neigh_ind)

            # get new hubs with complete MP matrix
            hub.fit(tot_mp)
            updated_hub_score = hub.score(has_self_distances=True)
            new_hubs = torch.tensor(updated_hub_score['hubs']).to(indices.device)
            k_occ = updated_hub_score['k_occurrence'][idx]
            if idx not in new_hubs:
                print('Before MP recomputation hub, now not anymore - continue computations...')

        # restore mp
        mp_dists[idx, :] = mp_dists[:, idx] = old_mp
        del old_mp
        # restore KL divergences
        kl_divs[idx, :] = kl_divs[:, idx] = old_kl
        del old_kl

        new_update_mask.append(0. if idx in new_hubs else 1.), k_occs.append(k_occ)
    return torch.tensor(new_update_mask), torch.tensor(k_occs)


def do_update(optimiser, middles, delta, target_hub_data, sp_model, config, gaussians, kl_divs, indices,
              update_mask, k_occs, cpu_model, mp_dists, hub, mp, e, target_indices=None):
    """ Does update on perturbation delta and the check whether attack was successful. """
    optimiser.zero_grad()
    if target_indices:
        loss = mp_delta_loss(middles, delta, target_hub_data, sp_model, config['alpha'], gaussians,
                             indices, target_indices)
    else:
        loss = kl_delta_loss(middles, delta, target_hub_data, sp_model, config['alpha'])
    print('Epoch {}/{}, loss: {}, updates: {}, k_occs: {}\r'.format(e + 1, config['max_epochs'], loss.item(),
                                                                    torch.sum(update_mask).item(), k_occs),
          flush=True, end='')
    loss.backward()
    with torch.no_grad():
        delta.grad = torch.sign(delta.grad) * update_mask.view(len(delta), 1, 1)
    optimiser.step()
    with torch.no_grad():
        delta.clamp_(min=-config['clip_eps'], max=config['clip_eps'])

    # check whether attack was successful
    if e % 10 == 0:
        update_mask, k_occs = check_for_hubs(cpu_model((middles + delta).detach().cpu()), indices,
                                             gaussians, kl_divs, mp_dists, update_mask, k_occs, hub, mp)
        update_mask = update_mask.to(middles.device)

    return delta, update_mask, k_occs


def compute_adversaries_batch(target_data, indices, middles, gaussians, mp_dists, kl_divs,
                              init_hub_score, sp_model, config, hub, mp):
    """ Compute MP adversarial perturbation for a batch. """
    cpu_model = SoundparkModel(torch.device('cpu'))
    init_hubs = init_hub_score['hubs']
    update_mask = torch.tensor([1. if idx not in init_hubs else 0. for idx in indices]).to(middles.device)
    k_occs = init_hub_score['k_occurrence'][indices]
    if torch.sum(update_mask) == 0:
        return [0] * len(indices), middles, k_occs

    # get target hub indices / data
    target_hub_indices = get_target_hubs(indices, init_hubs, mp_dists, config)
    target_hub_data = torch.stack([target_data[t.item()][-1] for t in target_hub_indices]).to(middles.device)

    # prepare delta, optimiser and loss function
    delta = init_delta(middles)
    optimiser = optim.Adam([delta], lr=config['lr'])

    for e in range(config['max_epochs']):
        target_indices = None if config['loss_function'] == 'kl_delta_loss' else [init_hubs[target_hub_indices]]
        delta, update_mask, k_occs = do_update(optimiser, middles, delta, target_hub_data, sp_model,
                                               config, gaussians, kl_divs, indices, update_mask, k_occs,
                                               cpu_model, mp_dists, hub, mp, e, target_indices)

        if torch.sum(update_mask) == 0:
            # all adversaries found, return
            return [not u for u in update_mask], delta.detach(), k_occs
    return [not u for u in update_mask], delta.detach(), k_occs


def compute_mp_adversaries(adv_path, cache_path, config, save_path):
    """ Compute adversaries for prepared data; we assume that the recommender here is defended with MP. """
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')
    sp_model = SoundparkModel(device)
    hub = Hubness(k=5, return_value='all', hub_size=config['hub_size'], metric='precomputed', store_k_occurrence=True)
    mp = MutualProximity(method='empiric', verbose=True)

    # get initial gaussians, KL divergences, MP distances
    gaussians = get_gaussians(save_path, torch.device('cpu'))
    files = list(gaussians.keys())
    kl_divs = get_kl_divs(save_path, device)
    mp_dists = get_mp_dists(save_path, device)

    # prepare data
    first_file = config['start_file'] if config['start_file'] >= 0 else 0
    last_file = first_file + config['nr_files'] if first_file + config['nr_files'] <= len(files) else len(files)
    data = get_cached_subset_dataloader(cache_path, torch.arange(first_file, last_file), 1)
    # prepare logger
    logger = Logger(str(adv_path / 'log_{}_{}.txt').format(first_file, last_file), columns=['file', 'db', 'conv', 'kocc'])

    # get initial hubs
    hub.fit(mp_dists.cpu())
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
        convs, perts, k_occs = compute_adversaries_batch(target_data, indices, middles, gaussians, mp_dists,
                                                         kl_divs, init_hub_score, sp_model, config, hub, mp)
        check_convergence(convs, perts.cpu(), k_occs, indices, middles.cpu(), init_hubs, files, adv_path, logger)


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
    compute_mp_adversaries(adv_path, cache_path, config, save_path)


if __name__ == '__main__':
    main()
