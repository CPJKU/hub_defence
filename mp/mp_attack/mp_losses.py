import torch
import numpy as np

from mp.attack.kl_losses import kl_delta_loss
from mp.utils.utils import get_single_mp, get_vec_kl_manual


def mp_delta_loss(source, delta, target, model, alpha, gaussians, indices, target_indices):
    """ Computes MP distance and tries to minimise norm of delta for optimisation. """
    source_gaussian = model(source + delta)
    target_gaussian = model(target)

    if not (len(source_gaussian) == len(target_gaussian) == len(delta) == len(indices) == len(target_indices)):
        raise ValueError('shapes should be equal!')

    loss = []
    all_files = list(gaussians.keys())
    all_gaussians = [gaussians[f] for f in all_files]
    for i, j, d, g, t in zip(indices, target_indices, delta, source_gaussian, target_gaussian):
        # compute new KL divergences
        kl_divs_i = get_vec_kl_manual(g, all_gaussians)
        kl_divs_i[i] = get_vec_kl_manual(g, [g])
        kl_divs_j = get_vec_kl_manual(t, all_gaussians)
        kl_divs_j[i] = get_vec_kl_manual(g, [t])

        # compute actual loss, append to loss tensor
        if alpha == np.inf:
            loss.append(get_single_mp(kl_divs_i, kl_divs_j, j))
        else:
            loss.append(get_single_mp(kl_divs_i, kl_divs_j, j) * alpha + torch.sum(d ** 2))

    return torch.mean(torch.stack(loss))
