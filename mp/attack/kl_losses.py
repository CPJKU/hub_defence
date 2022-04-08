import torch

from mp.utils.utils import get_single_kl


def kl_delta_loss(source, delta, target, model, alpha):
    """ Computes KL distance and tries to minimise norm of delta for optimisation. """
    source_gaussian = model(source + delta)
    target_gaussian = model(target)

    if not (len(source_gaussian) == len(target_gaussian) == len(delta)):
        raise ValueError('shapes should be equal!')

    loss = torch.stack([get_single_kl(s, t) * alpha + torch.sum(d ** 2)
                        for s, t, d in zip(source_gaussian, target_gaussian, delta)])
    return torch.mean(loss)
