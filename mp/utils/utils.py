import torch
import numpy as np

from pathlib import Path
from torch.distributions.kl import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import _batch_trace_XXT, _batch_mahalanobis


def get_gaussians(save_path, device):
    """ Loads pre-computed Gaussians. """
    save_path = save_path / 'gaussians.pt'
    str_dict = torch.load(save_path, map_location=device)
    gaussians = {Path(k): str_dict[k] for k in str_dict.keys()}
    return gaussians


def get_kl_divs(save_path, device):
    """ Loads pre-computed KL-divergences. """
    save_path = save_path / 'kl.npz'
    kl_divs = np.load(save_path)['arr_0']
    kl_divs = torch.tensor(kl_divs).to(device)
    return kl_divs


def get_mp_dists(save_path, device):
    """ Loads pre-computed MP-distances. """
    save_path = save_path / 'mp.npz'
    mp_dists = np.load(save_path)['arr_0']
    mp_dists = torch.tensor(mp_dists).to(device)
    return mp_dists


def get_single_kl_torch(p, q):
    """ Computes KL distance between two Gaussians. """
    # first check whether devices match
    if p.loc.device != q.loc.device:
        new_device = p.loc.device
        new_q = MultivariateNormal(q.loc.to(new_device), q.covariance_matrix.to(new_device))
        q = new_q
    return kl_divergence(p, q)


def get_single_kl(p, q):
    """ Computes symmetric KL. """
    return (get_single_kl_torch(p, q) + get_single_kl_torch(q, p)) / 2.


def get_vec_half_kl_manual(plocs, qlocs, pscales, qscales):
    """ Vectorised version of KL computation. """
    half_term1 = (qscales.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                  pscales.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    combined_batch_shape = torch._C._infer_size(qscales.shape[:-2],
                                                pscales.shape[:-2])
    n = len(plocs[0])
    q_scale_tril = qscales.expand(combined_batch_shape + (n, n))
    p_scale_tril = pscales.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])
    term3 = _batch_mahalanobis(qscales, (qlocs - plocs))
    return half_term1 + 0.5 * (term2 + term3 - n)


def get_vec_kl_manual(p, qs):
    """ Vectorised version of symmetric KL computation. """
    ps = [p for _ in range(len(qs))]
    plocs = torch.stack([p.loc for p in ps]).to(p.loc.device)
    qlocs = torch.stack([q.loc for q in qs]).to(p.loc.device)
    pscales = torch.stack([p._unbroadcasted_scale_tril for p in ps]).to(p.loc.device)
    qscales = torch.stack([q._unbroadcasted_scale_tril for q in qs]).to(p.loc.device)
    return (get_vec_half_kl_manual(plocs, qlocs, pscales, qscales)
            + get_vec_half_kl_manual(qlocs, plocs, qscales, pscales)) / 2.


def snr(x, x_hat):
    """ SNR computation according to https://github.com/coreyker/dnn-mgr/blob/master/utils/comp_ave_snr.py. """
    ign = 2048
    lng = min(x.shape[-1], x_hat.shape[-1])
    ratio = 20 * np.log10(np.linalg.norm(x[..., ign:lng - ign - 1]) /
                          np.linalg.norm(np.abs(x[..., ign:lng - ign - 1] - x_hat[..., ign:lng - ign - 1]) + 1e-12))
    return ratio


def get_row_mp(kl_divs, i):
    """ Computes MP for single row/column i. """
    neigh_ind = torch.tensor([list(range(0, kl_divs.shape[1])) for _ in range(kl_divs.shape[0])])

    n_test, n_indexed = kl_divs.shape
    hub_reduced_dist = torch.empty_like(kl_divs[i])
    range_n_indexed = range(n_indexed)

    max_ind = neigh_ind.max().item()
    dI = kl_divs[i, :].unsqueeze(0)
    dJ = torch.zeros(dI.shape[-1], n_indexed).to(kl_divs.device)
    for j in range_n_indexed:
        tmp = torch.zeros(max_ind + 1).to(kl_divs.device) + (kl_divs[neigh_ind[i, j], -1] + 1e-6)
        tmp[neigh_ind[neigh_ind[i, j]]] = kl_divs[neigh_ind[i, j]]
        dJ[j, :] = tmp[neigh_ind[i]]
    d = dI.T
    hub_reduced_dist[:] = 1. - (torch.sum((dI > d) & (dJ > d), dim=1) / float(n_indexed))

    return hub_reduced_dist


def get_single_mp(kl_divs_i, kl_divs_j, j):
    """ Given KL divergences of two elements, returns normalised MP value for them. This is a differentiable approx! """
    n_indexed = len(kl_divs_i)

    ddI = torch.max(torch.tanh(kl_divs_i - kl_divs_i[j]), torch.zeros_like(kl_divs_i))
    ddJ = torch.max(torch.tanh(kl_divs_j - kl_divs_i[j]), torch.zeros_like(kl_divs_j))
    hub_reduced_dist = 1. - (torch.sum(ddI * ddJ) / float(n_indexed))

    return hub_reduced_dist
