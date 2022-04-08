import torch
import torchaudio

from torch.distributions.multivariate_normal import MultivariateNormal


class SoundparkModel:
    """ Soundpark model, which computes MFCC and then Gaussian for a raw audio. """
    def __init__(self, device):
        self.mfcc_transf = torchaudio.transforms.MFCC(sample_rate=22050, n_mfcc=20,
                                                      melkwargs={'hop_length': 512, 'n_fft': 1024, 'n_mels': 36})
        self.mfcc_transf.to(device)

    def __call__(self, x):
        # first compute feature (mfcc)
        x = x.view(-1, 2646000)
        feature = self.mfcc_transf(x)

        # compute mean, and covariance
        avg = torch.mean(feature, -1, keepdim=True)
        fact = feature.shape[-1] - 1
        xm = feature - avg

        c = torch.matmul(xm, xm.transpose(-1, -2)) / fact

        # make sure it's positive semi-definite
        eig_val, eig_vec = torch.symeig(c, True)
        new_eig_val = torch.abs(eig_val)
        new_eig_val[new_eig_val <= 1e-04] = 1e-04
        covariance = torch.matmul(eig_vec * new_eig_val.unsqueeze(1), eig_vec.transpose(-1, -2))

        # create distribution
        res = tuple(MultivariateNormal(av.squeeze(), co.squeeze()) for av, co in zip(avg, covariance))
        return res
