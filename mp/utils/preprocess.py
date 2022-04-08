import h5py
import torch
import numpy as np

from pathlib import Path
from argparse import ArgumentParser
from mp.utils.utils import get_single_kl
from skhubness.reduction import MutualProximity
from mp.soundpark.soundpark_model import SoundparkModel
from mp.soundpark.soundpark_data import get_raw_soundpark_loader


def save_dataset(cache_path, data_loader):
    """ Saves all data to h5py-dataset located at defined path. """
    hf = h5py.File(cache_path, 'w')

    for b, (indices, middles) in enumerate(data_loader):
        print('File {}/{} is being stored...\r'.format(b + 1, len(data_loader)), flush=True, end='')
        hf.create_dataset(str(indices.item()), data=middles)

    hf.close()
    print('Successfully stored dataset!')


def save_gaussians(save_path, data_loader):
    """ Saves Gaussians for all files. """
    save_path = save_path / 'gaussians.pt'
    sp_model = SoundparkModel(torch.device('cpu'))
    gaussians = {}
    files = data_loader.dataset.files
    for b, (indices, middles) in enumerate(data_loader):
        print('Gaussian {}/{} is computed...\r'.format(b + 1, len(data_loader)), flush=True, end='')
        res = sp_model(middles)
        gaussians.update({files[i]: m for i, m in zip(indices, res)})
    torch.save({str(k): gaussians[k] for k in gaussians.keys()}, save_path)
    return gaussians


def save_kl_divs(save_path, gaussians):
    """ Saves KL divergences for all files. """
    save_path = save_path / 'kl.npz'
    file_names = list(gaussians.keys())
    kl_divs = torch.zeros(len(gaussians), len(gaussians))
    for i, f in enumerate(file_names):
        print('Computing KL-divergences for file {}/{}\r'.format(i + 1, len(file_names)), flush=True, end='')
        for j in range(i, len(file_names)):
            kl_divs[i, j] = kl_divs[j, i] = get_single_kl(gaussians[f], gaussians[file_names[j]])
    np.savez(save_path, kl_divs)
    return kl_divs


def save_mp_dists(save_path, kl_divs):
    """ Saves MP-distances for all files. """
    save_path = save_path / 'mp.npz'
    mp = MutualProximity(method='empiric', verbose=True)
    neigh_ind = np.array([list(range(0, kl_divs.shape[1])) for _ in range(kl_divs.shape[0])])
    mp.fit(kl_divs.cpu().numpy(), neigh_ind)
    mp_dists, _ = mp.transform(kl_divs.cpu().numpy(), neigh_ind)
    np.savez(save_path, mp_dists)


def main():
    parser = ArgumentParser(description='Script that caches songs with h5py.')
    parser.add_argument('--data_path', metavar='DIR', type=str, required=True,
                        help='Path pointing to data that needs to be cached.')
    parser.add_argument('--cache_file', metavar='FILE', type=str, required=True,
                        help='File pointing to where cache should be saved.')
    parser.add_argument('--save_path', metavar='DIR', type=str, required=True,
                        help='Path pointing to the directory where Gaussians etc. are stored.')
    options = parser.parse_args()

    data_path = Path(options.data_path)
    cache_file = Path(options.cache_file).with_suffix('.hdf5')
    save_path = Path(options.save_path)
    if not save_path.exists():
        save_path.mkdir()

    # get file names
    clean_files = [str(f) for f in sorted(list(data_path.rglob('*.mp3')))]

    # get data (in loader)
    data_loader = get_raw_soundpark_loader(clean_files, 1)
    # save all files to h5py dataset
    save_dataset(cache_file, data_loader)
    # precompute and save Gaussians, KL-divergences, Mututal Proximity
    gaussians = save_gaussians(save_path, data_loader)
    kl_divs = save_kl_divs(save_path, gaussians)
    save_mp_dists(save_path, kl_divs)


if __name__ == '__main__':
    main()
