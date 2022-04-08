import h5py
import torch
import librosa

from torch.utils.data import Dataset, DataLoader, Subset


def get_raw_subset_dataloader(files, indices, batch_size=1):
    """ Returns dataloader of subset of soundpark files. """
    ds = RawSoundparkSet(files)
    sub_ds = Subset(ds, indices)
    return DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=1)


def get_raw_soundpark_loader(files, batch_size):
    """ Returns DataLoader of Soundpark data. """
    ds = RawSoundparkSet(files)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)


class RawSoundparkSet(Dataset):
    """ Dataset loading Soundpark data. """
    def __init__(self, files):
        # define constants
        self.sr = 22050
        self.nr_samples = librosa.core.time_to_samples(60, sr=self.sr)

        # store files and verbosity
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        import warnings

        # load data
        cur_file = self.files[index]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sr = librosa.load(cur_file, sr=self.sr)
        data = torch.tensor(data)

        # get middle
        data_middle = data.shape[0] // 2
        data = data[data_middle - self.nr_samples:data_middle + self.nr_samples]

        return index, data.view(1, -1)


def get_cached_subset_dataloader(cache_path, indices, batch_size=1):
    """ Returns dataloader of subset of pre-saved soundpark files. """
    ds = CachedSoundparkSet(cache_path)
    sub_ds = Subset(ds, indices)
    return DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=1)


def get_cached_soundpark_dataloader(cache_path, batch_size=1):
    """ Returns dataloader of pre-saved soundpark files. """
    ds = CachedSoundparkSet(cache_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1)


class CachedSoundparkSet(Dataset):
    """ Dataset that returns loaded soundpark files from h5py dataset. """
    def __init__(self, cache_path):
        self.sr = 22050
        self.cache_path = cache_path

    def __len__(self):
        with h5py.File(self.cache_path, 'r') as hf:
            return len(hf)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        with h5py.File(self.cache_path, 'r') as hf:
            data = hf.get(str(index))
            return index, torch.tensor(data)
