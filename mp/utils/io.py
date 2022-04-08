import os
import csv
import torch
import librosa
import warnings

from scipy.io.wavfile import write


def read_config_file(config_file):
    """ Reads and parses configuration file. """
    with open(config_file, 'r') as f:
        return parse_variable_assignments([l.rstrip('\r\n') for l in f
                                           if l.rstrip('\r\n') and not l.startswith('#')])


def parse_variable_assignments(assignments):
    """
    Parses a list of key=value strings and returns a corresponding dictionary.
    Values are tried to be interpreted as float or int, otherwise left as str.
    """
    variables = {}
    for assignment in assignments or ():
        key, value = assignment.replace(' ', '').split('=', 1)
        for convert in (int, float, str):
            try:
                value = convert(value)
            except ValueError:
                continue
            else:
                break
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        variables[key] = value
    return variables


def save_config_file(save_path, config):
    """ Given dict containing configuration, saves to configuration file. """
    with open(save_path, 'w') as fp:
        fp.writelines([' = '.join((str(k), str(config[k]) + '\n')) for k in config.keys()])


def load_file(file_name, sample_rate, device):
    """ Loads file with librosa, resamples and converts it to mono. """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, sr = librosa.load(file_name, sr=sample_rate)
    data = torch.tensor(data)
    return data.to(device).detach()


def save_adversary(perturbation, file_path):
    """ Saves adversarial data. """
    write(str(file_path.with_suffix('.wav')), 22050, perturbation.view(-1, 1).numpy())


class Logger:
    """ Class that allows logging. """
    def __init__(self, log_file_path, columns=None):
        if '~' in log_file_path:
            log_file_path = os.path.expanduser(log_file_path)
        self.log_path = log_file_path
        if columns is None:
            self.columns = ['epoch', 'train loss', 'train accuracy']
        else:
            self.columns = columns

        with open(self.log_path, 'w') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(columns)

    def append(self, value_list):
        with open(self.log_path, 'a') as fp:
            writer = csv.writer(fp, delimiter=',')
            writer.writerow(value_list)
