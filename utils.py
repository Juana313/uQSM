import importlib
import logging
import os
import shutil
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
import glob
from random import shuffle
import h5py
import pickle


plt.ioff()
plt.switch_backend('agg')


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


def save_network_output(output_path, output, logger=None):
    if logger is not None:
        logger.info(f'Saving network output to: {output_path}...')
    output = output.detach().cpu()[0]
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def create_sample_plotter(sample_plotter_config):
    if sample_plotter_config is None:
        return None
    class_name = sample_plotter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**sample_plotter_config)

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

def fetch_data_files(datasets_path, modalities, subdir_filter="*"):
    training_data_files = list()

    for subject_dir in glob.glob(os.path.join(datasets_path, subdir_filter)):
        subject_files = list()
        files_existing = True

        for modality in modalities:
            if os.path.exists(os.path.join(subject_dir, modality)):
                subject_files.append(os.path.join(subject_dir, modality))
            else:
                files_existing = False

        if files_existing:
            training_data_files.append(subject_files)
    return training_data_files

def fetch_data_filesx(datasets_path, modalities, subdir_filters=["*"]):
    training_data_files = list()

    for subdir_filter in subdir_filters:
        for subject_dir in glob.glob(os.path.join(datasets_path, subdir_filter)):
            subject_files = list()
            files_existing = True
    
            for modality in modalities:
                if os.path.exists(os.path.join(subject_dir, modality)):
                    subject_files.append(os.path.join(subject_dir, modality))
                else:
                    files_existing = False
    
            if files_existing:
                training_data_files.append(subject_files)
    return training_data_files

def get_validation_split(data_files, training_file, validation_file, data_split=0.8, overwrite=False):
    def split_list(input_list, split=0.8, shuffle_list=True):
        if shuffle_list:
            shuffle(input_list)
        n_training = int(len(input_list) * split)
        training = input_list[:n_training]
        testing = input_list[n_training:]
        return training, testing
    
    if overwrite or not os.path.exists(training_file) or not os.path.exists(validation_file):
        print("Creating validation split...")
        nb_samples = len(data_files)
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split,shuffle_list=True)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)

