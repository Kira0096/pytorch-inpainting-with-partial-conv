import torch
from torch.utils.data.sampler import Sampler

class DataSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
    """

    def __init__(self, dataset, round_up=True):
        self.dataset = dataset
        self.round_up = round_up
        self.epoch = 0
        
        self.num_samples = len(self.dataset)

        self.total_size = len(self.dataset)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
