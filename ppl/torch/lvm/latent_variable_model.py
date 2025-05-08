import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import ppl.torch.pytorch_util as ptu
from ppl.policies.base import ExplorationPolicy
from ppl.torch.core import torch_ify, elem_or_tuple_to_numpy
from ppl.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from ppl.torch.networks import Mlp, CNN
from ppl.torch.networks.basic import MultiInputSequential
from ppl.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from ppl.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)


class LatentVariableModel(nn.Module):
    def __init__(
            self,
            encoder,
            decoder,
            **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
