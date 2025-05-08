"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from ppl.torch.networks.basic import (
    Clamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from ppl.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy
from ppl.torch.networks.dcnn import DCNN, TwoHeadDCNN
from ppl.torch.networks.feat_point_mlp import FeatPointMlp
from ppl.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from ppl.torch.networks.linear_transform import LinearTransform
from ppl.torch.networks.normalization import LayerNorm
from ppl.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)
from ppl.torch.networks.pretrained_cnn import PretrainedCNN
from ppl.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Clamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LayerNorm',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

