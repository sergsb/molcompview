__version__ = '0.1.0'
__smiles_name__ = 'smiles_processed_mcv'
__class_name__ = 'class_processed_mcv'
__loss_name__ = 'loss_processed_mcv'
__probs_name__ = 'probs_processed_mcv'
__set_name__ = 'set_processed_mcv'
__x_name__ = 'x_processed_mcv'
__y_name__ = 'y_processed_mcv'

from enum import IntEnum


class DatasetState(IntEnum):
    """Enum for dataset stages."""
    STRUCTURES_ONLY = 0
    ALTERNATIVE_MODE = 1
    NORMAL = 2
