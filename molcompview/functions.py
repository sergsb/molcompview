import hashlib
import json
import os
from enum import IntEnum
from os.path import isfile, join
import logging
import pandas as pd
from molcomplib import MolCompass
from . import __smiles_name__, __class_name__, __set_name__, __probs_name__, __loss_name__, __x_name__, __y_name__, \
    DatasetState
import numpy as np

from .actions import ColumnType

#Make loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) #FixMe: Set up based on command line argument





COLUMN_CORRESPONDENCE = {
    __smiles_name__: [
        'smiles', 'molecules', 'structures', 'mols', 'smi',
        'canonical_smiles', 'canonical_smi', 'canonicalsmiles',
        'canonical smiles', 'mol_strings', 'structure_strings'
    ],
    __class_name__: [
        'class', 'classes', 'active', 'act', 'target', 'targets',
        'y', 'activity', 'activities', 'label', 'labels', 'response',
        'responses', 'outcome'
    ],
    __set_name__: [
        'train', 'test', 'split', 'train/test', 'set', 'dataset_split',
        'data_division', 'partition', 'role'
    ],
    __probs_name__: [
        'logits', 'prob','probs', 'probabilities','probability', 'scores', 'confidence',
        'score', 'likelihood', 'chance', 'estimates'
    ]
}

def get_column_name(data, key, raise_error=True, modify_inplace=False):
    """Retrieve appropriate column name based on common aliases."""
    columns = [x for x in data.columns if x.lower() in COLUMN_CORRESPONDENCE[key]]
    if raise_error:
        assert len(columns) == 1, f"Dataframe should contain ONLY one {key} column, but found: {columns}"
    else:
        if not columns:
            return None
    if modify_inplace:
        data.rename(columns={columns[0]: key}, inplace=True)
    return columns[0]


def identify_column_types(data):
    """Classify columns into appropriate types (categorical, binary, numerical)."""
    appropriate_columns = list(data.select_dtypes(include=['object', 'float', 'double', 'int']).columns)
    logger.info(f"All columns: {appropriate_columns}")

    # Categorize columns
    categorical = [col for col in appropriate_columns if 2 < data[col].nunique() < 10]
    binary = [col for col in appropriate_columns if data[col].nunique() == 2]
    numerical = [col for col in appropriate_columns if data[col].dtype == float]

    logger.info(f"Categorical columns: {categorical}")
    logger.info(f"Binary columns: {binary}")
    logger.info(f"Numerical columns: {numerical}")

    return {'categorical': categorical, 'binary': binary, 'numerical': numerical}


def guess_probability_column(data):
    """Attempt to identify a probability column based on value range."""
    min_max = data.describe().loc[['min', 'max']]
    prob_columns = [col for col in min_max.columns if (0 <= min_max[col]['min'] < 1 and 0 < min_max[col]['max'] <= 1)]
    if len(prob_columns) == 1:
        logger.info(f"Guessed column with probabilities: {prob_columns[0]}")
        return prob_columns[0]
    elif len(prob_columns) > 1:
        logger.info("Multiple columns found with values between 0 and 1. Specify the column with probabilities.")
        return None
    else:
        logger.info("Couldn't identify the column with probabilities. Specify manually.")
        return None


def convert_to_column_type_dict(initial_dict):
    key_to_enum = {
        'binary': ColumnType.BINARY,
        'categorical': ColumnType.CATEGORICAL,
        'numerical': ColumnType.NUMERICAL
    }
    return {value: key_to_enum[key] for key, values in initial_dict.items() for value in values}

def process_new_file(filename):
    """Process and classify columns of a new data file."""
    data = pd.read_csv(filename, sep=',', low_memory=False)
    smiles_col = get_column_name(data, __smiles_name__, raise_error=False)
    class_col = get_column_name(data, __class_name__, raise_error=False)
    split_col = get_column_name(data, __set_name__, raise_error=False)
    prob_col = get_column_name(data, __probs_name__, raise_error=False) or guess_probability_column(data)

    column_mapping = {}
    if smiles_col: column_mapping[smiles_col] = __smiles_name__
    if class_col: column_mapping[class_col] = __class_name__
    if split_col: column_mapping[split_col] = __set_name__
    if prob_col: column_mapping[prob_col] = __probs_name__

    data.rename(columns=column_mapping, inplace=True)

    column_types = identify_column_types(data)
    if __class_name__ not in data.columns and __probs_name__ not in data.columns:
        dataset_state = DatasetState.STRUCTURES_ONLY
    if __class_name__ in data.columns and __probs_name__ not in data.columns:
        dataset_state = DatasetState.ALTERNATIVE_MODE
    if __probs_name__ in data.columns and __class_name__ not in data.columns:
        dataset_state = DatasetState.ALTERNATIVE_MODE
    if __probs_name__ in data.columns and __class_name__ in data.columns and __smiles_name__ in data.columns:
        dataset_state = DatasetState.NORMAL
        data[__loss_name__] = -(data[__class_name__]*np.log(data[__probs_name__]) + (1-data[__class_name__])*np.log(1-data[__probs_name__]))
        column_types['numerical'].append(__loss_name__)

    #Report Mode in Main log
    mode_txt = {0:"STRUCTURES_ONLY",1:"ALTERNATIVE_MODE",2:"NORMAL"}
    print(f"Molcompass is running in {mode_txt[dataset_state]} mode")
    #Run molcomplib
    compass = MolCompass()
    #report log
    logger.info(f"Processing data with MolCompass... (this may take a while)")
    data = compass.process(data)
    logger.info("End of processing data with MolCompass, coordinates are added to the dataframe")
    #Rename columns
    data.rename(columns={'x': __x_name__, 'y': __y_name__}, inplace=True)
    # else:
    #     # Other scenarios can be handled here, or you can raise an exception if the dataset state is not recognized
    #     raise Exception("Internal error. The dataset state could not be determined based on the columns present.")
    return data, dataset_state, column_types

# Other functions like `run_molcomplib` remain unchanged
