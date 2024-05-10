import argparse
import hashlib
import json
import sys
from enum import Enum, IntEnum

__package__ = "molcompview"
from os.path import isfile, join

import dash_bootstrap_components as dbc
import base64
import os

from appdata import AppDataPaths

from . import __x_name__, __y_name__, __smiles_name__, __version__, DatasetState
import dash
import pandas as pd
import numpy as np
import logging
from dash import Dash, dcc, html, no_update, Output, Input, MATCH, ALL
import plotly.graph_objects as go
from molcomplib import MolCompass
from molcompview.actions import init_callbacks
from molcompview.components import header, molcompass_layout, header_alt
from molcompview.functions import process_new_file, convert_to_column_type_dict
#Create enums for numerical, categorical and binary columns

app_paths = AppDataPaths()
app_paths.setup()


def _get_column_types(data):
    approp_columns = list(data.select_dtypes(include=['object','float','double','int']).columns)
    logging.info("String columns: {}".format(approp_columns))

    #Select all categorical columns
    categorical_columns = [col for col in approp_columns if 2 < data[col].nunique() < 10]
    logging.info("Categorical columns: {}".format(categorical_columns))
    binary_columns = [col for col in approp_columns if data[col].nunique() == 2]
    logging.info("Binary columns: {}".format(binary_columns))
    #Select all non-categorical columns with numerical values, check if they are double
    numerical_columns = [col for col in data.columns if col in approp_columns and data[col].dtype == float]
    logging.info("Numerical columns: {}".format(numerical_columns))
    return approp_columns, categorical_columns, binary_columns, numerical_columns

def get_column_(data, list_of_column_names,name,raise_error=True,modify_inplace=False):
    column = [x for x in data.columns if x.lower() in list_of_column_names]
    if raise_error:
        assert len(column) == 1, "Dataframe should contain ONLY one {} column, but found: {}".format(name,column)
    else:
        if len(column) == 0:
            return None
    if modify_inplace:
        data.rename(columns={column[0]: name}, inplace=True)
    return column[0]

def get_column(column,data,raise_error=True,modify_inplace=False):
    correspondence = {'smiles':['smiles', 'smiles', 'smiles', 'molecules', 'structures', 'mols', 'smi','canonical_smiles','canonical_smi','canonicalsmiles','canonical smiles'],
     'class':['class', 'classes', 'active','act','target','targets'],
     'train/test split':['train','test','split','train/test','set'],
     'logits':['logits','prob'],
     'x_coord':['x_coord','X_coord'],
     'y_coord':['y_coord','Y_coord']}
    return get_column_(data,correspondence[column],column,raise_error,modify_inplace)

def run_molcomplib(data,smilesColumn):
    compass = MolCompass()
    #Check if there is y column
    if 'y' in data.columns:
        data.rename(columns={'y': 'class'}, inplace=True)
    data = compass.process(data)
    return data


def get_column_prob(data):
    guess = get_column('logits',data,raise_error=False)
    if guess == None:
        logging.info("No logits column found with standard names, trying to guess the probability column")
        #Try to guess the column with probabilities
        #Get the distribution of values in all columns, and min and max
        min_max = data.describe().loc[['min','max']]
        #Get the columns with values between 0 and 1 but not 0 or 1
        prob_columns = [col for col in min_max.columns if (min_max[col]['min'] >= 0 and min_max[col]['max'] <= 1 and min_max[col]['min'] != 0 and min_max[col]['max'] != 1)]
        if len(prob_columns) == 1:
            guess = prob_columns[0]
            logging.info("Guessed column with probabilities: {}".format(guess))
        elif len(prob_columns) > 1:
            logging.info("More than one column with values between 0 and 1 found, please specify the column with probabilities")
            return None
        else:
            logging.info("Could not guess the column with probabilities, please specify it manually")
    return guess

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process a CSV file with MolCompass Viewer.")
    parser.add_argument('file', type=str, help='The path to the CSV file to process')
    parser.add_argument('--precompute', action='store_true', help='Whether to precompute certain data (default: False)')
    parser.add_argument('--log_level', type=str, default='ERROR', help='Logging level (default: ERROR)')
    # Parse the arguments
    args = parser.parse_args()
    # Now call your existing main logic with these arguments
    main_logic(args.file, precompute=args.precompute, log_level=args.log_level)

def main_logic(file,precompute=False,log_level="ERROR"):
    def make_dropdown(useful_columns):
        return dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in useful_columns],
            value=useful_columns[0]
        ) if len(useful_columns) > 0 else html.Div("No columns for coloring found")

    if not isfile(file):
        raise FileNotFoundError("File {} not found, please specify a csv file".format(file))

    logging.basicConfig(level=log_level)
    logging.info("Starting MolCompass Viewer")
    #Calculate MD5 hash of file
    hash = hashlib.md5(open(file, 'rb').read()).hexdigest()
    processed_file_name = join(app_paths.app_data_path, hash + ".csv")
    if not isfile(processed_file_name):
        logging.info("File {} not found in cache, processing it".format(processed_file_name))
        data,dataset_state,column_types = process_new_file(file)
        first = json.dumps({'state': dataset_state.value, 'column_types': column_types, 'version': __version__})
        data.to_csv(processed_file_name, index=False)
        # Add the fist row to the file with the dataset state and the column types
        with open(processed_file_name, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(first.rstrip('\r\n') + '\n' + content)
    else:
        logging.info("File {} found in cache, loading it".format(processed_file_name))
        with open(processed_file_name, 'r') as f:
            first = f.readline()
            dataset_state = DatasetState(int(json.loads(first)['state']))
            column_types = json.loads(first)['column_types']
            data = pd.read_csv(processed_file_name,skiprows=1)
    column_types = convert_to_column_type_dict(column_types)
    useful_columns = [col for col in data.columns if col not in [__x_name__,__y_name__,__smiles_name__]]
    logging.info("Useful columns: {}".format(useful_columns))
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(header_alt(), className='g-0')]),  # Header
        dbc.Row([dbc.Col(molcompass_layout(useful_columns), className='g-0')],id='main-layout'),  # Main
    ], fluid=True
    )
    init_callbacks(app,data,column_types,dataset_state)
    app.run_server(debug=True)


#
# def main():
#       fire.Fire(show)

if (__name__ == '__main__'): #For debug only
     main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
