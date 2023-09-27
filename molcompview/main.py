from enum import Enum
from os.path import isfile

import dash_bootstrap_components as dbc
import base64
import os

import dash
import fire
import pandas as pd
import numpy as np
import logging
from dash import Dash, dcc, html, no_update, Output, Input, MATCH, ALL
import plotly.graph_objects as go
from molcomplib import MolCompass

from molcompview.actions import init_callbacks, ColumnType
from molcompview.components import header, molcompass_layout, header_alt


#Create enums for numerical, categorical and binary columns



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


def entry_point():
    fire.Fire(main)

def main(file,precompute=False,log_level="ERROR"):
    def make_dropdown(useful_columns):
        return dcc.Dropdown(
            id='dropdown',
            options=[{'label': i, 'value': i} for i in useful_columns],
            value=useful_columns[0]
        ) if len(useful_columns) > 0 else html.Div("No columns for coloring found")

    if not isfile(file):
        raise FileNotFoundError("File {} not found, please specify a csv file".format(file))

    logging.basicConfig(level=log_level)
    data = pd.read_csv(file)

    #Drop rows with NaN values in smiles x_coord or y_coord
    # data.dropna(subset=['smiles','x_coord','y_coord'],inplace=True)
    get_column('smiles',data,raise_error=True,modify_inplace=True)

    #Check, if there is a column with smiles
    if 'smiles' not in data.columns:
        raise ValueError("Dataframe should contain a column with smiles! Guess was unsuccessful. Please specify it with --smilesColumn")

    prob_column = get_column_prob(data)
    if prob_column != None:
        data.rename(columns={prob_column: 'logits'}, inplace=True)
        # get_column('class',data,raise_error=False,modify_inplace=True)
    try:
        x_col = get_column('x_coord',data,raise_error=True,modify_inplace=True)
        y_col = get_column('y_coord',data,raise_error=True,modify_inplace=True)
    except:
        logging.info("No x/y columns found, will calculate them by molcomplib")
        data = run_molcomplib(data,"smiles")
        data.rename(columns={'x': 'x_coord', 'y': 'y_coord'}, inplace=True)
    #If we have both class and logits columns, we can calculate loss
    if 'class' not in data.columns:
        logging.error("No ground truth column found, MolCompass is running in limited mode. AD domain analysis will not be available")
    if 'class' in data.columns and 'logits' in data.columns:
        logging.error("We have both class and logits columns, calculating loss")
        data['loss'] = -data['class']*np.log(data['logits'])-(1-data['class'])*np.log(1-data['logits'])
    elif 'logits' in data.columns:
        logging.error("We have logits column, but no class column, calculating loss is not possible, AD domain analysis will not be available")
    else:
        logging.error("No logits column found, calculating loss is not possible, AD domain analysis will not be available")

    #Rename the columns x and y to x_coord and y_coord

    string_columns, categorical_columns, binary_columns, numerical_columns = _get_column_types(data)
    logging.info("String columns: {}".format(string_columns))
    logging.info("Categorical columns: {}".format(categorical_columns))
    logging.info("Binary columns: {}".format(binary_columns))
    logging.info("Numerical columns: {}".format(numerical_columns))
    #Convert to dictionary, where key is column name and value is the type of column
    column_types = {c: ColumnType.CATEGORICAL for c in data.columns if c in categorical_columns}
    column_types.update({c: ColumnType.BINARY for c in data.columns if c in binary_columns})
    column_types.update({c: ColumnType.NUMERICAL for c in data.columns if c in numerical_columns})
    useful_columns = categorical_columns + numerical_columns + binary_columns
    useful_columns = [col for col in useful_columns if col not in ["x_coord","X_coord","y_coord","Y_coord","smiles"]]
    print("!!! Important !!!")
    print("Useful columns are", useful_columns)
    print("Column types are", column_types)
    print("!!! Important !!!")
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(header_alt(), className='g-0')]),  # Header
        dbc.Row([dbc.Col(molcompass_layout(useful_columns), className='g-0')],id='main-layout'),  # Main
    ], fluid=True
    )
    init_callbacks(app,data,column_types)
    app.run_server(debug=True)


#
# def main():
#       fire.Fire(show)

# if (__name__ == '__main__'):
#       fire.Fire(main)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
