The MolCompass application
=======
### Introduction

<img align="left" src="https://user-images.githubusercontent.com/4963384/218703831-1460bc07-7e9f-417e-9b0c-c9675db5de9f.png"> 
<p align="justify">
MolCompassViewer is a part of the MolCompass project. It is a tool that provides a pretrained parametric t-SNE model for chemical space visualization and the visual validation of QSAR/QSPR models.  
</p>

<br clear="left">

## Installation
The package can be installed using `pip` ``pip install molcompview``
<br>

## Run
### Demo
To run the endocrine receptor example described in the manuscript (coming soon), you may do so by executing the following steps.
```
molcompview --demo
```
Then, wait some time for the calculation of coordinates. After that stage, the browser will open, and you can view the interactive visualization of the chemical space.
### Your own dataset
To run on your own dataset, just use:
```
molcompview <input.csv>
```
<br>

## Usage
`MolCompassViewer` intelligently identifies the types of columns within the CSV file, selecting an operational mode based on the presence of specific column types, primarily the molecular structures encoded as SMILES strings.
<br>
  
## Operational Modes 

* STRUCTURE ONLY:
Activated when only the SMILES column is identified.
Focuses on visualizing molecular structures, omitting additional features like color layers and QSAR/QSPR model analyses.
* ALTERNATIVE:
Triggered when additional categorical or numerical columns are found alongside the SMILES column, excluding the Ground Truth and Probabilities columns.
Focuses on exploring the chemical space of compounds, which does not apply to the analysis of models. Users can customize point colors in the visualization based on selected properties.
* FULL:
Dedicated to the visual analysis of binary QSAR/QSPR models.
Available when the CSV file comprises SMILES strings along with Ground Truth and predicted probabilities columns, unlocking access to exclusive features for visualizing binary QSAR/QSPR models.
<img align="left" src="https://github.com/sergsb/molcompview/assets/4963384/4716e786-466a-4412-9f04-b95136bfc1bd.png" width='300px'> 

<br clear="left">

## Applicability domain analysis 
The applicability domain analysis is possible only in FULL mode for binary classification models. To run this tool, first select a tool from the top left corner of the chemical map, then select an area of interest and release. A new frame will open on the right, displaying statistical parameters and charts calculated exclusively for the selected compounds.
![AD-analysis](https://github.com/sergsb/molcompview/assets/4963384/43348a05-d700-428d-9112-d11e29d937d4)

## Screenshots

<img align="left" src="https://github.com/sergsb/molcompview/assets/4963384/07be5580-8d21-4f50-b528-80f5b5d0e5f6.png" width='600px'> 

