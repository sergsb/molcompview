import base64
import dash
from enum import Enum
from dash import Output, Input, no_update, ALL, html, State, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
import numpy as np
from dash import dcc

class ColumnType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2
    BINARY = 3

def generate_figure_from_data(data=None,property=None,column_types=None,range=None):
    if data is None:
        return go.Figure()
    if property is not None:
        if column_types[property] != ColumnType.NUMERICAL:
            data[f'{property}_cat'] = data[property].astype('category')
            fig = px.scatter(data, x="x_coord", y="y_coord", color=f'{property}_cat',color_continuous_scale='viridis')
            #Add colorscale viridis
        else:
            data['color'] = 'gray'
            data['opacity'] = 0.2
            data['color'] = data[property][data[property].between(range[0],range[1])]
            data['opacity'] = data['opacity'].where(data['color'].isna(),1)
            fig = px.scatter(data, x="x_coord", y="y_coord", color=data['color'],opacity=data['opacity'],color_continuous_scale='viridis')

    else:
        fig = px.scatter(data, x="x_coord", y="y_coord")
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout(
        xaxis=dict(title='X - Coordinate'),
        yaxis=dict(title='Y - Coordinate'),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=.02,
        xanchor="right",
        x=.15
    ))
    return fig

def imgFromSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    else:
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 400)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svg_encoded = base64.b64encode(svg.encode('utf-8'))
        return "data:image/svg+xml;base64," + svg_encoded.decode('utf-8')




def init_callbacks(app,data,column_types):

    @app.callback(
    Output('molcompass-graph', 'figure'),
    Output('molcompass-range-slider-container', 'style'),
    Output('molcompass-range-slider', 'min'),
    Output('molcompass-range-slider', 'max'),
    Output('molcompass-range-slider', 'value'),
    State('molcompass-range-slider-container', 'style'),
    [Input('molcompass-select-property-dropdown', 'value'),
     Input('molcompass-range-slider', 'value')])
    def show_dataset(style,property,range):
        if property is None:
            style['visibility'] = 'hidden'
            min, max = 0, 0
            figure = generate_figure_from_data(data)
        elif column_types[property] == ColumnType.NUMERICAL:
            style['visibility'] = 'visible'
            min, max = data[property].min(), data[property].max()
            figure = generate_figure_from_data(data,property,column_types,range)
        else:
            style['visibility'] = 'hidden'
            min, max = 0, 0
            figure = generate_figure_from_data(data,property,column_types)

        return figure,style,min,max,range

    @app.callback(
        Output('analysis-layout', 'children'),
        Output('analysis-layout', 'is_open'),
        Input('molcompass-graph', 'selectedData'),
    )
    def callback(selection):
        # Get the selected points
        if selection is None:
            return html.Div(), False
        points = selection['points']
        # Get logits AND values for selected points
        df = data.loc[[p['pointNumber'] for p in points], ['smiles', 'class', 'logits']]
        fpr, tpr, thresholds = roc_curve(df['class'], df['logits'])
        # Plot ROC curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter
                      (x=fpr, y=tpr,
                       mode='lines',
                       name='ROC curve',
                       line=dict(color='firebrick', width=4)))
        fig_roc.update_layout(
            title='ROC curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            # width=500,
            # height=500,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        # ADD AUC to the plot
        auc = np.trapz(tpr, fpr)
        fig_roc.add_annotation(
            x=0.5,
            y=0.5,
            text=f"AUC: {round(auc, 3)}",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="left",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8
        )
        # fig = create_distplot(df, group_labels=['logits'], bin_size=0.05)
        fig = go.Figure()
        # Create a distribution plot, not a histogram
        fig.add_trace(go.Histogram(x=df['logits'], nbinsx=20))
        fig.update_layout(
            title='Distribution of logits',
            xaxis_title='Logits',
            yaxis_title='Count',
            template='plotly_white',
            # width=500,
            height=200,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        # Make a confusion matrix
        # Get the threshold
        threshold = 0.5
        # Get the predictions
        df['pred'] = df['logits'].apply(lambda x: 1 if x > threshold else 0)
        # Get the confusion matrix
        cm = confusion_matrix(df['class'], df['pred'])
        # Plot the confusion matrix
        fig2 = ff.create_annotated_heatmap(cm, x=['Inactive', 'Active'], y=['Inactive', 'Active'],
                                           colorscale='Viridis')
        # Print a table with Sensitivity,Specificity,Accuracy,Balanced Accuracy, F1 score, MCC,ROC AUC
        # Sensitivity
        sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        # Specificity
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        # Accuracy
        accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
        # Balanced Accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        # F1 score
        f1 = f1_score(df['class'], df['pred'])
        # MCC
        mcc = matthews_corrcoef(df['class'], df['pred'])
        # ROC AUC
        roc_auc = roc_auc_score(df['class'], df['logits'])
        # Create a table
        table = go.Figure(data=[go.Table(
            header=dict(values=['Metric', 'Value'],
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[
                ['Sensitivity', 'Specificity', 'Accuracy', 'Balanced Accuracy', 'F1 score', 'MCC', 'ROC AUC'],
                [round(sensitivity, 3), round(specificity, 3), round(accuracy, 3), round(balanced_accuracy, 3),
                 round(f1, 3), round(mcc, 3), round(roc_auc, 3)]],
                       fill_color='lavender',
                       align='left'))
        ])
        table.update_layout(
            title='Metrics',
            template='plotly_white',
            # width=500,
            height=200,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        return [dcc.Graph(figure=fig), dcc.Graph(figure=fig2), dcc.Graph(figure=table), dcc.Graph(figure=fig_roc)], True

    @app.callback(
        Output("molcompass-graph-tooltip", "show"),
        Output("molcompass-graph-tooltip", "bbox"),
        Output("molcompass-graph-tooltip", "children"),
        Input("molcompass-graph", "hoverData"),
        Input({'type': 'dynamic-dropdown', 'index': ALL}, "value"),
    )
    def display_hover(hoverData, dropdown):
        if hoverData is None:
            return False, no_update, no_update

        df_subset = data[(data['x_coord'] == hoverData['points'][0]['x']) & (data['y_coord'] == hoverData['points'][0]['y'])]
        pt = hoverData["points"][0]
        bbox = pt["bbox"]

        def make_molecular_card(point):
            def make_info(dropdown):
                name = []
                value = []
                if 'logits' in point:
                    name.append('Logits')
                    value.append('{:.2f}'.format(point['logits']))
                if 'loss' in point:
                    name.append('Loss')
                    value.append('{:.2f}'.format(point['loss']))
                if 'class' in point:
                    name.append('Class')
                    value.append(point['class'])
                # if dropdown is not []:
                #     name = dropdown
                #     value = point[dropdown]
                table = dbc.Table.from_dataframe(pd.DataFrame({"Name":name,"Value":value}), striped=True, bordered=False, hover=True)
                return table


            return html.Div([
                html.Img(src=imgFromSmiles(point['smiles']), style={'width': '20wh', 'height': '20vh'}),
                #Demonstrate the selected property
                html.Div(
                    make_info(dropdown),
                style={ 'color': '#000000',
                                                                         'background-color': '#f5f5f5', 'border': '1px solid #dcdcdc',
                                                                         'padding': '10px', 'text-align': 'center'})
                #html.P(, style={'width': '100%', 'height': '100%'}),
                #Make a table with Doha\'s models and Iris\'s models for each target
                #html.P(f'Doha\'s model: {round(point["doha"],3)}'),
                #html.P(f'Iris\' model: {round(point["iris"],3)}'),
            ], style={'font-size': '18px','display': 'inline', 'maxWidth': '100px', 'maxHeight': '100px'})


        children = html.Div([
            make_molecular_card(point) for point in df_subset.to_dict('records')
        ])

        return True, bbox, children

    @app.callback(
        Output('molcompass-range-slider', 'verticalHeight'),
        Input('interval-component', 'n_intervals')
    )
    def update_slider_height(n_intervals):
        if n_intervals == 0:
            raise PreventUpdate
        height = dash.callback_context.eval_js('getFigureHeight()')
        print("Height is", height)
        return height