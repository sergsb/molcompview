import base64
import dash
from enum import Enum, IntEnum
from dash import Output, Input, no_update, ALL, html, State, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
import numpy as np
from dash import dcc
from . import __smiles_name__, __class_name__, __set_name__, __probs_name__, __loss_name__, __x_name__, __y_name__, \
    DatasetState


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
            fig = px.scatter(data, x=__x_name__, y=__y_name__
                             , color=f'{property}_cat',color_continuous_scale='viridis')
            #Add colorscale viridis
        else:
            data['color'] = 'gray'
            data['opacity'] = 0.2
            data['color'] = data[property][data[property].between(range[0],range[1])]
            data['opacity'] = data['opacity'].where(data['color'].isna(),1)
            fig = px.scatter(data, x=__x_name__, y=__y_name__, color=data['color'],opacity=data['opacity'],color_continuous_scale='viridis')

    else:
        fig = px.scatter(data, x=__x_name__, y=__y_name__)
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




def init_callbacks(app,data,column_types,dataset_state):

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
        #print("Range is ",range," inside show_dataset")
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
        if range is [0,1]:
            range = [min,max]
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
        df = data.loc[[p['pointNumber'] for p in points], [__smiles_name__,__class_name__,__probs_name__]]
        fpr, tpr, thresholds = roc_curve(df[__class_name__], df[__probs_name__])
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
        x = np.linspace(min(df[__probs_name__]), max(df[__probs_name__]), 500)
        kde = gaussian_kde(df[__probs_name__])
        y = kde(x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='KDE'))

        fig.update_layout(
            title='Distribution of Probabilities',
            xaxis_title='Probability',
            yaxis_title='Density',
            template='plotly_white',
            margin=dict(l=0, r=0, t=10,b=0)
        )
        # Make a confusion matrix
        # Get the threshold
        threshold = 0.5
        # Get the predictions
        df['pred'] = df[__probs_name__].apply(lambda x: 1 if x > threshold else 0)
        # Get the confusion matrix
        cm = confusion_matrix(df[__class_name__], df['pred'])
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
        f1 = f1_score(df[__class_name__], df['pred'])
        # MCC
        mcc = matthews_corrcoef(df[__class_name__], df['pred'])
        # ROC AUC
        roc_auc = roc_auc_score(df[__class_name__], df[__probs_name__])
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
        def act_or_inact(x):
            if x == 0.0:
                return html.P('Inactive', style={'color': 'red', 'font-size': '16px', 'text-align': 'center'})
            else:
                return html.P('Active', style={'color': 'green', 'font-size': '16px', 'text-align': 'center'})
        def probs_and_loss(prob,loss):
            #Make two lines, one for probs and one for loss, turn loss to percentage
            face = ""  # No face
            if loss < 0.3 and (prob < 0.4 or prob > 0.6):
                face = "😊"  # Smiley face
            elif 0.4 <= prob <= 0.6:
                face = "😐"  # Neutral face
            elif loss > 0.7:
                face = "😠"  # Angry face
            line1 = html.P('Prob. (being active): {:.0f}%'.format(prob*100), style={'color': 'black', 'font-size': '16px', 'text-align': 'center'})
            line2 = html.P('Loss: {:.2f}'.format(loss), style={'color': 'black', 'font-size': '16px', 'text-align': 'center'})
            line3 = html.P(face, style={'color': 'black', 'font-size': '35px', 'text-align': 'center'})
            return html.Div([line1,line2,line3])

        df_subset = data[(data[__x_name__] == hoverData['points'][0]['x']) & (data[__y_name__] == hoverData['points'][0]['y'])]
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        #Check the mode (NORMAL, ALTERNATIVE_MODE, STRUCTURES_ONLY)



        def make_molecular_card(point):
            # def make_info(dropdown):
            #     name = []
            #     value = []
            #     if __probs_name__ in point:
            #         name.append('Probs')
            #         value.append('{:.2f}'.format(point[__probs_name__]))
            #     if __set_name__ in point:
            #         name.append('Loss')
            #         value.append('{:.2f}'.format(point[__loss_name__]))
            #     if __class_name__ in point:
            #         name.append('Ground Truth')
            #         value.append(point[__class_name__])
                # if dropdown is not []:
                #     name = dropdown
                #     value = point[dropdown]
                #table = dbc.Table.from_dataframe(pd.DataFrame({"Name":name,"Value":value}), striped=True, bordered=False, hover=True)
                #return table
            def generete_head_for_molcard():
                pass

            return html.Div([
                act_or_inact(point[__class_name__]) if DatasetState.NORMAL else html.Div(id='empty-div'),
                probs_and_loss(point[__probs_name__],point[__loss_name__]) if DatasetState.NORMAL else html.Div(id='empty-div'),
                html.Img(src=imgFromSmiles(point[__smiles_name__]), style={'width': '20wh', 'height': '20vh'}),
#                 html.Div(
# #                    make_info(dropdown),
#                 style={ 'color': '#000000',
#                                                                          'background-color': '#f5f5f5', 'border': '1px solid #dcdcdc',
#                                                                          'padding': '10px', 'text-align': 'center'})
            ], style={'font-size': '18px','display': 'inline', 'maxWidth': '100px', 'maxHeight': '100px'})


        children = html.Div([
            make_molecular_card(point) for point in df_subset.to_dict('records')
        ])

        return True, bbox, children