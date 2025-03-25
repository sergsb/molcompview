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
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
import numpy as np
from dash import dcc
from . import (
    __smiles_name__,
    __class_name__,
    __set_name__,
    __probs_name__,
    __loss_name__,
    __x_name__,
    __y_name__,
    DatasetState,
)


class ColumnType(Enum):
    NUMERICAL = 1
    CATEGORICAL = 2
    BINARY = 3


def generate_figure_from_data(data=None, property=None, column_types=None, range=None):
    if data is None:
        return go.Figure()
    if property is not None:
        if column_types[property] != ColumnType.NUMERICAL:
            data[f"{property}_cat"] = data[property].astype("category")
            fig = px.scatter(
                data,
                x=__x_name__,
                y=__y_name__,
                color=f"{property}_cat",
                color_continuous_scale="viridis",
            )
            # Add colorscale viridis
        else:
            # Create base scatter plot with all points
            fig = px.scatter(
                data,
                x=__x_name__,
                y=__y_name__,
                color=property,
                color_continuous_scale='viridis'
            )
            
            # Update marker opacity based on range
            mask = data[property].between(range[0], range[1])
            opacities = [1.0 if m else 0.1 for m in mask]
            
            # Apply opacity to markers
            fig.update_traces(marker=dict(opacity=opacities))

    else:
        fig = px.scatter(data, x=__x_name__, y=__y_name__)
    # Configure base trace properties
    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
        selectedpoints=None,  # Disable default selection styling
        mode='markers',
        selected=dict(
            marker=dict(color='red', opacity=0.3)  # Light red for selected points
        ),
        unselected=dict(
            marker=dict(opacity=1)  # Keep unselected points at their current opacity
        )
    )

    # Configure selection box and general layout
    fig.update_layout(
        xaxis=dict(title="X - Coordinate"),
        yaxis=dict(title="Y - Coordinate"),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        dragmode='zoom',  # Default to zoom mode
        selectdirection='any',  # Allow selection in any direction
        newselection=dict(
            line=dict(color='red', width=4, dash='solid')  # Make selection box red and thick
        )
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="right", x=0.15)
    )
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
        svg_encoded = base64.b64encode(svg.encode("utf-8"))
        return "data:image/svg+xml;base64," + svg_encoded.decode("utf-8")


def init_callbacks(app, data, column_types, dataset_state):
    @app.callback(
        Output("mode-toggle-button", "children"),
        Output("molcompass-graph", "figure"),
        Output("molcompass-range-slider-container", "style"),
        Output("molcompass-range-slider", "min"),
        Output("molcompass-range-slider", "max"),
        Output("molcompass-range-slider", "value"),
        Input("mode-toggle-button", "n_clicks"),
        Input("molcompass-select-property-dropdown", "value"),
        Input("molcompass-range-slider", "value"),
        State("molcompass-graph", "figure"),
        State("molcompass-range-slider-container", "style"),
    )
    def update_graph(n_clicks, property, range, current_figure, style):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == "mode-toggle-button":
            # Handle mode toggle
            if n_clicks is None:
                button_text = "🔍 Zoom Mode"
                figure = dash.no_update
            else:
                current_mode = current_figure.get('layout', {}).get('dragmode', 'zoom')
                if current_mode == 'zoom':
                    current_figure['layout']['dragmode'] = 'select'
                    button_text = "⬚ Select Mode"
                    figure = current_figure
                else:
                    current_figure['layout']['dragmode'] = 'zoom'
                    button_text = "🔍 Zoom Mode"
                    figure = current_figure
            return button_text, figure, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        else:
            # Handle property/range updates
            current_mode = current_figure.get('layout', {}).get('dragmode', 'zoom')
            button_text = "🔍 Zoom Mode" if current_mode == 'zoom' else "⬚ Select Mode"
            
            if property is None:
                style["visibility"] = "hidden"
                min, max = 0, 0
                figure = generate_figure_from_data(data)
            elif column_types[property] == ColumnType.NUMERICAL:
                style["visibility"] = "visible"
                min, max = data[property].min(), data[property].max()
                figure = generate_figure_from_data(data, property, column_types, range)
            else:
                style["visibility"] = "hidden"
                min, max = 0, 0
                figure = generate_figure_from_data(data, property, column_types)
            
            # Set dragmode to current mode or default to zoom
            figure['layout']['dragmode'] = current_mode
            
            if range is [0, 1]:
                range = [min, max]
            return button_text, figure, style, min, max, range

    @app.callback(
        Output("analysis-layout", "children"),
        Output("analysis-layout", "is_open"),
        Input("molcompass-graph", "selectedData"),
        Input("molcompass-select-property-dropdown", "value")
    )
    def callback(selection, property_value):
        # Get the selected points
        if selection is None:
            return html.Div(), False
        points = selection["points"]
        
        if dataset_state == DatasetState.FULL:
            # Get logits AND values for selected points
            df = data.loc[
                [p["pointNumber"] for p in points],
                [__smiles_name__, __class_name__, __probs_name__],
            ]
            # Calculate predictions using threshold of 0.5
            df['pred'] = (df[__probs_name__] > 0.5).astype(int)
            
            # Calculate confusion matrix
            cm = confusion_matrix(df[__class_name__], df['pred'])
            
            # Create heatmap of confusion matrix
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Viridis',
                showscale=False
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                template='plotly_white',
                height=300,
                margin=dict(l=40, r=20, t=30, b=40)
            )
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(df[__class_name__], df[__probs_name__])
            
            # Plot ROC curve
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name="ROC curve",
                    line=dict(color="firebrick", width=4),
                )
            )
            
            # Add diagonal reference line
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    line=dict(color='black', width=2, dash='dash'),
                    name='Random'
                )
            )
            
            fig_roc.update_layout(
                title="ROC curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template="plotly_white",
                margin=dict(l=40, r=20, t=30, b=40),
                height=300
            )
            
            # Calculate metrics
            sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            accuracy = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
            balanced_accuracy = (sensitivity + specificity) / 2
            f1 = f1_score(df[__class_name__], df['pred'])
            mcc = matthews_corrcoef(df[__class_name__], df['pred'])
            roc_auc = roc_auc_score(df[__class_name__], df[__probs_name__])
            
            # Create metrics table
            table = go.Figure(data=[go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='rgb(55, 83, 109)',
                    font=dict(color='white'),
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Sensitivity', 'Specificity', 'Accuracy', 'Balanced Accuracy', 'F1 score', 'MCC', 'ROC AUC'],
                        [f"{sensitivity:.3f}", f"{specificity:.3f}", f"{accuracy:.3f}", 
                         f"{balanced_accuracy:.3f}", f"{f1:.3f}", f"{mcc:.3f}", f"{roc_auc:.3f}"]
                    ],
                    fill_color=['rgb(245, 245, 245)', 'white'],
                    align='left'
                )
            )])
            table.update_layout(
                title='Statistical Metrics',
                template='plotly_white',
                height=250,
                margin=dict(l=40, r=20, t=30, b=20)
            )
            
            # Create KDE plots for available numerical columns
            kde_figs = []
            plot_configs = []
            
            # Only add plots for columns that exist in the dataframe
            if __probs_name__ in df.columns:
                plot_configs.append((df[__probs_name__], 'Probability Distribution'))
            if __loss_name__ in df.columns:
                plot_configs.append((df[__loss_name__], 'Loss Distribution'))
                
            for prop, title in plot_configs:
                kde_fig = go.Figure()
                
                # Calculate KDE
                kde_points = prop.dropna().values
                if len(kde_points) > 1:  # Need at least 2 points for KDE
                    kde = gaussian_kde(kde_points)
                    x_range = np.linspace(min(kde_points), max(kde_points), 200)
                    y = kde(x_range)

                    # Add filled KDE
                    kde_fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y,
                        mode='lines',
                        line=dict(color='rgb(55, 83, 109)', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(55, 83, 109, 0.2)'
                    ))

                    # Add rug plot
                    kde_fig.add_trace(go.Scatter(
                        x=kde_points,
                        y=[-0.02] * len(kde_points),
                        mode='markers',
                        marker=dict(
                            symbol='line-ns',
                            line=dict(width=1, color='rgb(55, 83, 109)'),
                            size=8
                        ),
                        hoverinfo='x'
                    ))
                else:
                    # Fallback to scatter plot if not enough points
                    kde_fig.add_trace(go.Scatter(
                        x=kde_points,
                        y=[0] * len(kde_points),
                        mode='markers',
                        marker=dict(color='rgb(55, 83, 109)', size=8)
                    ))
                
                # Update layout
                kde_fig.update_layout(
                    title=title,
                    template='plotly_white',
                    showlegend=False,
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=300,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=False,
                        rangemode='nonnegative'
                    )
                )
                kde_figs.append(dcc.Graph(figure=kde_fig))
            
            return [
                dcc.Graph(figure=fig),      # Confusion Matrix
                dcc.Graph(figure=fig_roc),  # ROC Curve
                dcc.Graph(figure=table),    # Metrics Table
                *kde_figs                   # KDE plots for probabilities and loss
            ], True
            
        elif dataset_state == DatasetState.PROPERTY:
            # If no property is selected, show nothing
            if not property_value:
                return html.Div(), False

            # Get all properties for selected points
            selected_df = data.loc[[p["pointNumber"] for p in points]]
            
            # Find numerical properties
            numerical_props = [col for col, type_ in column_types.items() 
                             if type_ == ColumnType.NUMERICAL and 
                             col not in [__x_name__, __y_name__, __probs_name__, __loss_name__]]
            
            if not numerical_props:
                return html.Div("No numerical properties to display"), True
                
            # Create distribution plots for numerical properties
            figs = []
            for prop in numerical_props:
                # Create KDE plot
                fig = go.Figure()
                
                # Calculate KDE for full dataset
                full_kde_points = data[prop].dropna().values
                if len(full_kde_points) > 1:
                    full_kde = gaussian_kde(full_kde_points)
                    x_range = np.linspace(min(full_kde_points), max(full_kde_points), 200)
                    y_full = full_kde(x_range)

                    # Add filled KDE for full dataset
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_full,
                        mode='lines',
                        name='Full Dataset',
                        line=dict(color='rgb(169, 169, 169)', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(169, 169, 169, 0.1)'
                    ))

                # Calculate KDE for selected points
                kde_points = selected_df[prop].dropna().values
                if len(kde_points) > 1:  # Need at least 2 points for KDE
                    kde = gaussian_kde(kde_points)
                    x_range = np.linspace(min(kde_points), max(kde_points), 200)
                    y = kde(x_range)

                    # Add filled KDE for selected points
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y,
                        mode='lines',
                        name='Selected Points',
                        line=dict(color='rgb(55, 83, 109)', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(55, 83, 109, 0.2)'
                    ))

                    # Add rug plot (small lines at actual data points)
                    fig.add_trace(go.Scatter(
                        x=kde_points,
                        y=[-0.02] * len(kde_points),  # Slightly below x-axis
                        mode='markers',
                        marker=dict(
                            symbol='line-ns',
                            line=dict(width=1, color='rgb(55, 83, 109)'),
                            size=8
                        ),
                        name='Data points',
                        hoverinfo='x'
                    ))
                else:
                    # If not enough points for KDE, show scatter plot
                    fig.add_trace(go.Scatter(
                        x=kde_points,
                        y=[0] * len(kde_points),
                        mode='markers',
                        marker=dict(color='rgb(55, 83, 109)', size=8),
                        name='Data points'
                    ))
                
                # Update layout with prettier styling
                fig.update_layout(
                    title=f"{prop} Distribution",
                    xaxis_title=prop,
                    yaxis_title="Density",
                    template="plotly_white",
                    showlegend=True,  # Show legend
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor='rgba(255, 255, 255, 0.8)',  # Semi-transparent white background
                        bordercolor='rgba(211, 211, 211, 0.5)',
                        borderwidth=1
                    ),
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=300,
                    hovermode='x unified',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(211, 211, 211, 0.5)',
                        zeroline=False,
                        rangemode='nonnegative'  # Ensure density is always positive
                    )
                )
                figs.append(dcc.Graph(figure=fig))
            
            return html.Div(figs, style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}), True
        
        # Default case - no visualization
        return html.Div(), False
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="KDE"))

        fig.update_layout(
            title="Distribution of Probabilities",
            xaxis_title="Probability",
            yaxis_title="Density",
            template="plotly_white",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        # Make a confusion matrix
        # Get the threshold
        threshold = 0.5
        # Get the predictions
        df["pred"] = df[__probs_name__].apply(lambda x: 1 if x > threshold else 0)
        # Get the confusion matrix
        cm = confusion_matrix(df[__class_name__], df["pred"])
        # Plot the confusion matrix
        fig2 = ff.create_annotated_heatmap(
            cm, x=["Inactive", "Active"], y=["Inactive", "Active"], colorscale="Viridis"
        )
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
        f1 = f1_score(df[__class_name__], df["pred"])
        # MCC
        mcc = matthews_corrcoef(df[__class_name__], df["pred"])
        # ROC AUC
        roc_auc = roc_auc_score(df[__class_name__], df[__probs_name__])
        # Create a table
        table = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=["Metric", "Value"],
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            [
                                "Sensitivity",
                                "Specificity",
                                "Accuracy",
                                "Balanced Accuracy",
                                "F1 score",
                                "MCC",
                                "ROC AUC",
                            ],
                            [
                                round(sensitivity, 3),
                                round(specificity, 3),
                                round(accuracy, 3),
                                round(balanced_accuracy, 3),
                                round(f1, 3),
                                round(mcc, 3),
                                round(roc_auc, 3),
                            ],
                        ],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )
        table.update_layout(
            title="Metrics",
            template="plotly_white",
            # width=500,
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        return [
            dcc.Graph(figure=fig),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=table),
            dcc.Graph(figure=fig_roc),
        ], True

    @app.callback(
        Output("molcompass-graph-tooltip", "show"),
        Output("molcompass-graph-tooltip", "bbox"),
        Output("molcompass-graph-tooltip", "children"),
        Input("molcompass-graph", "hoverData"),
        Input({"type": "dynamic-dropdown", "index": ALL}, "value"),
    )
    def display_hover(hoverData, dropdown):
        if hoverData is None:
            return False, no_update, no_update

        def act_or_inact(x):
            if x == 0.0:
                return html.P(
                    "Inactive",
                    style={"color": "red", "font-size": "16px", "text-align": "center"},
                )
            else:
                return html.P(
                    "Active",
                    style={
                        "color": "green",
                        "font-size": "16px",
                        "text-align": "center",
                    },
                )

        def probs_and_loss(prob, loss):
            # Make two lines, one for probs and one for loss, turn loss to percentage
            face = ""  # No face
            if loss < 0.3 and (prob < 0.4 or prob > 0.6):
                face = "😊"  # Smiley face
            elif 0.4 <= prob <= 0.6:
                face = "😐"  # Neutral face
            elif loss > 0.7:
                face = "😠"  # Angry face
            line1 = html.P(
                "Prob. (being active): {:.0f}%".format(prob * 100),
                style={"color": "black", "font-size": "16px", "text-align": "center"},
            )
            line2 = html.P(
                "Loss: {:.2f}".format(loss),
                style={"color": "black", "font-size": "16px", "text-align": "center"},
            )
            line3 = html.P(
                face,
                style={"color": "black", "font-size": "35px", "text-align": "center"},
            )
            return html.Div([line1, line2, line3])

        df_subset = data[
            (data[__x_name__] == hoverData["points"][0]["x"])
            & (data[__y_name__] == hoverData["points"][0]["y"])
        ]
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        # Check the mode (NORMAL, ALTERNATIVE_MODE, STRUCTURES_ONLY)

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
            # table = dbc.Table.from_dataframe(pd.DataFrame({"Name":name,"Value":value}), striped=True, bordered=False, hover=True)
            # return table
            def generete_head_for_molcard():
                pass

            # Create property info display for PROPERTIES_ONLY mode
            def make_property_info(point):
                # Create table header
                header = html.Thead(html.Tr([
                    html.Th('Property', style={'text-align': 'left', 'padding': '4px 8px', 'border-bottom': '2px solid #ddd'}),
                    html.Th('Value', style={'text-align': 'right', 'padding': '4px 8px', 'border-bottom': '2px solid #ddd'})
                ]))
                
                # Create table rows for all properties except internal ones
                rows = []
                internal_cols = [__smiles_name__, __x_name__, __y_name__, 'color', 'opacity']
                
                for col, value in point.items():
                    if col not in internal_cols:
                        if isinstance(value, float):
                            formatted_value = f"{value:.3f}"
                        else:
                            formatted_value = str(value)
                        
                        row = html.Tr([
                            html.Td(col, style={'text-align': 'left', 'padding': '4px 8px', 'border-bottom': '1px solid #eee'}),
                            html.Td(formatted_value, style={'text-align': 'right', 'padding': '4px 8px', 'border-bottom': '1px solid #eee'})
                        ])
                        rows.append(row)
                
                # Create table body
                body = html.Tbody(rows)
                
                # Create table with styling
                table = html.Table(
                    [header, body],
                    style={
                        'width': '100%',
                        'border-collapse': 'collapse',
                        'font-family': 'Arial, sans-serif',
                        'font-size': '14px',
                        'margin': '8px 0',
                        'background-color': 'white',
                        'border-radius': '4px',
                        'box-shadow': '0 1px 3px rgba(0,0,0,0.1)'
                    }
                )
                
                return table

            return html.Div(
                [
                    # Only show class and probability info in FULL mode
                    (
                        act_or_inact(point[__class_name__])
                        if dataset_state == DatasetState.FULL
                        else html.Div(id="empty-div")
                    ),
                    (
                        probs_and_loss(point[__probs_name__], point[__loss_name__])
                        if dataset_state == DatasetState.FULL
                        else html.Div(id="empty-div")
                    ),
                    html.Img(
                        src=imgFromSmiles(point[__smiles_name__]),
                        style={"width": "20wh", "height": "20vh"},
                    ),
                    # Show property info in PROPERTY mode
                    (
                        make_property_info(point)
                        if dataset_state == DatasetState.PROPERTY
                        else html.Div(id="empty-div")
                    )
                ],
                style={
                    "font-size": "18px",
                    "display": "inline",
                    "maxWidth": "100px",
                    "maxHeight": "100px",
                    "padding": "5px",
                },
            )

        children = html.Div(
            [make_molecular_card(point) for point in df_subset.to_dict("records")]
        )

        return True, bbox, children
