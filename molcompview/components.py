import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc
from dash import html
from molcompview.actions import generate_figure_from_data
from . import __class_name__,__probs_name__,__set_name__,__smiles_name__,__x_name__,__y_name__,__loss_name__
def header_alt():
    return dbc.Navbar(
        [
            dbc.Col(html.Img(src = "assets/univie_white.svg", height= "40px"), width={"size": "auto"}),
            dbc.Col(
                html.H4('MolCompassView: Visual analysis of chemical space and AD for QSAR/QSPR modelling', className = "text-center text-white"),width={"size": "auto", "offset": 1}),
        ],  color="#0063a6ff",
        dark=False)


def header():
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(id='logo', children=
                    [
                        dbc.Col(html.Img(src="assets/MolCompass.svg", height="30px")),
                        dbc.Col(dbc.NavbarBrand("MolCompass Viewer", className="ms-2")),
                    ],
                            align="center",
                            className="g-0",
                            ),
                    className="ml-auto",
                ),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        className="mb-4",  # Adjusted margin-bottom
    )
def manual_layout():
    return dbc.Row([
        dbc.Col([
            html.H3("Manual"),
            html.P("This is the manual"),
        ], width=10, align="center")
    ], id="manual-layout-content")


# def tabs():
#     return dbc.Tabs(
#         [
#             dbc.Tab(label="MolCompass", tab_id="molcompass-layout", children=molcompass_layout()),
#             dbc.Tab(label="Manual", tab_id="manual-layout", children=manual_layout()),
#         ],
#         id="tabs",
#         active_tab="molcompass-layout",
#         className="mb-3",
#     )

def get_show_property_options():
    return [{'label': 'test', 'value': 'test', 'disabled': True}]

def select_property_dropdown(property_options):
    def soft_replacement(x):
        internal = {__probs_name__: 'Probability', __class_name__: 'Ground Truth',__loss_name__:"Loss"}
        return internal.get(x, x)

        #Merge the dictionaries


    return dbc.Col(
        dbc.Row([
            dbc.Label("Select property:",width='auto'),
            dbc.Col(
                dcc.Dropdown(
                    id="molcompass-select-property-dropdown",
                    options={f:soft_replacement(f) for f in property_options},
                )
            )
        ])
    , width=5, align="center", className='g-2 p-2')



def molcompass_figure():
    wrapper = dbc.Col([
        dcc.Graph(id='molcompass-graph', figure=generate_figure_from_data(), style={'height': '80vh', 'width': '94vw', 'margin-left': '20px', 'margin-right': '0'}),
        dcc.Tooltip(id="molcompass-graph-tooltip")
    ], width={"size": 11}, align="center")
    return wrapper

def range_selector(min=0, max=1, step=0.01, value=None):
    #Get vertical height of graph
    return html.Div([dcc.RangeSlider(
            id='molcompass-range-slider',
            min=min,
            max=max,
            marks=None,
            # step=step,
            # marks={float(i): '{:2f}'.format(float(i)) for i in np.arange(min, round(max+0.01,2))},
            value=(min, max) if value is None else value,
        )],
       id='molcompass-range-slider-container',
       style={'margin-right': '10px','margin-left':'0','visibility':'hidden','width':'60%'})

def analysis_layout():
    return dbc.Offcanvas(id="analysis-layout", is_open=False, placement="end")

def molcompass_layout(selectable_columns):
    return dbc.Row([
        dbc.Row(select_property_dropdown(selectable_columns), id="molcompass-show-property-dropdown", justify="center"),
            dbc.Container(id='main-figure-container', className='g-0', children=[
                dbc.Row([
                    range_selector(),
                    dbc.Container([molcompass_figure(),analysis_layout()],
                                  className='g-0', fluid=True),
            ],justify='center')  # Tabs content
                ])
    ],id="molcompass-layout-content", className='g-0')



