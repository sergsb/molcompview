import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import redis
from molcompview.actions import generate_figure_from_data

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
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(id='logo', children=
                    [
                        dbc.Col(html.Img(src="assets/MolCompass.svg", height="30px")),
                        dbc.Col(dbc.NavbarBrand("MolCompass Viewer", className="ms-2")),
                    ],
                            align="center",
                            className="g-0",
                            ),
                    # href="https://molinfo.space",
                    style={"textDecoration": "none"},
                    className="ml-auto",
                ),
            ],
            fluid=True,
            style={"padding": "0.1rem 3rem"},
        ),
        color="dark",
        dark=True,
        className="mb-6",
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
    return dbc.Col(
        dbc.Row([
            dbc.Label("Select property:",width='auto'),
            dbc.Col(
                dcc.Dropdown(
                    id="molcompass-select-property-dropdown",
                    options={f:f for f in property_options},
                )
            )
        ])
    , width=6, align="center", className='g-2 p-2')



def molcompass_figure():
    wrapper = dbc.Col([
        dcc.Graph(id='molcompass-graph', figure=generate_figure_from_data(), style={'height': '90vh', 'width': '98vw'}),
        dcc.Tooltip(id="molcompass-graph-tooltip")
    ], width={"size": 12}, align="center")
    return wrapper

def range_selector(min=0, max=1, step=0.01, value=[0, 1]):
    #Get vertical height of graph
    return html.Div([dcc.RangeSlider(
            id='molcompass-range-slider',
            min=min,
            max=max,
            step=step,
            value=value,
            # marks={i: str(i) for i in range(min, max, step)},
            marks=None,
            vertical=True,
            #verticalHeight=0.9
        #Caluclate the height of the slider from the height of the figure
        ), dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
        )],
       id='molcompass-range-slider-container',
       style={'margin-right': '10px','margin-left':'0','visibility':'hidden'})

def analysis_layout():
    return dbc.Offcanvas(id="analysis-layout", is_open=False, placement="end")

def molcompass_layout(selectable_columns):
    return dbc.Row([
        dbc.Row(select_property_dropdown(selectable_columns), id="molcompass-show-property-dropdown", justify="center"),
            dbc.Container(id='main-figure-container', className='g-0', children=[
                dbc.Row([
                    dbc.Container([molcompass_figure(),range_selector(),analysis_layout()],
                                  style={'display':'grid',"grid-template-columns": "95% 5% 5%"}, className='g-0', fluid=True),
            ],justify='center')  # Tabs content
                ])
    ],id="molcompass-layout-content", className='g-0')



