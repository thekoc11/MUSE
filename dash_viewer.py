#!/usr/bin/env python
"""
Dash application to visualize MUSE training progress.
Run this in a separate terminal while unsupervised.py is training.
"""

import os
import json
import time
import argparse
# Update imports for newer Dash versions
from dash import Dash, dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import plotly.io as pio
import os.path

# Paths for reading metrics
METRICS_FILE = 'training_metrics.json'
ALIGNMENT_METRICS_FILE = 'alignment_metrics.json'
PLOTS_FOLDER = 'plots'

# Create plots folder if it doesn't exist
if not os.path.exists(PLOTS_FOLDER):
    os.makedirs(PLOTS_FOLDER)

# Create Dash app
app = Dash(__name__, suppress_callback_exceptions=True)

# App layout with tabs for different visualizations
app.layout = html.Div([
    html.H1("MUSE Visualization Dashboard", style={'textAlign': 'center'}),
    
    # Tabs for different types of visualization
    dcc.Tabs(id='tabs', value='unsupervised-tab', children=[
        # Tab 1: Unsupervised Training Visualization
        dcc.Tab(label="Unsupervised Training", value='unsupervised-tab', children=[]),
        
        # Tab 2: Alignment Experiments Visualization
        dcc.Tab(label="Alignment Experiments", value='alignment-tab', children=[]),
    ]),
    
    # Content div - this will be updated based on which tab is active
    html.Div(id='tab-content'),
    
    # Hidden div to store active tab state
    html.Div(id='active-tab', children='unsupervised-tab', style={'display': 'none'}),
])

# Callback to update the active tab
@app.callback(
    Output('active-tab', 'children'),
    [Input('tabs', 'value')]
)
def update_active_tab(tab):
    return tab

# Callback to update tab content based on active tab
@app.callback(
    Output('tab-content', 'children'),
    [Input('active-tab', 'children')]
)
def render_tab_content(active_tab):
    if active_tab == 'unsupervised-tab':
        # Content for unsupervised training tab
        return html.Div([
    html.Div([
        html.Button('Refresh', id='refresh-button', n_clicks=0, 
                    style={'marginRight': '10px'}),
        dcc.Dropdown(
            id='refresh-interval',
            options=[
                {'label': '1 second', 'value': 1},
                {'label': '5 seconds', 'value': 5},
                {'label': '10 seconds', 'value': 10},
                {'label': '30 seconds', 'value': 30},
                {'label': 'Manual refresh only', 'value': 0}
            ],
            value=5,
            style={'width': '200px', 'display': 'inline-block'}
        ),
        html.Div(id='last-update-time', style={'marginLeft': '20px', 'display': 'inline-block'}),
                
                # Add view toggle button
                html.Button(
                    'Toggle Paper View', 
                    id='view-toggle', 
                    n_clicks=0,
                    style={'marginLeft': '20px'}
                ),
                
                # Add export button
                html.Button(
                    'Export Charts', 
                    id='unsupervised-export-button', 
                    n_clicks=0,
                    style={'marginLeft': '20px'}
                ),
    ], style={'margin': '10px'}),
    html.Div([
                dcc.Graph(id='metrics-graph', style={'height': '600px'}),
    ]),
    # Add a new graph for refinement
    html.Div([
        html.H3("Refinement Phase Progress (Precision@1 CSLS)", style={'textAlign': 'center'}),
        dcc.Graph(id='refinement-graph', style={'height': '400px'})
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # default 5 seconds
        n_intervals=0
    ),
            # Hidden div to store data and view state
            html.Div(id='metrics-store', style={'display': 'none'}),
            html.Div(id='view-state', children='dashboard', style={'display': 'none'}),
            html.Div(id='unsupervised-export-status', style={'margin': '10px', 'color': 'green'}),
        ])
    
    elif active_tab == 'alignment-tab':
        # Content for alignment experiments tab
        return html.Div([
            html.Div([
                html.Button('Refresh Alignment', id='refresh-alignment-button', n_clicks=0, 
                            style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='alignment-refresh-interval',
                    options=[
                        {'label': '1 second', 'value': 1},
                        {'label': '5 seconds', 'value': 5},
                        {'label': '10 seconds', 'value': 10},
                        {'label': '30 seconds', 'value': 30},
                        {'label': 'Manual refresh only', 'value': 0}
                    ],
                    value=5,
                    style={'width': '200px', 'display': 'inline-block'}
                ),
                html.Div(id='alignment-update-time', style={'marginLeft': '20px', 'display': 'inline-block'}),
                
                # Dropdown to select visualization type
                dcc.Dropdown(
                    id='alignment-view-type',
                    options=[
                        {'label': 'Dictionary Size Comparison', 'value': 'dict_size'},
                        {'label': 'Refinement Steps', 'value': 'refinement'},
                    ],
                    value='dict_size',
                    style={'width': '300px', 'display': 'inline-block', 'marginLeft': '20px'}
                ),
                
                # Dropdown to select dictionary size for refinement view
                dcc.Dropdown(
                    id='dictionary-size-selector',
                    placeholder="Select dictionary size",
                    style={'width': '200px', 'display': 'inline-block', 'marginLeft': '20px'}
                ),
                
                # Add export button
                html.Button(
                    'Export Charts', 
                    id='alignment-export-button', 
                    n_clicks=0,
                    style={'marginLeft': '20px'}
                ),
            ], style={'margin': '10px'}),
            html.Div([
                dcc.Graph(id='alignment-graph', style={'height': '800px'}),
            ]),
            dcc.Interval(
                id='alignment-interval-component',
                interval=5*1000,  # default 5 seconds
                n_intervals=0
            ),
            # Hidden div to store alignment data
            html.Div(id='alignment-metrics-store', style={'display': 'none'}),
            html.Div(id='alignment-export-status', style={'margin': '10px', 'color': 'green'}),
        ])
    
    # Default content if tab value is not recognized
    return html.Div([
        html.H3("Tab content not found")
])

@app.callback(
    Output('interval-component', 'interval'),
    [Input('refresh-interval', 'value')]
)
def update_interval(value):
    if value == 0:  # Manual refresh only
        return 24 * 60 * 60 * 1000  # Set to 1 day (effectively disabled)
    return value * 1000  # Convert to milliseconds

@app.callback(
    Output('alignment-interval-component', 'interval'),
    [Input('alignment-refresh-interval', 'value')]
)
def update_alignment_interval(value):
    if value == 0:  # Manual refresh only
        return 24 * 60 * 60 * 1000  # Set to 1 day (effectively disabled)
    return value * 1000  # Convert to milliseconds

@app.callback(
    [Output('metrics-graph', 'figure'),
     Output('refinement-graph', 'figure'),
     Output('last-update-time', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('refresh-button', 'n_clicks'),
     Input('view-state', 'children')],
    [State('active-tab', 'children')]  # Add active-tab as state
)
def update_graph(n_intervals, n_clicks, view_mode, active_tab):
    # Only update if the unsupervised tab is active
    if active_tab != 'unsupervised-tab':
        # Return empty figures when tab is not active
        return go.Figure(), go.Figure(), "No data available yet"
    
    # Initialize empty figures
    fig_main = go.Figure()
    fig_refinement = go.Figure()
    update_time_str = "No data available yet"

    # Read current metrics
    try:
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)
        update_time_str = f"Last updated: {time.strftime('%H:%M:%S', time.localtime())}"
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty figures if no data yet
        return fig_main, fig_refinement, update_time_str

    # Check if data exists
    if not data or 'epochs' not in data or not data['epochs']:
        return fig_main, fig_refinement, update_time_str
    
    # --- Data Processing --- 
    epochs = data['epochs']
    wta = data.get('word_translation_accuracy', [])
    disc_acc = data.get('discriminator_accuracy', [])
    unsup_crit = data.get('unsupervised_criterion', [])
    disc_loss = data.get('discriminator_loss', []) # Get discriminator loss

    # Ensure all metric lists have the same length as epochs
    # Pad with None if necessary (though ideally they should match)
    wta += [None] * (len(epochs) - len(wta))
    disc_acc += [None] * (len(epochs) - len(disc_acc))
    unsup_crit += [None] * (len(epochs) - len(unsup_crit))
    disc_loss += [None] * (len(epochs) - len(disc_loss)) # Pad loss list

    # Identify refinement phase
    refinement_start_index = -1
    last_adv_epoch_val = 0
    for i, epoch in enumerate(epochs):
        if isinstance(epoch, (int, float)) and not float(epoch).is_integer():
            last_adv_epoch_val = max(last_adv_epoch_val, epoch)
        elif isinstance(epoch, (int, float)) and float(epoch).is_integer() and epoch > last_adv_epoch_val + 1: 
             # Heuristic: Refinement starts when integer epoch jumps more than 1 after last fractional epoch
             # Or it's the first integer epoch strictly greater than the last fractional epoch + 1
             if refinement_start_index == -1:
                 # Check if the *previous* epoch was the last adversarial eval epoch
                 if i > 0 and float(epochs[i-1]).is_integer() and epochs[i-1] >= last_adv_epoch_val: 
                     refinement_start_index = i
                     break # Found start of refinement

    # Separate data
    adv_epochs, adv_wta, adv_disc_acc, adv_unsup_crit, adv_disc_loss = [], [], [], [], []
    ref_iterations, ref_wta = [], []

    if refinement_start_index != -1:
        adv_indices = list(range(refinement_start_index))
        ref_indices = list(range(refinement_start_index, len(epochs)))
    else:
        adv_indices = list(range(len(epochs)))
        ref_indices = []

    # Populate adversarial data
    for i in adv_indices:
        adv_epochs.append(epochs[i])
        adv_wta.append(wta[i])
        adv_disc_acc.append(disc_acc[i])
        adv_unsup_crit.append(unsup_crit[i])
        adv_disc_loss.append(disc_loss[i]) # Separate loss data

    # Populate refinement data
    for i, idx in enumerate(ref_indices):
        ref_iterations.append(i) # 0, 1, 2...
        ref_wta.append(wta[idx])
        
    # --- Plotting Logic --- 
    current_time = time.strftime("%H:%M:%S", time.localtime())

    # Paper view (single plot like the reference image) - Adversarial Phase Only
    if view_mode == 'paper':
        if adv_epochs:
            # Filter for integer epochs (full evaluation points) in adversarial phase
            is_integer_epoch = [float(e).is_integer() for e in adv_epochs]
            epoch_indices = [i for i, is_int in enumerate(is_integer_epoch) if is_int]
            
            if epoch_indices:
                # Extract epoch-level data for adversarial phase
                epoch_x = [adv_epochs[i] for i in epoch_indices]
                word_trans_acc = [adv_wta[i] for i in epoch_indices]
                disc_accuracy = [adv_disc_acc[i] for i in epoch_indices]
                unsup_criterion = [adv_unsup_crit[i] for i in epoch_indices]

                # Filter out None values for plotting
                valid_indices = [j for j, (w, d, u) in enumerate(zip(word_trans_acc, disc_accuracy, unsup_criterion)) if w is not None and d is not None and u is not None]
                epoch_x = [epoch_x[j] for j in valid_indices]
                word_trans_acc = [word_trans_acc[j] for j in valid_indices]
                disc_accuracy = [disc_accuracy[j] for j in valid_indices]
                unsup_criterion = [unsup_criterion[j] for j in valid_indices]

                if epoch_x: 
                    # Add vertical grid lines like in the paper
                    max_epoch_val = max(epoch_x) if epoch_x else 0
                    for x_val in range(0, int(max_epoch_val) + 20, 20):
                        fig_main.add_vline(
                            x=x_val, line_width=1, line_dash="solid", line_color="lightgray"
                        )
                    
                    # Word Translation Accuracy (blue)
                    fig_main.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=word_trans_acc,
                        mode='lines', 
                        name='Word Translation Accuracy',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Discriminator Accuracy (red)
                    fig_main.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=disc_accuracy,
                        mode='lines', 
                        name='Discriminator Accuracy',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Unsupervised Criterion (black)
                    fig_main.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=unsup_criterion,
                        mode='lines', 
                        name='Unsupervised Criterion',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Layout to match the paper exactly
                    fig_main.update_layout(
                        title='Unsupervised Model Selection (Adversarial Phase)',
                        xaxis_title='Epoch',
                        yaxis_title='Accuracy / Value',
                        legend=dict(x=0.7, y=0.1, font=dict(size=10)),
                        height=600,
                        width=800,
                        margin=dict(l=50, r=20, t=50, b=50),
                        template="plotly_white",
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=12),
                        xaxis=dict(
                            range=[0, max_epoch_val + 10 if epoch_x else 10],
                            showgrid=True,
                            gridwidth=1,
                            gridcolor='lightgray'
                        ),
                        yaxis=dict(showgrid=False)
                    )
            update_time_str = f"Paper view | {current_time}"
        
        # Plot refinement separately
        if ref_iterations and any(v is not None for v in ref_wta):
            valid_ref_indices = [i for i, v in enumerate(ref_wta) if v is not None]
            ref_x = [ref_iterations[i] for i in valid_ref_indices]
            ref_y = [ref_wta[i] for i in valid_ref_indices]

            if ref_x:
                fig_refinement.add_trace(go.Scatter(
                    x=ref_x,
                    y=ref_y,
                    mode='lines+markers',
                    name='P@1 CSLS',
                    line=dict(color='purple', width=2)
                ))
                fig_refinement.update_layout(
                    title="Refinement Phase (Precision@1 CSLS)",
                    xaxis_title='Refinement Iteration',
                    yaxis_title='Accuracy (%)',
                    template='plotly_white',
                    xaxis=dict(tickmode='linear'), # Ensure integer ticks
                    margin=dict(t=50, b=40)
                )
                fig_refinement.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_refinement.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig_main, fig_refinement, update_time_str

    # Dashboard view (original two subplots + refinement plot)
    else:
        fig_main = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("Iteration-Level Discriminator Loss (Adversarial)", "Epoch-Level Evaluation (Adversarial)"),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        if adv_epochs:
            # Filter iteration vs epoch points for adversarial phase
            is_integer_epoch_adv = [float(e).is_integer() for e in adv_epochs]
            iter_indices_adv = [i for i, is_int in enumerate(is_integer_epoch_adv) if not is_int and adv_disc_loss[i] is not None] # Filter based on loss
            epoch_indices_adv = [i for i, is_int in enumerate(is_integer_epoch_adv) if is_int and adv_wta[i] is not None and adv_unsup_crit[i] is not None and adv_disc_acc[i] is not None]
            
            # SUBPLOT 1: Iteration-level discriminator cost (Adversarial)
            if iter_indices_adv:
                iter_epochs_adv = [adv_epochs[i] for i in iter_indices_adv]
                iter_disc_loss_adv = [adv_disc_loss[i] for i in iter_indices_adv] # Use loss data
                
                fig_main.add_trace(
                    go.Scatter(
                        x=iter_epochs_adv, 
                        y=iter_disc_loss_adv, # Plot loss
                        mode='lines', 
                        name='Discriminator Loss',
                        line=dict(color='red', width=2),
                    ),
                    row=1, col=1
                )
                
                # Add epoch boundaries as vertical lines
                epoch_boundaries_adv = sorted(set([int(e) for e in adv_epochs if float(e).is_integer()]))
                for epoch in epoch_boundaries_adv:
                    fig_main.add_vline(
                        x=epoch, line_width=1, line_dash="dash", line_color="gray",
                        row=1, col=1
                    )
            
            # SUBPLOT 2: Epoch-level evaluation metrics (Adversarial)
            if epoch_indices_adv:
                epoch_x_adv = [adv_epochs[i] for i in epoch_indices_adv]
                wta_adv = [adv_wta[i] for i in epoch_indices_adv]
                unsup_crit_adv = [adv_unsup_crit[i] for i in epoch_indices_adv]
                disc_acc_adv_epoch = [adv_disc_acc[i] for i in epoch_indices_adv]

                # Add vertical grid lines
                for x_val in epoch_x_adv:
                    fig_main.add_vline(
                        x=x_val, line_width=1, line_dash="solid", line_color="lightgray",
                        row=2, col=1
                    )
                
                # Word translation accuracy (blue line)
                fig_main.add_trace(
                    go.Scatter(
                        x=epoch_x_adv, 
                        y=wta_adv,
                        mode='lines', 
                        name='Word Translation Accuracy',
                        line=dict(color='blue', width=2),
                    ),
                    row=2, col=1
                )
                
                # Discriminator accuracy (red line)
                fig_main.add_trace(
                    go.Scatter(
                        x=epoch_x_adv, 
                        y=disc_acc_adv_epoch,
                        mode='lines', 
                        name='Discriminator Accuracy',
                        line=dict(color='red', width=2),
                    ),
                    row=2, col=1
                )
                
                # Unsupervised criterion (black line)
                fig_main.add_trace(
                    go.Scatter(
                        x=epoch_x_adv, 
                        y=unsup_crit_adv,
                        mode='lines', 
                        name='Unsupervised Criterion',
                        line=dict(color='black', width=2),
                    ),
                    row=2, col=1
                )
        
        # Update main figure layout
        fig_main.update_layout(
            title='MUSE Training Progress (Adversarial Phase)',
            height=600, # Keep adjusted height
            width=None,  # Responsive width
            template="plotly_white",
            font=dict(family="Arial", size=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update axes labels and grid for main figure
        fig_main.update_xaxes(title_text="Iteration (Epoch)", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_main.update_yaxes(title_text="Loss", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_main.update_xaxes(title_text="Epoch", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_main.update_yaxes(title_text="Accuracy / Value", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='LightGray')

        # Plot refinement phase separately
        if ref_iterations and any(v is not None for v in ref_wta):
            valid_ref_indices = [i for i, v in enumerate(ref_wta) if v is not None]
            ref_x = [ref_iterations[i] for i in valid_ref_indices]
            ref_y = [ref_wta[i] for i in valid_ref_indices]
            
            if ref_x:
                fig_refinement.add_trace(go.Scatter(
                    x=ref_x,
                    y=ref_y,
                    mode='lines+markers',
                    name='P@1 CSLS',
                    line=dict(color='purple', width=2)
                ))
                fig_refinement.update_layout(
                    # Title moved to H3 element above the graph
                    xaxis_title='Refinement Iteration',
                    yaxis_title='Accuracy (%)',
                    template='plotly_white',
                    height=400,
                    xaxis=dict(tickmode='linear'), # Ensure integer ticks
                    margin=dict(t=20, b=40) # Reduced top margin
                )
                fig_refinement.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig_refinement.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        update_time_str = f"Dashboard view | {current_time}"
        return fig_main, fig_refinement, update_time_str

    # Fallback if no data processed
    return fig_main, fig_refinement, update_time_str

@app.callback(
    [Output('dictionary-size-selector', 'options'),
     Output('dictionary-size-selector', 'value')],
    [Input('alignment-view-type', 'value'),
     Input('alignment-interval-component', 'n_intervals'),
     Input('refresh-alignment-button', 'n_clicks')],
    [State('active-tab', 'children')]  # Add active-tab as state
)
def update_dictionary_dropdown(view_type, n_intervals, n_clicks, active_tab):
    # Only update if alignment tab is active
    if active_tab != 'alignment-tab':
        return [], None
    
    # Only populate dictionary size selector when in refinement view
    if view_type != 'refinement':
        return [], None
    
    try:
        with open(ALIGNMENT_METRICS_FILE, 'r') as f:
            data = json.load(f)
        
        if 'dictionary_sizes' in data and data['dictionary_sizes']:
            options = [{'label': f"{size}", 'value': str(size)} for size in data['dictionary_sizes']]
            # Default to first size
            default_value = str(data['dictionary_sizes'][0]) if data['dictionary_sizes'] else None
            return options, default_value
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    return [], None

@app.callback(
    [Output('alignment-graph', 'figure'),
     Output('alignment-update-time', 'children')],
    [Input('alignment-interval-component', 'n_intervals'),
     Input('refresh-alignment-button', 'n_clicks'),
     Input('alignment-view-type', 'value'),
     Input('dictionary-size-selector', 'value')],
    [State('active-tab', 'children')]  # Add active-tab as state
)
def update_alignment_graph(n_intervals, n_clicks, view_type, dict_size, active_tab):
    # Only update if alignment tab is active
    if active_tab != 'alignment-tab':
        # Return empty figure when tab is not active
        return go.Figure(), ""
    
    # Try to get data from the context of the callback to see which input triggered it
    ctx = callback_context
    
    # Read current alignment metrics
    try:
        with open(ALIGNMENT_METRICS_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return go.Figure(), "No alignment data available yet"
    
    # Dictionary size comparison view
    if view_type == 'dict_size':
        # Create figure with subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Precision@1", "Precision@5", 
                "Average Cosine Similarity", "Orthogonality Error"
            ),
            shared_xaxes=True
        )
        
        # Make sure we have dictionary sizes and metrics
        if ('dictionary_sizes' in data and data['dictionary_sizes'] and
            'precision_at_1_nn' in data and 'precision_at_1_csls' in data and
            'precision_at_5_nn' in data and 'precision_at_5_csls' in data and
            'avg_cosine_similarity' in data and 'orthogonality_error' in data):
            
            # Sort all data by dictionary size for consistent plotting
            size_indices = sorted(range(len(data['dictionary_sizes'])), key=lambda i: data['dictionary_sizes'][i])
            dict_sizes = [str(data['dictionary_sizes'][i]) for i in size_indices]  # Convert to strings for categorical x-axis
            
            # Get precision values directly without further modification
            p1_nn = [data['precision_at_1_nn'][i] for i in size_indices]
            p1_csls = [data['precision_at_1_csls'][i] for i in size_indices]
            p5_nn = [data['precision_at_5_nn'][i] for i in size_indices]
            p5_csls = [data['precision_at_5_csls'][i] for i in size_indices]
            cos_sim = [data['avg_cosine_similarity'][i] for i in size_indices]
            ortho_err = [data['orthogonality_error'][i] for i in size_indices]
            
            # Add precision@1 traces
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=p1_nn, mode='lines+markers', name='NN',
                           line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=p1_csls, mode='lines+markers', name='CSLS',
                           line=dict(color='red')),
                row=1, col=1
            )
            
            # Add precision@5 traces
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=p5_nn, mode='lines+markers', name='NN',
                           line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=p5_csls, mode='lines+markers', name='CSLS',
                           line=dict(color='red')),
                row=1, col=2
            )
            
            # Add cosine similarity
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=cos_sim, mode='lines+markers',
                           line=dict(color='green')),
                row=2, col=1
            )
            
            # Add orthogonality error
            fig.add_trace(
                go.Scatter(x=dict_sizes, y=ortho_err, mode='lines+markers',
                           line=dict(color='orange')),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title='Word Translation Accuracy by Dictionary Size',
                height=800,
                width=None,  # Responsive width
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
            )
            
            # Update x and y axis labels - including first row x-axis labels
            # Set x-axis type to categorical for dictionary sizes and ensure tick labels are visible
            fig.update_xaxes(title_text="Dictionary Size", type='category', showticklabels=True, row=1, col=1)
            fig.update_xaxes(title_text="Dictionary Size", type='category', showticklabels=True, row=1, col=2)
            fig.update_xaxes(title_text="Dictionary Size", type='category', row=2, col=1)
            fig.update_xaxes(title_text="Dictionary Size", type='category', row=2, col=2)
            
            # Update y-axes to use appropriate ranges - remove any fixed ranges that might 
            # be causing the 0-100 scaling
            fig.update_yaxes(title_text="Precision", row=1, col=1)
            fig.update_yaxes(title_text="Precision", row=1, col=2)
            fig.update_yaxes(title_text="Cosine Similarity", row=2, col=1)
            fig.update_yaxes(title_text="Error", row=2, col=2)
            
            # Add grid lines
            for row in range(1, 3):
                for col in range(1, 3):
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
    
    # Refinement steps view
    elif view_type == 'refinement' and dict_size:
        # Find the index for the selected dictionary size
        if 'dictionary_sizes' in data and dict_size:
            try:
                dict_size_int = int(dict_size)
                if dict_size_int in data['dictionary_sizes']:
                    idx = data['dictionary_sizes'].index(dict_size_int)
                    refinement_data = data['refinement_steps'][idx]
                    
                    # Create figure with subplots for different metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            "Precision@1 (NN)", "Precision@1 (CSLS)", 
                            "Average Cosine Similarity", "Orthogonality Error"
                        ),
                        shared_xaxes=True
                    )
                    
                    # Convert steps to strings for categorical x-axis
                    steps = [str(step) for step in refinement_data['steps']]
                    
                    # Add precision@1 NN
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['precision_at_1_nn'], 
                                   mode='lines+markers', name='Precision@1 (NN)',
                                   line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    # Add precision@1 CSLS
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['precision_at_1_csls'], 
                                   mode='lines+markers', name='Precision@1 (CSLS)',
                                   line=dict(color='red')),
                        row=1, col=2
                    )
                    
                    # Add cosine similarity
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['avg_cosine_similarity'], 
                                   mode='lines+markers', name='Avg Cosine Similarity',
                                   line=dict(color='green')),
                        row=2, col=1
                    )
                    
                    # Add orthogonality error
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['orthogonality_error'], 
                                   mode='lines+markers', name='Orthogonality Error',
                                   line=dict(color='orange')),
                        row=2, col=2
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Refinement Steps for Dictionary Size {dict_size}',
                        height=800,
                        width=None,  # Responsive width
                        template="plotly_white"
                    )
                    
                    # Update x and y axis labels, including first row
                    # Set x-axis type to categorical for steps and ensure tick labels are visible
                    fig.update_xaxes(title_text="Refinement Step", type='category', showticklabels=True, row=1, col=1)
                    fig.update_xaxes(title_text="Refinement Step", type='category', showticklabels=True, row=1, col=2)
                    fig.update_xaxes(title_text="Refinement Step", type='category', row=2, col=1)
                    fig.update_xaxes(title_text="Refinement Step", type='category', row=2, col=2)
                    
                    # Update y-axes to use appropriate ranges - remove any fixed ranges 
                    fig.update_yaxes(title_text="Precision", row=1, col=1)
                    fig.update_yaxes(title_text="Precision", row=1, col=2)
                    fig.update_yaxes(title_text="Cosine Similarity", row=2, col=1)
                    fig.update_yaxes(title_text="Error", row=2, col=2)
                    
                    # Add grid lines
                    for row in range(1, 3):
                        for col in range(1, 3):
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
                else:
                    fig = go.Figure()
                    fig.update_layout(
                        title=f'Dictionary size {dict_size} not found in data',
                        height=800
                    )
            except (ValueError, IndexError):
                fig = go.Figure()
                fig.update_layout(
                    title='Error loading refinement data',
                    height=800
                )
        else:
            fig = go.Figure()
            fig.update_layout(
                title='No refinement data available',
                height=800
            )
    else:
        # Default empty figure
        fig = go.Figure()
        fig.update_layout(
            title='Select a view type to display alignment data',
            height=800
        )
    
    current_time = time.strftime("%H:%M:%S", time.localtime())
    return fig, f"Last updated: {current_time}"

# Add callback for view toggle
@app.callback(
    Output('view-state', 'children'),
    [Input('view-toggle', 'n_clicks')]
)
def toggle_view(n_clicks):
    if n_clicks % 2 == 0:
        return 'dashboard'
    else:
        return 'paper'

# Function to export chart figures
def export_unsupervised_charts():
    """Export unsupervised training charts to the plots folder"""
    try:
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No data available for export"
    
    export_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    # Create dashboard view for export
    if data['epochs'] and len(data['epochs']) > 0:
        # Separate iteration-level and epoch-level data
        is_integer_epoch = [float(e).is_integer() for e in data['epochs']]
        
        # Filter out None values from metrics
        non_none_indices = []
        for i, (wta, ua, da) in enumerate(zip(
            data['word_translation_accuracy'],
            data['unsupervised_criterion'],
            data['discriminator_accuracy']
        )):
            if wta is not None and ua is not None:
                non_none_indices.append(i)
        
        # Iteration indices are where epoch is not an integer
        iter_indices = [i for i, is_int in enumerate(is_integer_epoch) if not is_int]
        # Epoch indices are where epoch is an integer or we have valid evaluation metrics
        epoch_indices = list(set([i for i, is_int in enumerate(is_integer_epoch) if is_int] + non_none_indices))
        
        # Only plot data if we have any
        if epoch_indices:
            # Get epoch-level data, filtering out None values
            epoch_x = []
            word_trans_acc = []
            unsup_criterion = []
            disc_acc = []
            
            for i in epoch_indices:
                # Only include points where metrics are not None
                if (data['word_translation_accuracy'][i] is not None and 
                    data['unsupervised_criterion'][i] is not None):
                    epoch_x.append(data['epochs'][i])
                    word_trans_acc.append(data['word_translation_accuracy'][i])
                    unsup_criterion.append(data['unsupervised_criterion'][i])
                    # For discriminator accuracy at epoch level, use the value if available
                    if data['discriminator_accuracy'][i] is not None:
                        disc_acc.append(data['discriminator_accuracy'][i])
                    else:
                        disc_acc.append(0)  # Fallback if not available
            
            if epoch_x:  # Only plot if we have valid data
                # Create and save the paper view (main reference plot)
                fig_paper = go.Figure()
                
                # Add vertical grid lines like in the paper
                for x_val in range(0, int(max(epoch_x)) + 20, 20):
                    fig_paper.add_vline(
                        x=x_val, line_width=1, line_dash="solid", line_color="lightgray"
                    )
                
                # Word Translation Accuracy (blue)
                fig_paper.add_trace(go.Scatter(
                    x=epoch_x, 
                    y=word_trans_acc,
                    mode='lines', 
                    name='Word Translation Accuracy',
                    line=dict(color='blue', width=2)
                ))
                
                # Discriminator Accuracy (red)
                fig_paper.add_trace(go.Scatter(
                    x=epoch_x, 
                    y=disc_acc,
                    mode='lines', 
                    name='Discriminator Accuracy',
                    line=dict(color='red', width=2)
                ))
                
                # Unsupervised Criterion (black)
                fig_paper.add_trace(go.Scatter(
                    x=epoch_x, 
                    y=unsup_criterion,
                    mode='lines', 
                    name='Unsupervised Criterion',
                    line=dict(color='black', width=2)
                ))
                
                # Layout to match the paper exactly
                fig_paper.update_layout(
                    title='Unsupervised Model Selection',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    legend=dict(
                        x=0.7,
                        y=0.1,
                        font=dict(size=10)
                    ),
                    height=600,
                    width=800,
                    margin=dict(l=50, r=20, t=50, b=50),
                    template="plotly_white",
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    xaxis=dict(
                        range=[0, max(epoch_x) + 10],
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        showgrid=False
                    )
                )
                
                # Save the paper view
                paper_file = os.path.join(PLOTS_FOLDER, f"unsupervised_model_selection_{export_time}.png")
                pio.write_image(fig_paper, paper_file, scale=2)
                
                # Create and save individual metric plots
                # 1. Word Translation Accuracy
                fig_wta = go.Figure()
                fig_wta.add_trace(go.Scatter(
                    x=epoch_x, y=word_trans_acc, mode='lines+markers', 
                    name='Word Translation Accuracy', line=dict(color='blue', width=2)
                ))
                fig_wta.update_layout(
                    title='Word Translation Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy (%)',
                    template="plotly_white"
                )
                wta_file = os.path.join(PLOTS_FOLDER, f"word_translation_accuracy_{export_time}.png")
                pio.write_image(fig_wta, wta_file, scale=2)
                
                # 2. Discriminator Accuracy
                fig_disc = go.Figure()
                fig_disc.add_trace(go.Scatter(
                    x=epoch_x, y=disc_acc, mode='lines+markers', 
                    name='Discriminator Accuracy', line=dict(color='red', width=2)
                ))
                fig_disc.update_layout(
                    title='Discriminator Accuracy',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy (%)',
                    template="plotly_white"
                )
                disc_file = os.path.join(PLOTS_FOLDER, f"discriminator_accuracy_{export_time}.png")
                pio.write_image(fig_disc, disc_file, scale=2)
                
                # 3. Unsupervised Criterion
                fig_unsup = go.Figure()
                fig_unsup.add_trace(go.Scatter(
                    x=epoch_x, y=unsup_criterion, mode='lines+markers', 
                    name='Unsupervised Criterion', line=dict(color='black', width=2)
                ))
                fig_unsup.update_layout(
                    title='Unsupervised Criterion',
                    xaxis_title='Epoch',
                    yaxis_title='Value',
                    template="plotly_white"
                )
                unsup_file = os.path.join(PLOTS_FOLDER, f"unsupervised_criterion_{export_time}.png")
                pio.write_image(fig_unsup, unsup_file, scale=2)
                
                return f"Exported unsupervised charts to {PLOTS_FOLDER} folder at {export_time}"
    
    return "No valid data to export"

def export_alignment_charts(view_type, dict_size=None):
    """Export alignment experiment charts to the plots folder"""
    try:
        with open(ALIGNMENT_METRICS_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "No alignment data available for export"
    
    export_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    # Dictionary size comparison view
    if view_type == 'dict_size':
        # Make sure we have dictionary sizes and metrics
        if ('dictionary_sizes' in data and data['dictionary_sizes'] and
            'precision_at_1_nn' in data and 'precision_at_1_csls' in data and
            'precision_at_5_nn' in data and 'precision_at_5_csls' in data and
            'avg_cosine_similarity' in data and 'orthogonality_error' in data):
            
            # Sort all data by dictionary size for consistent plotting
            size_indices = sorted(range(len(data['dictionary_sizes'])), key=lambda i: data['dictionary_sizes'][i])
            dict_sizes = [str(data['dictionary_sizes'][i]) for i in size_indices]  # Convert to strings for categorical x-axis
            
            # Get precision values
            p1_nn = [data['precision_at_1_nn'][i] for i in size_indices]
            p1_csls = [data['precision_at_1_csls'][i] for i in size_indices]
            p5_nn = [data['precision_at_5_nn'][i] for i in size_indices]
            p5_csls = [data['precision_at_5_csls'][i] for i in size_indices]
            cos_sim = [data['avg_cosine_similarity'][i] for i in size_indices]
            ortho_err = [data['orthogonality_error'][i] for i in size_indices]
            
            # Create and save precision@1 plot
            fig_p1 = go.Figure()
            fig_p1.add_trace(go.Scatter(
                x=dict_sizes, y=p1_nn, mode='lines+markers', name='NN', line=dict(color='blue')
            ))
            fig_p1.add_trace(go.Scatter(
                x=dict_sizes, y=p1_csls, mode='lines+markers', name='CSLS', line=dict(color='red')
            ))
            fig_p1.update_layout(
                title='Precision@1 by Dictionary Size',
                xaxis_title='Dictionary Size',
                yaxis_title='Precision',
                template="plotly_white"
            )
            p1_file = os.path.join(PLOTS_FOLDER, f"precision_at_1_by_dict_size_{export_time}.png")
            pio.write_image(fig_p1, p1_file, scale=2)
            
            # Create and save precision@5 plot
            fig_p5 = go.Figure()
            fig_p5.add_trace(go.Scatter(
                x=dict_sizes, y=p5_nn, mode='lines+markers', name='NN', line=dict(color='blue')
            ))
            fig_p5.add_trace(go.Scatter(
                x=dict_sizes, y=p5_csls, mode='lines+markers', name='CSLS', line=dict(color='red')
            ))
            fig_p5.update_layout(
                title='Precision@5 by Dictionary Size',
                xaxis_title='Dictionary Size',
                yaxis_title='Precision',
                template="plotly_white"
            )
            p5_file = os.path.join(PLOTS_FOLDER, f"precision_at_5_by_dict_size_{export_time}.png")
            pio.write_image(fig_p5, p5_file, scale=2)
            
            # Create and save cosine similarity plot
            fig_cos = go.Figure()
            fig_cos.add_trace(go.Scatter(
                x=dict_sizes, y=cos_sim, mode='lines+markers', line=dict(color='green')
            ))
            fig_cos.update_layout(
                title='Average Cosine Similarity by Dictionary Size',
                xaxis_title='Dictionary Size',
                yaxis_title='Cosine Similarity',
                template="plotly_white"
            )
            cos_file = os.path.join(PLOTS_FOLDER, f"avg_cosine_similarity_by_dict_size_{export_time}.png")
            pio.write_image(fig_cos, cos_file, scale=2)
            
            # Create and save orthogonality error plot
            fig_ortho = go.Figure()
            fig_ortho.add_trace(go.Scatter(
                x=dict_sizes, y=ortho_err, mode='lines+markers', line=dict(color='orange')
            ))
            fig_ortho.update_layout(
                title='Orthogonality Error by Dictionary Size',
                xaxis_title='Dictionary Size',
                yaxis_title='Error',
                template="plotly_white"
            )
            ortho_file = os.path.join(PLOTS_FOLDER, f"orthogonality_error_by_dict_size_{export_time}.png")
            pio.write_image(fig_ortho, ortho_file, scale=2)
            
            # Create and save combined plot
            fig_combined = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Precision@1", "Precision@5", 
                    "Average Cosine Similarity", "Orthogonality Error"
                ),
                shared_xaxes=True
            )
            
            # Add precision@1 traces
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=p1_nn, mode='lines+markers', name='NN',
                        line=dict(color='blue')),
                row=1, col=1
            )
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=p1_csls, mode='lines+markers', name='CSLS',
                        line=dict(color='red')),
                row=1, col=1
            )
            
            # Add precision@5 traces
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=p5_nn, mode='lines+markers', name='NN',
                        line=dict(color='blue')),
                row=1, col=2
            )
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=p5_csls, mode='lines+markers', name='CSLS',
                        line=dict(color='red')),
                row=1, col=2
            )
            
            # Add cosine similarity
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=cos_sim, mode='lines+markers',
                        line=dict(color='green')),
                row=2, col=1
            )
            
            # Add orthogonality error
            fig_combined.add_trace(
                go.Scatter(x=dict_sizes, y=ortho_err, mode='lines+markers',
                        line=dict(color='orange')),
                row=2, col=2
            )
            
            # Update layout
            fig_combined.update_layout(
                title='Word Translation Accuracy by Dictionary Size',
                height=800,
                width=1000,
                template="plotly_white"
            )
            
            # Update axes
            for row in range(1, 3):
                for col in range(1, 3):
                    fig_combined.update_xaxes(title_text="Dictionary Size", type='category', row=row, col=col)
                    fig_combined.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
            
            fig_combined.update_yaxes(title_text="Precision", row=1, col=1)
            fig_combined.update_yaxes(title_text="Precision", row=1, col=2)
            fig_combined.update_yaxes(title_text="Cosine Similarity", row=2, col=1)
            fig_combined.update_yaxes(title_text="Error", row=2, col=2)
            
            combined_file = os.path.join(PLOTS_FOLDER, f"all_metrics_by_dict_size_{export_time}.png")
            pio.write_image(fig_combined, combined_file, scale=2)
            
            return f"Exported dictionary size comparison charts to {PLOTS_FOLDER} folder at {export_time}"
            
    # Refinement steps view
    elif view_type == 'refinement' and dict_size:
        # Find the index for the selected dictionary size
        if 'dictionary_sizes' in data and dict_size:
            try:
                dict_size_int = int(dict_size)
                if dict_size_int in data['dictionary_sizes']:
                    idx = data['dictionary_sizes'].index(dict_size_int)
                    refinement_data = data['refinement_steps'][idx]
                    
                    # Create figure with subplots for different metrics
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=(
                            "Precision@1 (NN)", "Precision@1 (CSLS)", 
                            "Average Cosine Similarity", "Orthogonality Error"
                        ),
                        shared_xaxes=True
                    )
                    
                    # Convert steps to strings for categorical x-axis
                    steps = [str(step) for step in refinement_data['steps']]
                    
                    # Add precision@1 NN
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['precision_at_1_nn'], 
                                   mode='lines+markers', name='Precision@1 (NN)',
                                   line=dict(color='blue')),
                        row=1, col=1
                    )
                    
                    # Add precision@1 CSLS
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['precision_at_1_csls'], 
                                   mode='lines+markers', name='Precision@1 (CSLS)',
                                   line=dict(color='red')),
                        row=1, col=2
                    )
                    
                    # Add cosine similarity
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['avg_cosine_similarity'], 
                                   mode='lines+markers', name='Avg Cosine Similarity',
                                   line=dict(color='green')),
                        row=2, col=1
                    )
                    
                    # Add orthogonality error
                    fig.add_trace(
                        go.Scatter(x=steps, y=refinement_data['orthogonality_error'], 
                                   mode='lines+markers', name='Orthogonality Error',
                                   line=dict(color='orange')),
                        row=2, col=2
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Refinement Steps for Dictionary Size {dict_size}',
                        height=800,
                        width=1000,
                        template="plotly_white"
                    )
                    
                    # Update axes
                    for row in range(1, 3):
                        for col in range(1, 3):
                            fig.update_xaxes(title_text="Refinement Step", type='category', row=row, col=col)
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=row, col=col)
                    
                    fig_combined.update_yaxes(title_text="Precision", row=1, col=1)
                    fig_combined.update_yaxes(title_text="Precision", row=1, col=2)
                    fig_combined.update_yaxes(title_text="Cosine Similarity", row=2, col=1)
                    fig_combined.update_yaxes(title_text="Error", row=2, col=2)
                    
                    combined_file = os.path.join(PLOTS_FOLDER, f"all_metrics_dict_size_{dict_size}_refinement_{export_time}.png")
                    pio.write_image(fig_combined, combined_file, scale=2)
                    
                    return f"Exported refinement charts for dictionary size {dict_size} to {PLOTS_FOLDER} folder at {export_time}"
            except (ValueError, IndexError):
                return f"Error: Could not find or process data for dictionary size {dict_size}"
    
    return "No valid data to export"

# Add callback for export button in unsupervised tab
@app.callback(
    Output('unsupervised-export-status', 'children'),
    [Input('unsupervised-export-button', 'n_clicks')],
    [State('active-tab', 'children')]
)
def export_unsupervised_plots(n_clicks, active_tab):
    if n_clicks > 0 and active_tab == 'unsupervised-tab':
        status = export_unsupervised_charts()
        return status
    return ""

# Add callback for export button in alignment tab
@app.callback(
    Output('alignment-export-status', 'children'),
    [Input('alignment-export-button', 'n_clicks')],
    [State('active-tab', 'children'),
     State('alignment-view-type', 'value'),
     State('dictionary-size-selector', 'value')]
)
def export_alignment_plots(n_clicks, active_tab, view_type, dict_size):
    if n_clicks > 0 and active_tab == 'alignment-tab':
        status = export_alignment_charts(view_type, dict_size)
        return status
    return ""

# Add argparse to customize the app
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MUSE Training Visualization')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--metrics-file', type=str, default=METRICS_FILE, 
                      help='Path to the training metrics JSON file')
    parser.add_argument('--alignment-metrics-file', type=str, default=ALIGNMENT_METRICS_FILE,
                      help='Path to the alignment metrics JSON file')
    parser.add_argument('--plots-folder', type=str, default=PLOTS_FOLDER,
                      help='Folder to save exported plots')
    
    args = parser.parse_args()
    
    # Update file paths if provided
    METRICS_FILE = args.metrics_file
    ALIGNMENT_METRICS_FILE = args.alignment_metrics_file
    PLOTS_FOLDER = args.plots_folder
    
    # Create plots folder if it doesn't exist
    if not os.path.exists(PLOTS_FOLDER):
        os.makedirs(PLOTS_FOLDER)
    
    print(f"Starting Dash server at http://{args.host}:{args.port}/")
    print(f"Watching metrics files:")
    print(f"  - Training metrics: {METRICS_FILE}")
    print(f"  - Alignment metrics: {ALIGNMENT_METRICS_FILE}")
    print(f"Exported plots will be saved to: {PLOTS_FOLDER}")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host=args.host, port=args.port) 