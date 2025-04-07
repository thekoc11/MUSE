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
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np

# Path for reading metrics
METRICS_FILE = 'training_metrics.json'

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("MUSE Training Visualization", style={'textAlign': 'center'}),
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
    ], style={'margin': '10px'}),
    html.Div([
        dcc.Graph(id='metrics-graph', style={'height': '900px'}),
    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # default 5 seconds
        n_intervals=0
    ),
    # Hidden div to store data and view state
    html.Div(id='metrics-store', style={'display': 'none'}),
    html.Div(id='view-state', children='dashboard', style={'display': 'none'})
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
    [Output('metrics-graph', 'figure'),
     Output('last-update-time', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('refresh-button', 'n_clicks'),
     Input('view-state', 'children')]
)
def update_graph(n_intervals, n_clicks, view_mode):
    # Read current metrics
    try:
        with open(METRICS_FILE, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty figure if no data yet
        return make_subplots(rows=2, cols=1), "No data available yet"
    
    # Paper view (single plot like the reference image)
    if view_mode == 'paper':
        fig = go.Figure()
        
        if data['epochs'] and len(data['epochs']) > 0:
            # Filter for integer epochs (full evaluation points)
            is_integer_epoch = [float(e).is_integer() for e in data['epochs']]
            epoch_indices = [i for i, is_int in enumerate(is_integer_epoch) if is_int]
            
            if epoch_indices:
                # Extract epoch-level data
                epoch_x = []
                word_trans_acc = []
                disc_acc = []
                unsup_criterion = []
                
                for i in epoch_indices:
                    # Filter out None values
                    if (data['word_translation_accuracy'][i] is not None and 
                        data['unsupervised_criterion'][i] is not None):
                        epoch_x.append(data['epochs'][i])
                        word_trans_acc.append(data['word_translation_accuracy'][i])
                        unsup_criterion.append(data['unsupervised_criterion'][i])
                        
                        # Include discriminator accuracy
                        if data['discriminator_accuracy'][i] is not None:
                            disc_acc.append(data['discriminator_accuracy'][i])
                        else:
                            disc_acc.append(0)
                
                if epoch_x:
                    # Add vertical grid lines like in the paper
                    for x_val in range(0, int(max(epoch_x)) + 20, 20):
                        fig.add_vline(
                            x=x_val, line_width=1, line_dash="solid", line_color="lightgray"
                        )
                    
                    # Word Translation Accuracy (blue)
                    fig.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=word_trans_acc,
                        mode='lines', 
                        name='Word Translation Accuracy',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Discriminator Accuracy (red)
                    fig.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=disc_acc,
                        mode='lines', 
                        name='Discriminator Accuracy',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Unsupervised Criterion (black)
                    fig.add_trace(go.Scatter(
                        x=epoch_x, 
                        y=unsup_criterion,
                        mode='lines', 
                        name='Unsupervised Criterion',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Layout to match the paper exactly
                    fig.update_layout(
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
                            range=[0, 100],
                            showgrid=False
                        )
                    )
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        return fig, f"Paper view | Last updated: {current_time}"
    
    # Dashboard view (original two subplots)
    else:
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("Iteration-Level Training", "Epoch-Level Evaluation"),
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        # Add traces for each metric
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
            if iter_indices:
                # SUBPLOT 1: Iteration-level metrics (just discriminator accuracy)
                iter_epochs = [data['epochs'][i] for i in iter_indices]
                iter_disc_accu = [data['discriminator_accuracy'][i] for i in iter_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=iter_epochs, 
                        y=iter_disc_accu,
                        mode='lines', 
                        name='Discriminator Cost',
                        line=dict(color='red', width=2),
                    ),
                    row=1, col=1
                )
                
                # Add epoch boundaries as vertical lines
                epoch_boundaries = sorted(set([int(e) for e in data['epochs'] if float(e).is_integer()]))
                for epoch in epoch_boundaries:
                    fig.add_vline(
                        x=epoch, line_width=1, line_dash="dash", line_color="gray",
                        row=1, col=1
                    )
            
            # SUBPLOT 2: Epoch-level evaluation metrics
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
                    # Add vertical grid lines (matching the reference image)
                    for x_val in epoch_x:
                        if float(x_val).is_integer():
                            fig.add_vline(
                                x=x_val, line_width=1, line_dash="solid", line_color="lightgray",
                                row=2, col=1
                            )
                    
                    # Word translation accuracy (blue line)
                    fig.add_trace(
                        go.Scatter(
                            x=epoch_x, 
                            y=word_trans_acc,
                            mode='lines', 
                            name='Word Translation Accuracy',
                            line=dict(color='blue', width=2),
                        ),
                        row=2, col=1
                    )
                    
                    # Discriminator accuracy (red line) - added to match the reference image
                    fig.add_trace(
                        go.Scatter(
                            x=epoch_x, 
                            y=disc_acc,
                            mode='lines', 
                            name='Discriminator Accuracy',
                            line=dict(color='red', width=2),
                        ),
                        row=2, col=1
                    )
                    
                    # Unsupervised criterion (black line)
                    fig.add_trace(
                        go.Scatter(
                            x=epoch_x, 
                            y=unsup_criterion,
                            mode='lines', 
                            name='Unsupervised Criterion',
                            line=dict(color='black', width=2),
                        ),
                        row=2, col=1
                    )
                    
                    # Update subplot title to match reference image
                    for i, annotation in enumerate(fig.layout.annotations):
                        if annotation.text == "Epoch-Level Evaluation":
                            fig.layout.annotations[i].text = "Unsupervised Model Selection"
                            break
            
            # Update layout
            fig.update_layout(
                title='MUSE Training Progress',
                height=900,
                width=None,  # Responsive width
                template="plotly_white",
                font=dict(family="Arial", size=14),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update x and y axis labels
            fig.update_xaxes(title_text="Iteration (Epoch)", row=1, col=1)
            fig.update_yaxes(title_text="Cost / Value", row=1, col=1)
            
            # Match the paper's styling for the second subplot
            fig.update_xaxes(
                title_text="Epoch", 
                row=2, col=1,
                showgrid=True, 
                gridwidth=1, 
                gridcolor='LightGray',
                range=[0, max(epoch_x) + 5] if epoch_x else [0, 100]  # Add padding like in the reference
            )
            fig.update_yaxes(
                title_text="Accuracy (%)", 
                row=2, col=1,
                showgrid=False,  # No horizontal gridlines in the reference
                range=[0, 100]   # Set y-axis range to 0-100 as in reference
            )
            
            # Add grid lines for the first subplot
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=1, col=1)
        
        current_time = time.strftime("%H:%M:%S", time.localtime())
        return fig, f"Dashboard view | Last updated: {current_time}"
    
    return make_subplots(rows=2, cols=1), "No data available yet"

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

# Add argparse to customize the app
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MUSE Training Visualization')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--metrics-file', type=str, default=METRICS_FILE, 
                      help='Path to the metrics JSON file')
    
    args = parser.parse_args()
    
    # Update metrics file path if provided
    METRICS_FILE = args.metrics_file
    
    print(f"Starting Dash server at http://{args.host}:{args.port}/")
    print(f"Watching metrics file: {METRICS_FILE}")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host=args.host, port=args.port) 