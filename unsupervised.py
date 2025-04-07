# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import json
import argparse
from collections import OrderedDict
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from datetime import datetime
# For static image export
try:
    import kaleido
except ImportError:
    print("Warning: kaleido package not found. Static image export may not work.")
    print("Install with: pip install kaleido")

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator


VALIDATION_METRIC = 'mean_cosine-csls_knn_10-S2T-10000'

# Store metrics during training for visualization
training_metrics = {
    'word_translation_accuracy': [],
    'discriminator_accuracy': [],
    'discriminator_loss': [],
    'unsupervised_criterion': [],
    'epochs': []
}

# Path for saving metrics
METRICS_FILE = 'training_metrics.json'

# Function to save metrics to a file
def save_metrics():
    with open(METRICS_FILE, 'w') as f:
        json.dump(training_metrics, f)

# Function to update metrics during iterations (for more frequent updates)
def update_iteration_metrics(n_epoch, n_iter, stats):
    # Use fractional epochs for x-axis to show iterations within an epoch
    # For example, epoch 2, iteration 1000 out of 10000 would be 2.1
    if 'epoch_size' in params and params.epoch_size > 0:
        fractional_epoch = n_epoch + (n_iter / params.epoch_size)
    else:
        fractional_epoch = n_epoch + (n_iter / 10000)  # default if epoch_size not set
    
    training_metrics['epochs'].append(fractional_epoch)
    
    # We only log actual discriminator accuracy and unsupervised criterion at epoch end.
    # For consistency during iterations, we append the loss or None for accuracy/criterion.
    
    # Add discriminator loss 
    if 'DIS_COSTS' in stats and stats['DIS_COSTS']:
        dis_cost = np.mean(stats['DIS_COSTS'])
        training_metrics['discriminator_loss'].append(dis_cost)
    else:
        training_metrics['discriminator_loss'].append(None) # Append None if no cost data
    
    # Discriminator accuracy - use last known value
    training_metrics['discriminator_accuracy'].append(None)
    # Word translation accuracy - use last known value
    training_metrics['word_translation_accuracy'].append(None)
    # Unsupervised criterion - use last known value
    training_metrics['unsupervised_criterion'].append(None)
    
    # Save to file for Dash app
    save_metrics()

# Function to create and save a static plot
def save_static_plot(output_dir=None):
    if not output_dir:
        output_dir = "plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each metric
    if training_metrics['epochs'] and len(training_metrics['epochs']) > 0:
        # Word translation accuracy (blue line)
        fig.add_trace(go.Scatter(
            x=training_metrics['epochs'], 
            y=training_metrics['word_translation_accuracy'],
            mode='lines', 
            name='Word Translation Accuracy',
            line=dict(color='blue')
        ))
        
        # Discriminator accuracy (red line)
        fig.add_trace(go.Scatter(
            x=training_metrics['epochs'], 
            y=training_metrics['discriminator_accuracy'],
            mode='lines', 
            name='Discriminator Accuracy',
            line=dict(color='red', width=1, dash='dot')
        ))
        
        # Unsupervised criterion (black line)
        fig.add_trace(go.Scatter(
            x=training_metrics['epochs'], 
            y=training_metrics['unsupervised_criterion'],
            mode='lines', 
            name='Unsupervised Criterion',
            line=dict(color='black')
        ))
        
        # Update layout to match paper style
        fig.update_layout(
            title='Unsupervised Model Selection',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            legend=dict(x=0.7, y=0.1),
            height=600,
            width=800,
            template="plotly_white",
            font=dict(family="Arial", size=14)
        )
        
        # Add vertical grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"training_plot_{timestamp}.html")
        fig.write_html(filename)
        
        # Also save as static image
        img_filename = os.path.join(output_dir, f"training_plot_{timestamp}.png")
        fig.write_image(img_filename)
        
        return filename
    
    return None

# main
parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="pth", help="Export embeddings after training (txt / pth)")
# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# mapping
parser.add_argument("--map_id_init", type=bool_flag, default=True, help="Initialize the mapping as an identity matrix")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
# discriminator
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0., help="Discriminator dropout")
parser.add_argument("--dis_input_dropout", type=float, default=0.1, help="Discriminator input dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--dis_lambda", type=float, default=1, help="Discriminator loss feedback coefficient")
parser.add_argument("--dis_most_frequent", type=int, default=75000, help="Select embeddings of the k most frequent words for discrimination (0 to disable)")
parser.add_argument("--dis_smooth", type=float, default=0.1, help="Discriminator smooth predictions")
parser.add_argument("--dis_clip_weights", type=float, default=0, help="Clip discriminator weights (0 to disable)")
# training adversarial
parser.add_argument("--adversarial", type=bool_flag, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--epoch_size", type=int, default=1000000, help="Iterations per epoch")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.05", help="Discriminator optimizer")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--lr_shrink", type=float, default=0.5, help="Shrink the learning rate if the validation metric decreases (1 to disable)")
parser.add_argument("--early_stopping_patience", type=int, default=0, help="Stop training if validation metric doesn't improve for this many epochs (0 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_15', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T|T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=15000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--dico_train", type=str, default="identical_char", help="Training dictionary type (identical_char/default/custom_seed)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
# visualization params
parser.add_argument("--plot_results", type=bool_flag, default=True, help="Plot training progress")
parser.add_argument("--plot_output_dir", type=str, default="plots", help="Directory to save plots")
parser.add_argument("--eval_frequency", type=int, default=500, help="Frequency of evaluation during training")
# evaluation dictionary handling
parser.add_argument("--skip_validation", type=bool_flag, default=False, help="Skip validation if dictionary not found")
# final alignment visualization params
parser.add_argument("--visualize_final_alignment", type=bool_flag, default=True, help="Visualize word pair alignment after training")
parser.add_argument("--viz_dictionary", type=str, default="", help="Path to dictionary for visualization (if different from dico_eval)")
parser.add_argument("--viz_max_pairs", type=int, default=200, help="Maximum number of pairs for visualization")
parser.add_argument("--viz_output_dir", type=str, default="plots", help="Directory to save visualization plots")
# option to skip training and load a pre-trained model directly
parser.add_argument("--pretrained_mapping", type=str, default="", help="Path to a pre-trained mapping model (.pth file) to skip training and go straight to export")
parser.add_argument("--skip_training", type=bool_flag, default=False, help="Skip training and use pre-trained model")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert 0 <= params.dis_dropout < 1
assert 0 <= params.dis_input_dropout < 1
assert 0 <= params.dis_smooth < 0.5
assert params.dis_lambda > 0 and params.dis_steps > 0
assert 0 < params.lr_shrink <= 1
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# if using pretrained mapping, check that it exists
if params.skip_training and params.pretrained_mapping:
    assert os.path.isfile(params.pretrained_mapping), f"Pretrained mapping file not found: {params.pretrained_mapping}"

# build model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, discriminator = build_model(params, True)
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)
evaluator = Evaluator(trainer)

# Remove Dash app creation and threading
# Just initialize metrics file if visualizing
if params.plot_results:
    # Initialize metrics file
    save_metrics()
    logger.info('Metrics will be saved to %s for visualization' % METRICS_FILE)
    logger.info('Run the dash_viewer.py script to view training progress')


"""
Learning loop for Adversarial Training
"""
if params.adversarial and not params.skip_training:
    logger.info('----> ADVERSARIAL TRAINING <----\n\n')

    best_valid_metric = -float('inf')
    patience_counter = 0

    # training loop
    for n_epoch in range(params.n_epochs):

        logger.info('Starting adversarial training epoch %i...' % n_epoch)
        tic = time.time()
        n_words_proc = 0
        stats = {'DIS_COSTS': []}

        for n_iter in range(0, params.epoch_size, params.batch_size):

            # discriminator training
            for _ in range(params.dis_steps):
                trainer.dis_step(stats)

            # mapping training (discriminator fooling)
            n_words_proc += trainer.mapping_step(stats)

            # log stats
            if n_iter % 500 == 0:
                stats_str = [('DIS_COSTS', 'Discriminator loss')]
                stats_log = ['%s: %.4f' % (v, np.mean(stats[k]))
                             for k, v in stats_str if len(stats[k]) > 0]
                stats_log.append('%i samples/s' % int(n_words_proc / (time.time() - tic)))
                logger.info(('%06i - ' % n_iter) + ' - '.join(stats_log))
                
                # Update and save metrics more frequently if plotting is enabled
                if params.plot_results:
                    update_iteration_metrics(n_epoch, n_iter, stats)

                # reset
                tic = time.time()
                n_words_proc = 0
                for k, _ in stats_str:
                    del stats[k][:]

        # embeddings / discriminator evaluation
        to_log = OrderedDict({'n_epoch': n_epoch})
        evaluator.all_eval(to_log)
        evaluator.eval_dis(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of epoch %i.\n\n' % n_epoch)
        
        # Update metrics for visualization
        if params.plot_results:
            # Print available metrics for debugging
            logger.info("Available metrics: %s" % list(to_log.keys()))
            
            # Append None for discriminator loss at epoch end
            training_metrics['discriminator_loss'].append(None)
            
            # Create a clean integer epoch entry for epoch boundaries
            training_metrics['epochs'].append(float(n_epoch + 1))
            
            # Key for actual word translation accuracy P@1 using CSLS
            WORD_TRANSLATION_METRIC_KEY = 'precision_at_1-csls_knn_10'
            
            # Word translation accuracy - use the specific precision metric
            if WORD_TRANSLATION_METRIC_KEY in to_log:
                training_metrics['word_translation_accuracy'].append(to_log[WORD_TRANSLATION_METRIC_KEY]) # Already percentage
            else:
                logger.warning(f"{WORD_TRANSLATION_METRIC_KEY} not found in logs for epoch {n_epoch}. Using previous value or 0.")
                training_metrics['word_translation_accuracy'].append(
                    training_metrics['word_translation_accuracy'][-1] if training_metrics['word_translation_accuracy'] else 0
                )
            
            # Discriminator accuracy
            if 'dis_accu' in to_log:
                training_metrics['discriminator_accuracy'].append(to_log['dis_accu'] * 100) # Convert to percentage
            else:
                logger.warning(f"dis_accu not found in logs for epoch {n_epoch}. Using previous value or 0.")
                training_metrics['discriminator_accuracy'].append(
                    training_metrics['discriminator_accuracy'][-1] if training_metrics['discriminator_accuracy'] else 0
                )
            
            # Unsupervised criterion - use the VALIDATION_METRIC (mean cosine similarity)
            # Multiply by 100 to make it a percentage-like scale for plotting
            if VALIDATION_METRIC in to_log:
                training_metrics['unsupervised_criterion'].append(to_log[VALIDATION_METRIC] * 100) 
            else:
                logger.warning(f"{VALIDATION_METRIC} not found in logs for epoch {n_epoch}. Using previous value or 0.")
                training_metrics['unsupervised_criterion'].append(
                    training_metrics['unsupervised_criterion'][-1] if training_metrics['unsupervised_criterion'] else 0
                )
            
            # Save metrics to file for Dash app
            save_metrics()

        # update the learning rate (stop if too small)
        trainer.update_lr(to_log, VALIDATION_METRIC)
        if trainer.map_optimizer.param_groups[0]['lr'] < params.min_lr:
            logger.info('Learning rate < 1e-6. BREAK.')
            break

        # Check for early stopping
        if params.early_stopping_patience > 0:
            current_valid_metric = to_log.get(VALIDATION_METRIC, -float('inf'))
            if current_valid_metric > best_valid_metric:
                best_valid_metric = current_valid_metric
                patience_counter = 0
                logger.info(f"Validation metric improved to {best_valid_metric:.4f}. Resetting patience.")
            else:
                patience_counter += 1
                logger.info(f"Validation metric did not improve ({current_valid_metric:.4f} vs best {best_valid_metric:.4f}). Patience: {patience_counter}/{params.early_stopping_patience}")
                if patience_counter >= params.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                    break


"""
Learning loop for Procrustes Iterative Refinement
"""
if params.n_refinement > 0 and not params.skip_training:
    # Get the best mapping according to VALIDATION_METRIC
    logger.info('----> ITERATIVE PROCRUSTES REFINEMENT <----\n\n')
    trainer.reload_best()

    # training loop
    for n_iter in range(params.n_refinement):

        logger.info('Starting refinement iteration %i...' % n_iter)

        # build a dictionary from aligned embeddings
        trainer.build_dictionary()

        # apply the Procrustes solution
        trainer.procrustes()

        # embeddings evaluation
        to_log = OrderedDict({'n_iter': n_iter})
        evaluator.all_eval(to_log)

        # JSON log / save best model / end of epoch
        logger.info("__log__:%s" % json.dumps(to_log))
        trainer.save_best(to_log, VALIDATION_METRIC)
        logger.info('End of refinement iteration %i.\n\n' % n_iter)
        
        # Update metrics for visualization (refinement phase)
        if params.plot_results:
            # Print available metrics for debugging
            logger.info("Refinement metrics: %s" % list(to_log.keys()))
            
            # Append None for discriminator loss during refinement
            training_metrics['discriminator_loss'].append(None)
            
            # For refinement, we'll add to the same epochs list
            # Use integer epochs to clearly distinguish between phases
            last_epoch = int(max(training_metrics['epochs'])) if training_metrics['epochs'] else 0
            current_epoch = last_epoch + n_iter + 1
            training_metrics['epochs'].append(float(current_epoch))
            
            # Key for actual word translation accuracy P@1 using CSLS
            WORD_TRANSLATION_METRIC_KEY = 'precision_at_1-csls_knn_10'
            
            # Word translation accuracy
            if WORD_TRANSLATION_METRIC_KEY in to_log:
                training_metrics['word_translation_accuracy'].append(to_log[WORD_TRANSLATION_METRIC_KEY]) # Already percentage
            else:
                logger.warning(f"{WORD_TRANSLATION_METRIC_KEY} not found in logs for refinement iter {n_iter}. Using previous value or 0.")
                training_metrics['word_translation_accuracy'].append(
                    training_metrics['word_translation_accuracy'][-1] if training_metrics['word_translation_accuracy'] else 0
                )
            
            # Discriminator accuracy (usually not applicable/logged in refinement, use fallback)
            if 'dis_accu' in to_log:
                training_metrics['discriminator_accuracy'].append(to_log['dis_accu'] * 100) # Convert to percentage
            else:
                # Use previous value or a placeholder (e.g., last known value or 0)
                training_metrics['discriminator_accuracy'].append(
                    training_metrics['discriminator_accuracy'][-1] if training_metrics['discriminator_accuracy'] else 0
                )
            
            # Unsupervised criterion - use the VALIDATION_METRIC (mean cosine similarity)
            # Multiply by 100 to make it a percentage-like scale for plotting
            if VALIDATION_METRIC in to_log:
                training_metrics['unsupervised_criterion'].append(to_log[VALIDATION_METRIC] * 100)
            else:
                logger.warning(f"{VALIDATION_METRIC} not found in logs for refinement iter {n_iter}. Using previous value or 0.")
                training_metrics['unsupervised_criterion'].append(
                    training_metrics['unsupervised_criterion'][-1] if training_metrics['unsupervised_criterion'] else 0
                )
            
            # Save metrics to file for Dash app
            save_metrics()


# Load the pre-trained model if specified
if params.skip_training and params.pretrained_mapping:
    logger.info('----> LOADING PRE-TRAINED MODEL <----\n\n')
    logger.info(f'Loading pre-trained mapping from: {params.pretrained_mapping}')
    
    # Load the model directly into the mapping
    assert os.path.isfile(params.pretrained_mapping), f"Cannot find mapping file: {params.pretrained_mapping}"
    to_reload = torch.from_numpy(torch.load(params.pretrained_mapping))
    W = trainer.mapping.weight.data
    assert to_reload.size() == W.size(), f"Model size mismatch: {to_reload.size()} vs {W.size()}"
    W.copy_(to_reload.type_as(W))
    
    logger.info(f'Pre-trained mapping loaded successfully')
    
    # Optionally evaluate the model
    to_log = OrderedDict({'pretrained': True})
    evaluator.all_eval(to_log)
    logger.info("__log__:%s" % json.dumps(to_log))
elif params.skip_training:
    # If skip_training is True but no model path is provided, reload the best model from the current experiment path
    logger.info('----> RELOADING BEST MODEL <----\n\n')
    trainer.reload_best()

# export embeddings
if params.export:
    if not params.skip_training:
        trainer.reload_best()
    # trainer.export() # Original export call
    # Call export and capture the returned embeddings and dictionaries
    mapped_src_emb_tensor, normalized_tgt_emb_tensor, src_dico, tgt_dico = trainer.export()
    
    # Visualize final alignment if requested
    if params.visualize_final_alignment:
        logger.info("Preparing data for final alignment visualization...")
        try:
            from visualize_aligned_embeddings_pairs import visualize_aligned_pairs
            mapped_src_emb_tensor = mapped_src_emb_tensor.cpu()
            normalized_tgt_emb_tensor = normalized_tgt_emb_tensor.cpu()
            # Convert tensors and Dico to the required dictionary format
            src_vectors = {word: mapped_src_emb_tensor[i].numpy() for i, word in src_dico.id2word.items() if i < len(mapped_src_emb_tensor)}
            tgt_vectors = {word: normalized_tgt_emb_tensor[i].numpy() for i, word in tgt_dico.id2word.items() if i < len(normalized_tgt_emb_tensor)}
            
            # Determine visualization dictionary path
            viz_dict_path = params.viz_dictionary
            if not viz_dict_path or not os.path.isfile(viz_dict_path):
                logger.warning(f"Visualization dictionary '{viz_dict_path}' not found or not specified. Falling back to evaluation dictionary: '{params.dico_eval}'")
                if params.dico_eval == 'default':
                    # Construct default path if needed
                    default_dict_file = f"{params.src_lang}-{params.tgt_lang}.0-5000.txt"
                    viz_dict_path = os.path.join("data", "dictionaries", default_dict_file)
                else:
                    viz_dict_path = params.dico_eval
            
            if not os.path.isfile(viz_dict_path):
                logger.error(f"Could not find a valid dictionary for visualization: '{viz_dict_path}'. Skipping visualization.")
            else:
                # Prepare output path
                if not os.path.exists(params.viz_output_dir):
                    os.makedirs(params.viz_output_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                viz_output_file = os.path.join(params.viz_output_dir, f"aligned_pairs_{params.src_lang}-{params.tgt_lang}_{timestamp}.html")
                
                # Call the visualization function
                visualize_aligned_pairs(
                    src_lang=params.src_lang,
                    tgt_lang=params.tgt_lang,
                    src_vectors=src_vectors, 
                    tgt_vectors=tgt_vectors,
                    dict_path=viz_dict_path,
                    output_path=viz_output_file,
                    max_pairs=params.viz_max_pairs
                )
                logger.info(f"Final alignment visualization saved to {viz_output_file}")

        except ImportError:
            logger.error("Could not import 'visualize_aligned_pairs'. Skipping visualization. Ensure 'visualize_aligned_embeddings_pairs.py' is accessible.")
        except Exception as e:
            logger.error(f"An error occurred during final alignment visualization: {e}")

# Save static plot at the end of training
if params.plot_results and len(training_metrics['epochs']) > 0:
    plot_path = save_static_plot(params.plot_output_dir)
    if plot_path:
        logger.info(f"Static training plot saved to {plot_path}")
