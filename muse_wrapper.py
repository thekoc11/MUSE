import os
import argparse
import numpy as np
import torch
import json
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, OrderedDict
import pandas as pd
import random

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.evaluation.word_translation import get_word_translation_accuracy, load_dictionary

# Constants
SOURCE_LANG = 'en'
TARGET_LANG = 'hi'
DICTIONARY_SIZES = [5000, 10000, 20000]
MUSE_DATA_PATH = './data/dictionaries'

def verify_orthogonality(W):
    """
    Verify that the mapping matrix W is orthogonal
    (W^T * W should be close to identity matrix)
    """
    # For an orthogonal matrix W, W^T * W should equal identity matrix
    WtW = torch.mm(W.t(), W)
    I = torch.eye(W.shape[1], device=W.device)
    error = torch.norm(WtW - I).item()
    is_orthogonal = error < 1e-4  
    return is_orthogonal, error

def load_combined_dictionaries(src_lang, tgt_lang, dico_path, word2id1, word2id2, exclude_validation=True):
    """
    Load and combine multiple dictionary files to create a larger pool of word pairs.
    Optionally exclude validation pairs to prevent data leakage.
    """
    # Potential dictionary files to combine
    dictionary_files = [
        f'{src_lang}-{tgt_lang}.0-5000.txt',  # Forward dictionary
        # f'{tgt_lang}-{src_lang}.0-5000.txt',  # Reverse dictionary (swapped)
        # f'{src_lang}-{tgt_lang}.txt',         # Full forward dictionary if exists
        # f'{tgt_lang}-{src_lang}.txt'          # Full reverse dictionary if exists
    ]
    
    # Validation dictionary to exclude
    validation_path = os.path.join(dico_path, f'{src_lang}-{tgt_lang}.5000-6500.txt')
    
    # Load validation pairs to exclude
    validation_pairs = set()
    if exclude_validation and os.path.exists(validation_path):
        with open(validation_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().lower().split()
                if len(parts) >= 2:
                    validation_pairs.add((parts[0], parts[1]))
    
    # Load and combine dictionaries
    combined_pairs = set()
    
    for dict_file in dictionary_files:
        dict_path = os.path.join(dico_path, dict_file)
        if os.path.exists(dict_path):
            # Check if we need to swap src/tgt for reverse dictionaries
            is_reverse = dict_file.startswith(f'{tgt_lang}-{src_lang}')
            
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().lower().split()
                    if len(parts) >= 2:
                        # For reverse dictionaries, swap word order
                        if is_reverse:
                            word_pair = (parts[1], parts[0])
                        else:
                            word_pair = (parts[0], parts[1])
                            
                        # Only add if not in validation set
                        if word_pair not in validation_pairs:
                            combined_pairs.add(word_pair)
    
    # Convert to list of pairs that exist in our vocabulary
    valid_pairs = []
    for src_word, tgt_word in combined_pairs:
        if src_word in word2id1 and tgt_word in word2id2:
            valid_pairs.append((word2id1[src_word], word2id2[tgt_word]))
    
    # Create tensor with word ids
    if valid_pairs:
        dico = torch.tensor(valid_pairs, dtype=torch.long)
        print(f"Combined dictionary created with {len(dico)} pairs (filtered from {len(combined_pairs)} total pairs)")
    else:
        raise RuntimeError("No valid word pairs found in dictionaries!")
    
    return dico

def prepare_dictionary(word2id1, word2id2, params, dico_size):
    """
    Prepare a dictionary of a specific size while ensuring validation pairs are excluded.
    """
    # Get combined dictionary excluding validation pairs
    full_dico = load_combined_dictionaries(
        params.src_lang,
        params.tgt_lang,
        params.dico_path,
        word2id1,
        word2id2
    )
    
    # If the requested size is larger than available, return the full dictionary
    if dico_size >= len(full_dico):
        print(f"Warning: Requested dictionary size {dico_size} exceeds available pairs ({len(full_dico)}). Using all available pairs.")
        return full_dico
    
    # Otherwise, randomly sample the requested number of pairs
    indices = torch.randperm(len(full_dico))[:dico_size]
    return full_dico[indices]

def compute_cosine_similarity(src_vectors, tgt_vectors, dico):
    """
    Compute cosine similarity between source and target word pairs in the dictionary
    """
    cos_similarities = []
    for i in range(len(dico)):
        src_id, tgt_id = dico[i][0].item(), dico[i][1].item()
        src_vec = src_vectors[src_id]
        tgt_vec = tgt_vectors[tgt_id]
        
        # Normalize vectors
        src_vec = src_vec / src_vec.norm()
        tgt_vec = tgt_vec / tgt_vec.norm()
        
        # Compute cosine similarity
        cos_sim = torch.dot(src_vec, tgt_vec).item()
        cos_similarities.append(cos_sim)
    
    return cos_similarities

def run_alignment_experiment(dico_size, params):
    """
    Run supervised alignment experiment with a specific dictionary size
    """
    # Build model, trainer and evaluator
    src_emb, tgt_emb, mapping, _ = build_model(params, False)
    trainer = Trainer(src_emb, tgt_emb, mapping, None, params)
    evaluator = Evaluator(trainer)
    
    # Get logger
    from logging import getLogger
    logger = getLogger()
    
    # Create a training dictionary of specified size
    trainer.dico = prepare_dictionary(
        params.src_dico.word2id, 
        params.tgt_dico.word2id,
        params,
        dico_size
    )
    
    if params.cuda:
        trainer.dico = trainer.dico.cuda()
    
    # Apply Procrustes alignment with refinement iterations
    for n_iter in range(params.n_refinement + 1):
        logger.info(f'Starting refinement iteration {n_iter} for dictionary size {dico_size}...')
        
        # For iterations beyond the first, build a new dictionary from alignments
        if n_iter > 0:
            trainer.build_dictionary()
            logger.info(f'Built new dictionary of size {len(trainer.dico)}')
        
        # Apply Procrustes solution
        trainer.procrustes()
        
        # Verify orthogonality
        W = trainer.mapping.weight.data
        is_orthogonal, ortho_error = verify_orthogonality(W)
        logger.info(f"Refinement iteration {n_iter}, is mapping orthogonal: {is_orthogonal}, Error: {ortho_error:.8f}")
        
        # If not orthogonal enough, enforce orthogonality
        if not is_orthogonal and hasattr(params, 'map_beta') and params.map_beta > 0:
            print(f"Initial orthogonality error: {ortho_error:.8f}, applying orthogonalization")
            
            # Apply orthogonalization repeatedly until desired tolerance is reached
            max_iterations = 5
            for i in range(max_iterations):
                trainer.orthogonalize()
                is_orthogonal, ortho_error = verify_orthogonality(W)
                print(f"Iteration {i+1}, orthogonality error: {ortho_error:.8f}")
                
                if is_orthogonal:
                    print(f"Achieved orthogonality after {i+1} iterations")
                    break
            
            if not is_orthogonal:
                print(f"Warning: Could not achieve perfect orthogonality after {max_iterations} iterations")
                print(f"Final error: {ortho_error:.8f}")
    
    # Create an OrderedDict to store all results
    to_log = OrderedDict()
    to_log['dictionary_size'] = dico_size
    to_log['orthogonality_error'] = ortho_error
    to_log['is_orthogonal'] = is_orthogonal
    
    # Get mapped embeddings for custom evaluations that the built-in evaluator doesn't provide
    src_emb_mapped = trainer.mapping(trainer.src_emb.weight).data
    tgt_emb = trainer.tgt_emb.weight.data
    
    # Compute cosine similarities for word pairs (custom metric)
    eval_dico = load_dictionary(
        params.dico_eval,
        params.src_dico.word2id, 
        params.tgt_dico.word2id
    )
    
    cos_similarities = compute_cosine_similarity(
        src_emb_mapped, 
        tgt_emb, 
        eval_dico
    )
    
    avg_cos_sim = np.mean(cos_similarities)
    to_log['avg_cosine_similarity'] = avg_cos_sim
    
    # Use the built-in evaluator for word translation evaluation
    # This internally calls get_word_translation_accuracy for both 'nn' and 'csls_knn_10' methods
    # and computes precision@1, precision@5, and precision@10
    evaluator.word_translation(to_log)
    
    # Convert OrderedDict to regular dict for easier handling later
    results = dict(to_log)
    
    return results

def visualize_results(all_results):
    """
    Create visualizations of the results using plotly
    """
    # Convert results to DataFrame for easier manipulation
    df = pd.DataFrame(all_results)
    
    # Create precision plot
    fig1 = make_subplots(rows=1, cols=2, 
                         subplot_titles=['Precision@1', 'Precision@5'],
                         shared_yaxes=True)
    
    # Add precision@1 data
    fig1.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['precision_at_1-nn'], 
                  mode='lines+markers', name='Nearest Neighbor'),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['precision_at_1-csls_knn_10'], 
                  mode='lines+markers', name='CSLS (k=10)'),
        row=1, col=1
    )
    
    # Add precision@5 data
    fig1.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['precision_at_5-nn'], 
                  mode='lines+markers', name='Nearest Neighbor'),
        row=1, col=2
    )
    
    fig1.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['precision_at_5-csls_knn_10'], 
                  mode='lines+markers', name='CSLS (k=10)'),
        row=1, col=2
    )
    
    fig1.update_layout(
        title='Word Translation Accuracy by Dictionary Size',
        xaxis_title='Dictionary Size',
        yaxis_title='Precision (%)',
        legend_title='Method',
        height=500,
        width=900
    )
    
    # Create cosine similarity and orthogonality plot
    fig2 = make_subplots(rows=1, cols=2, 
                         subplot_titles=['Average Cosine Similarity', 'Orthogonality Error'],
                         shared_xaxes=True)
    
    fig2.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['avg_cosine_similarity'], 
                  mode='lines+markers'),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=df['dictionary_size'], y=df['orthogonality_error'], 
                  mode='lines+markers'),
        row=1, col=2
    )
    
    fig2.update_layout(
        title='Mapping Quality Metrics by Dictionary Size',
        xaxis_title='Dictionary Size',
        yaxis_title='Value',
        height=500,
        width=900
    )
    
    # Save figures
    fig1.write_html('precision_results.html')
    fig2.write_html('mapping_quality.html')
    
    print("Visualization files saved: precision_results.html and mapping_quality.html")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Cross-lingual word embedding alignment for Hindi')
    
    # Main experiment parameters (matching supervised.py)
    parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
    parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
    parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
    parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")
    
    # Data parameters
    parser.add_argument("--src_lang", type=str, default=SOURCE_LANG, help="Source language")
    parser.add_argument("--tgt_lang", type=str, default=TARGET_LANG, help="Target language")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--max_vocab", type=int, default=100000, help="Maximum vocabulary size (-1 to disable)")
    
    # Training refinement
    parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable)")
    
    # Dictionary parameters
    parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary")
    parser.add_argument("--dico_path", type=str, default=MUSE_DATA_PATH, help="Path to dictionary directory")
    parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
    parser.add_argument("--dico_method", type=str, default='csls_knn_15', help="Method for dictionary generation")
    parser.add_argument("--dico_build", type=str, default='S2T|T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
    parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
    parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
    parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
    parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
    
    # Reload pre-trained embeddings
    parser.add_argument("--src_emb", type=str, required=True, help="Source embeddings path")
    parser.add_argument("--tgt_emb", type=str, required=True, help="Target embeddings path")
    parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")
    
    # For orthogonalization
    parser.add_argument("--map_beta", type=float, default=0.01, help="Beta for orthogonalization")
    
    params = parser.parse_args()
    
    # Initialize experiment
    logger = initialize_exp(params)
    
    # Handle default dictionary paths
    if params.dico_eval == 'default':
        params.dico_eval = os.path.join(params.dico_path, '%s-%s.5000-6500.txt' % (params.src_lang, params.tgt_lang))
        logger.info(f"Using default evaluation dictionary: {params.dico_eval}")
    
    # Run experiments with different dictionary sizes
    results = []
    for dico_size in DICTIONARY_SIZES:
        result = run_alignment_experiment(dico_size, params)
        results.append(result)
        
        # Log results
        logger.info("=" * 50)
        logger.info(f"Results for dictionary size {dico_size}:")
        for metric, value in result.items():
            logger.info(f"{metric}: {value}")
        logger.info("=" * 50)
    
    # Save all results
    results_path = os.path.join(params.exp_path, params.exp_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    with open(os.path.join(results_path, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Visualize results
    visualize_results(results)

if __name__ == '__main__':
    main() 