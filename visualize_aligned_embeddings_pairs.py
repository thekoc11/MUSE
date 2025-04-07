import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import logging
import argparse
import torch
from sklearn.manifold import TSNE
import fasttext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_vectors(path, max_vocab=100000):
    """
    Load word vectors from a .vec file
    Returns a dictionary mapping words to their vectors
    Only loads the top max_vocab words (most frequent ones)
    """
    logging.info(f"Loading vectors from {path} (limiting to top {max_vocab} words)")
    
    vectors = {}
    dim = None
    with open(path, 'r', encoding='utf-8') as f:
        # First line contains vocab size and dimension
        line = f.readline()
        parts = line.strip().split()
        vocab_size, dim = int(parts[0]), int(parts[1])
        
        logging.info(f"File vocabulary size: {vocab_size}, Vector dimension: {dim}")
        
        # Read only up to max_vocab words
        for i, line in enumerate(f):
            if i >= max_vocab:
                break
                
            if i % 10000 == 0:
                logging.info(f"Read {i} vectors")
            
            parts = line.strip().split(' ')
            word = parts[0]
            
            # Ensure consistent vector dimensions
            vector_parts = parts[1:]
            if len(vector_parts) != dim:
                logging.warning(f"Vector for word '{word}' has {len(vector_parts)} dimensions, expected {dim}")
                # Skip inconsistent vectors
                continue
                
            try:
                vector = np.array([float(x) for x in vector_parts])
                vectors[word] = vector
            except ValueError:
                logging.warning(f"Could not convert vector for word '{word}'")
                continue
    
    logging.info(f"Loaded {len(vectors)} vectors of dimension {dim}")
    return vectors, dim

def apply_mapping(src_vectors, mapping_file, src_dim):
    """
    Apply alignment mapping to source vectors
    
    Args:
        src_vectors: Dictionary of source language vectors
        mapping_file: Path to the mapping file (.pth format)
        src_dim: Dimension of source vectors
    
    Returns:
        Dictionary of mapped source vectors
    """
    logging.info(f"Applying mapping from {mapping_file}")
    
    # Load the mapping matrix
    W = torch.load(mapping_file)
    logging.info(f"Loaded mapping matrix of shape {W.shape}")
    
    # Convert mapping to numpy for easier manipulation
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        logging.info(f"Loaded object is of type {type(W)}")
        W_np = np.array(W)
    
    logging.info(f"Converted mapping to numpy array of shape {W_np.shape}")
    
    # Check dimensions
    mapping_input_dim = W_np.shape[0]
    mapping_output_dim = W_np.shape[1]
    
    logging.info(f"Vector dimension: {src_dim}, Mapping input dimension: {mapping_input_dim}")
    
    # Validate all source vectors have the same dimension
    all_vecs_list = []
    valid_words = []
    
    for word, vec in src_vectors.items():
        if len(vec) == src_dim:
            all_vecs_list.append(vec)
            valid_words.append(word)
        else:
            logging.warning(f"Skipping word '{word}' with invalid dimension: {len(vec)}")
    
    # Convert to numpy array only after validating dimensions
    all_vecs = np.array(all_vecs_list)
    
    # Center the embeddings (subtract mean)
    mean_vec = np.mean(all_vecs, axis=0)
    logging.info(f"Centering source vectors by subtracting mean")
    
    # Apply mapping to each vector following MUSE procedure
    mapped_vectors = {}
    for i, word in enumerate(valid_words):
        vec = src_vectors[word]
        
        # Center the vector
        vec_centered = vec - mean_vec
        
        # Adjust vector dimension if needed
        if len(vec_centered) != mapping_input_dim:
            if len(vec_centered) < mapping_input_dim:
                # Pad with zeros
                vec_adjusted = np.pad(vec_centered, (0, mapping_input_dim - len(vec_centered)), 'constant')
            else:
                # Truncate
                vec_adjusted = vec_centered[:mapping_input_dim]
        else:
            vec_adjusted = vec_centered
        
        # Apply mapping
        mapped_vec = vec_adjusted @ W_np
        
        # Normalize to unit length
        norm = np.linalg.norm(mapped_vec)
        if norm > 0:
            mapped_vectors[word] = mapped_vec / norm
        else:
            mapped_vectors[word] = mapped_vec
    
    return mapped_vectors

def load_dictionary(dict_path, src_word2id=None, tgt_word2id=None):
    """
    Load a bilingual dictionary file
    
    Args:
        dict_path: Path to the dictionary file
        src_word2id: Source language word to ID mapping (optional)
        tgt_word2id: Target language word to ID mapping (optional)
    
    Returns:
        List of word pairs (src_word, tgt_word)
    """
    word_pairs = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src_word, tgt_word = parts[0], parts[1]
                
                # Filter pairs if word2id mappings are provided
                if src_word2id is not None and tgt_word2id is not None:
                    if src_word in src_word2id and tgt_word in tgt_word2id:
                        word_pairs.append((src_word, tgt_word))
                else:
                    word_pairs.append((src_word, tgt_word))
    
    return word_pairs

def visualize_aligned_pairs(src_lang, tgt_lang, src_vectors, tgt_vectors, dict_path, output_path, 
                           max_pairs=500, random_seed=42):
    """
    Create a visualization showing alignment between word pairs
    
    Args:
        src_lang: Source language name
        tgt_lang: Target language name
        src_vectors: Dictionary of source language vectors (mapped)
        tgt_vectors: Dictionary of target language vectors
        dict_path: Path to dictionary file with word pairs
        output_path: Path to save the visualization
        max_pairs: Maximum number of pairs to show
        random_seed: Random seed for reproducibility
    """
    logging.info(f"Creating visualization of aligned word pairs using dictionary {dict_path}")
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Create word2id mappings for filtering
    src_word2id = {word: i for i, word in enumerate(src_vectors.keys())}
    tgt_word2id = {word: i for i, word in enumerate(tgt_vectors.keys())}
    
    # Load dictionary of word pairs
    word_pairs = load_dictionary(dict_path, src_word2id, tgt_word2id)
    logging.info(f"Loaded {len(word_pairs)} word pairs from dictionary")
    
    # Shuffle pairs and limit to max_pairs
    if len(word_pairs) > max_pairs:
        np.random.shuffle(word_pairs)
        word_pairs = word_pairs[:max_pairs]
    
    # Collect embeddings for valid pairs
    valid_pairs = []
    src_points = []
    tgt_points = []
    
    for src_word, tgt_word in word_pairs:
        if src_word in src_vectors and tgt_word in tgt_vectors:
            valid_pairs.append((src_word, tgt_word))
            src_points.append(src_vectors[src_word])
            tgt_points.append(tgt_vectors[tgt_word])
    
    logging.info(f"Found {len(valid_pairs)} valid pairs for visualization")
    
    if len(valid_pairs) == 0:
        logging.error("No valid pairs found for visualization")
        return
    
    # Combine all points for t-SNE
    all_points = np.array(src_points + tgt_points)
    
    # Apply t-SNE
    logging.info("Applying t-SNE to embeddings")
    tsne = TSNE(
        n_components=2,
        random_state=random_seed,
        perplexity=min(30, len(all_points) // 5),
        early_exaggeration=4.0,
        learning_rate='auto',
        n_iter=2500,
        metric='cosine'
    )
    points_2d = tsne.fit_transform(all_points)
    
    # Split back into source and target points
    n_pairs = len(valid_pairs)
    src_points_2d = points_2d[:n_pairs]
    tgt_points_2d = points_2d[n_pairs:]
    
    # Create figure
    fig = go.Figure()
    
    # Plot source points
    fig.add_trace(go.Scatter(
        x=src_points_2d[:, 0],
        y=src_points_2d[:, 1],
        mode='markers+text',
        text=[pair[0] for pair in valid_pairs],
        textposition='top center',
        marker=dict(
            size=8,
            color='blue',
            symbol='circle'
        ),
        textfont=dict(
            size=10,
            color='blue'
        ),
        name=src_lang
    ))
    
    # Plot target points
    fig.add_trace(go.Scatter(
        x=tgt_points_2d[:, 0],
        y=tgt_points_2d[:, 1],
        mode='markers+text',
        text=[pair[1] for pair in valid_pairs],
        textposition='bottom center',
        marker=dict(
            size=8,
            color='red',
            symbol='circle'
        ),
        textfont=dict(
            size=10,
            color='red'
        ),
        name=tgt_lang
    ))
    
    # Draw lines between corresponding points to show alignment
    for i in range(n_pairs):
        fig.add_trace(go.Scatter(
            x=[src_points_2d[i, 0], tgt_points_2d[i, 0]],
            y=[src_points_2d[i, 1], tgt_points_2d[i, 1]],
            mode='lines',
            line=dict(
                color='rgba(120, 120, 120, 0.2)',
                width=1
            ),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Word-to-Word Alignment: {src_lang} → {tgt_lang}',
        template='plotly_white',
        legend=dict(
            title="Languages",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=900,
        height=700
    )
    
    # Save the plot
    fig.write_html(output_path)
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path)
    logging.info(f"Word alignment visualization saved to {output_path} and {png_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize aligned word pairs')
    parser.add_argument('--src_lang', type=str, default='en', help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default='hi', help='Target language code')
    parser.add_argument('--src_emb', type=str, default='data/wiki.en.vec', help='Source embedding file')
    parser.add_argument('--tgt_emb', type=str, default='data/wiki.hi.vec', help='Target embedding file')
    parser.add_argument('--mapping', type=str, default='dumped/debug/mkmjhaj4yv/best_mapping_dico20000.pth', 
                        help='Mapping file path (.pth format)')
    parser.add_argument('--dictionary', type=str, default='data/dictionaries/en-hi.txt',
                        help='Dictionary file with word pairs')
    parser.add_argument('--eval_dict', type=str, default='data/dictionaries/en-hi.5000-6500.txt',
                        help='Evaluation dictionary file (if dictionary not specified)')
    parser.add_argument('--output', type=str, default='data/word_pair_alignment.html', 
                        help='Output visualization file path')
    parser.add_argument('--max_pairs', type=int, default=200, 
                        help='Maximum number of word pairs to visualize')
    parser.add_argument('--max_vocab', type=int, default=100000, 
                        help='Maximum vocabulary size to load from each language')
    
    args = parser.parse_args()
    
    # Load vectors
    src_vectors, src_dim = load_vectors(args.src_emb, args.max_vocab)
    tgt_vectors, tgt_dim = load_vectors(args.tgt_emb, args.max_vocab)
    
    # Apply mapping to source vectors
    mapped_src_vectors = apply_mapping(src_vectors, args.mapping, src_dim)
    
    # Process target vectors to match MUSE procedure
    logging.info("Processing target vectors to match MUSE procedure")
    
    # Validate all target vectors have the same dimension
    valid_tgt_vectors = {}
    for word, vec in tgt_vectors.items():
        if len(vec) == tgt_dim:
            valid_tgt_vectors[word] = vec
    
    # Convert to numpy array for mean calculation
    all_tgt_vecs = np.array(list(valid_tgt_vectors.values()))
    
    # Center and normalize
    tgt_mean = np.mean(all_tgt_vecs, axis=0)
    processed_tgt_vectors = {}
    
    for word, vec in valid_tgt_vectors.items():
        # Center
        vec_centered = vec - tgt_mean
        
        # Normalize
        norm = np.linalg.norm(vec_centered)
        if norm > 0:
            processed_tgt_vectors[word] = vec_centered / norm
        else:
            processed_tgt_vectors[word] = vec_centered
    
    # Set language display names
    src_display = 'English' if args.src_lang == 'en' else args.src_lang
    tgt_display = 'Hindi' if args.tgt_lang == 'hi' else args.tgt_lang
    
    # Use evaluation dictionary if main dictionary doesn't exist
    dict_path = args.dictionary
    if not os.path.exists(dict_path) and os.path.exists(args.eval_dict):
        logging.info(f"Dictionary {dict_path} not found, using evaluation dictionary {args.eval_dict}")
        dict_path = args.eval_dict
    
    # Create visualization
    visualize_aligned_pairs(
        src_lang=src_display,
        tgt_lang=tgt_display,
        src_vectors=mapped_src_vectors,
        tgt_vectors=processed_tgt_vectors,
        dict_path=dict_path,
        output_path=args.output,
        max_pairs=args.max_pairs
    )

if __name__ == "__main__":
    main() 