import logging
import requests
from tqdm import tqdm
import os
import numpy as np
import plotly.graph_objects as go
import fasttext
from gensim.corpora.wikicorpus import WikiCorpus
import re
import pandas as pd
from sklearn.manifold import TSNE

# Use the indic-transliteration library for better results
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple tokenizer function for all languages
def simple_tokenize(text, token_min_len=2, token_max_len=100, lower=True):
    """
    Simple whitespace tokenization function that works well for
    most languages with space-separated words.
    
    Args:
        text: Text to tokenize
        token_min_len: Minimum token length
        token_max_len: Maximum token length
        lower: Whether to lowercase the tokens
    
    Returns:
        List of tokens
    """
    # Apply lowercase if requested
    if lower:
        text = text.lower()
    
    # Simple whitespace tokenization
    tokens = text.split()
    
    # Filter by length
    tokens = [token for token in tokens if token_min_len <= len(token) <= token_max_len]
    
    return tokens

def download_wiki_dump(lang, output_path):
    """Download Wikipedia dump for a specific language"""
    # Check if file already exists
    if os.path.exists(output_path):
        logging.info(f"File already exists at {output_path}, skipping download")
        return
        
    url = f'https://dumps.wikimedia.org/{lang}wiki/latest/{lang}wiki-latest-pages-articles-multistream.xml.bz2'
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as file, tqdm(
        desc=f"Downloading {lang} Wikipedia dump",
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as bar:
        for data in response.iter_content(1024):
            size = file.write(data)
            bar.update(size)


def extract_wiki_text(dump_path, output_path, limit=10000, lang=None):
    """Extract text from Wikipedia articles
    
    Args:
        dump_path: Path to Wikipedia XML dump
        output_path: Path to output file
        limit: Maximum number of articles to process
        lang: Language code (not used, kept for backward compatibility)
    """
    logging.info(f"Extracting text from {dump_path}")
    
    # Use the simple tokenizer for all languages
    wiki = WikiCorpus(dump_path, dictionary={}, tokenizer_func=simple_tokenize)
    
    with open(output_path, 'w', encoding='utf-8') as output:
        for i, text in enumerate(wiki.get_texts()):
            if i % 1000 == 0:
                logging.info(f"Processed {i} articles")
            output.write(' '.join(text) + '\n')
            # if i >= limit-1:
            #     break
    
    logging.info(f"Extracted {min(limit, i+1)} articles to {output_path}")

def transliterate_text(input_path, output_path, source_script='devanagari', target_script='latin'):
    """Transliterate text from source script to target script
    
    Args:
        input_path: Path to input text file
        output_path: Path to output transliterated file
        source_script: Source script (e.g., 'devanagari')
        target_script: Target script (e.g., 'latin')
    """
    logging.info(f"Transliterating text from {source_script} to {target_script}")
    
    try:
        # Map source script to sanscript schemes
        script_map = {
            'devanagari': sanscript.DEVANAGARI,
            'bengali': sanscript.BENGALI,
            'gujarati': sanscript.GUJARATI,
            'gurmukhi': sanscript.GURMUKHI,
            'kannada': sanscript.KANNADA,
            'malayalam': sanscript.MALAYALAM,
            'oriya': sanscript.ORIYA,
            'tamil': sanscript.TAMIL,
            'telugu': sanscript.TELUGU
        }
        
        # Map target script to sanscript schemes
        target_map = {
            'latin': sanscript.IAST,  # Use IAST as the default Latin scheme
            'iast': sanscript.IAST,
            'itrans': sanscript.ITRANS,
            'hk': sanscript.HK,  # Harvard-Kyoto
            'slp1': sanscript.SLP1
        }
        
        # Get the appropriate schemes
        source_scheme = script_map.get(source_script.lower(), None)
        target_scheme = target_map.get(target_script.lower(), sanscript.IAST)
        
        if not source_scheme:
            logging.error(f"Unsupported source script: {source_script}")
            return False
        
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for i, line in enumerate(tqdm(infile, desc="Transliterating")):
                if i % 10000 == 0:
                    logging.info(f"Transliterated {i} lines")
                
                # Transliterate the line
                transliterated = transliterate(line, source_scheme, target_scheme)
                outfile.write(transliterated + '\n')
        
        logging.info(f"Transliteration completed, saved to {output_path}")
        return True
    
    except ImportError:
        logging.error("indic-transliteration library not found. Install with: pip install indic-transliteration")
        return False

def train_fasttext_model(input_path, output_path, dim=300, epoch=5, min_count=5, lr=0.05):
    """Train a fastText model on Wikipedia text with visualization of progress"""
    logging.info(f"Training fastText model on {input_path}")
    
    # Train the model without callback
    model = fasttext.train_unsupervised(
        input_path, 
        model='skipgram',
        dim=dim,
        minCount=min_count,
        epoch=epoch,
        lr=lr
    )
    
    # Save the model
    model.save_model(output_path)
    logging.info(f"Model saved to {output_path}")
    
    # Since we can't get loss values through callback, create a simpler visualization
    # showing model information instead
    fig = go.Figure()
    
    # Create a table with model parameters
    fig.add_trace(go.Table(
        header=dict(values=['Parameter', 'Value'],
                   fill_color='paleturquoise',
                   align='left'),
        cells=dict(values=[
            ['Embedding Dimension', 'Model Type', 'Minimum Count', 'Epochs', 'Learning Rate', 
             'Vocabulary Size', 'Word Vectors', 'File Size'],
            [dim, 'skipgram', min_count, epoch, lr, 
             len(model.words), f"{dim}d vectors", f"{os.path.getsize(output_path)/1024/1024:.2f} MB"]
        ],
        fill_color='lavender',
        align='left')
    ))
    
    fig.update_layout(
        title=f'FastText Model Information',
        template='plotly_white'
    )
    
    # Save the plot using standardized naming
    # From wiki.XX.bin to wiki.XX_model_info.html
    plot_path = output_path.replace('.bin', '_model_info.html')
    fig.write_html(plot_path)
    logging.info(f"Model information visualization saved to {plot_path}")
    
    return model


def filter_top_words(model, output_path, limit=200000):
    """
    Export word vectors in fastText standard format (.vec)
    
    Args:
        model: Trained fastText model
        output_path: Path to save the vectors
        limit: Maximum number of words to include
    """
    logging.info(f"Exporting top {limit} words from model to {output_path}")
    
    # Get all words sorted by frequency (most frequent first)
    words = sorted(model.words, key=lambda w: model.get_word_id(w))
    
    # Limit to the specified number
    words = words[:min(limit, len(words))]
    
    # Save the vectors in .vec format
    with open(output_path, 'w', encoding='utf-8') as f:
        # First line: number_of_words vector_dimension
        f.write(f"{len(words)} {model.get_dimension()}\n")
        
        # Write each word and its vector
        for word in words:
            vector = model.get_word_vector(word)
            vector_str = ' '.join(f"{v:.6f}" for v in vector)
            f.write(f"{word} {vector_str}\n")
    
    logging.info(f"Saved {len(words)} word vectors to {output_path}")


def visualize_embeddings(model, output_path, n_words=1000):
    """Create a t-SNE visualization of word embeddings"""
    # Get the most frequent words (limited by actual vocabulary size)
    n_words = min(n_words, len(model.words))
    words = sorted(model.words, key=lambda w: model.get_word_id(w))[:n_words]
    
    if len(words) < 2:
        logging.warning(f"Not enough words ({len(words)}) for visualization, skipping")
        return
    
    # Get word vectors and convert to numpy array
    word_vectors = np.array([model.get_word_vector(word) for word in words])
    
    logging.info(f"Applying t-SNE to {len(words)} word vectors of dimension {word_vectors.shape[1]}")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(word_vectors_2d, columns=['x', 'y'])
    df['word'] = words
    
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        text=df['word'],
        textposition='top center',
        marker=dict(size=5),
        textfont=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'Word Embeddings Visualization (t-SNE) - {len(words)} Words',
        template='plotly_white'
    )
    
    # Save the plot
    fig.write_html(output_path)
    # Save the plot as a PNG image
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path)
    logging.info(f"Word embeddings visualization saved to {output_path}")


def run_pipeline(lang, base_dir='data', article_limit=10000, word_limit=200000, transliterate=False, transliteration_scheme='iast'):
    """Run the complete Wikipedia embeddings pipeline for a language"""
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    
    # Define paths using fastText standard naming convention
    dump_path = os.path.join(base_dir, f'{lang}_wiki_dump.xml.bz2')
    text_path = os.path.join(base_dir, f'{lang}_wiki_text.txt')
    transliterated_path = os.path.join(base_dir, f'{lang}_wiki_text_latin.txt')
    model_path = os.path.join(base_dir, f'wiki.{lang}.bin')  # Standard fastText naming
    vectors_path = os.path.join(base_dir, f'wiki.{lang}.vec')  # Standard fastText naming
    viz_path = os.path.join(base_dir, f'wiki.{lang}_embeddings_viz.html')
    
    if transliterate:
        model_path = os.path.join(base_dir, f'wiki.{lang}_latin.bin')
        vectors_path = os.path.join(base_dir, f'wiki.{lang}_latin.vec')
        viz_path = os.path.join(base_dir, f'wiki.{lang}_latin_embeddings_viz.html')
    
    try:
        # Step 1: Download Wikipedia dump
        logging.info(f"=== Step 1: Downloading Wikipedia dump for {lang} ===")
        download_wiki_dump(lang, dump_path)
        
        try:
            # Step 2: Extract text
            logging.info(f"=== Step 2: Extracting text from Wikipedia dump ===")
            extract_wiki_text(dump_path, text_path, limit=article_limit)
        except Exception as e:
            logging.error(f"Error extracting text: {str(e)}")
            raise
        
        # Step 3: Transliteration (only for specific languages)
        if transliterate and lang in ['hi', 'sa', 'mr', 'bn', 'pa', 'gu', 'ta', 'te', 'kn', 'ml', 'or', 'as']:
            logging.info(f"=== Step 3: Transliterating text to Latin script ({transliteration_scheme}) ===")
            src_script = 'devanagari' if lang in ['hi', 'sa', 'mr'] else lang
            success = transliterate_text(text_path, transliterated_path, 
                                         source_script=src_script, target_script=transliteration_scheme)
            if success:
                # Use transliterated text for training instead
                input_for_training = transliterated_path
            else:
                input_for_training = text_path
                logging.warning("Transliteration failed, using original text for training")
        else:
            input_for_training = text_path
        
        try:
            # Step 4: Train fastText model
            logging.info(f"=== Step 4: Training fastText model ===")
            model = train_fasttext_model(input_for_training, model_path)
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
        
        try:
            # Step 5: Filter top words
            logging.info(f"=== Step 5: Filtering top words ===")
            filter_top_words(model, vectors_path, limit=word_limit)
        except Exception as e:
            logging.error(f"Error filtering words: {str(e)}")
            raise
        
        try:
            # Step 6: Visualize embeddings
            logging.info(f"=== Step 6: Visualizing word embeddings ===")
            visualize_embeddings(model, viz_path)
        except Exception as e:
            logging.error(f"Error visualizing embeddings: {str(e)}")
            logging.warning("Continuing despite visualization error")
        
        logging.info(f"Pipeline completed successfully for {lang}")
        return True
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fastText embeddings on Wikipedia data')
    parser.add_argument('--lang', type=str, required=True, help='Language code (e.g., en, hi, sa)')
    parser.add_argument('--base_dir', type=str, default='data', help='Base directory for data')
    parser.add_argument('--article_limit', type=int, default=10_000, help='Number of articles to process')
    parser.add_argument('--word_limit', type=int, default=200_000, help='Number of words to keep')
    parser.add_argument('--transliterate', action='store_true', help='Transliterate text to Latin script')
    parser.add_argument('--transliteration_scheme', type=str, default='iast', 
                        choices=['iast', 'itrans', 'hk', 'slp1'], 
                        help='Transliteration scheme to use (iast, itrans, hk, slp1)')

    args = parser.parse_args()
    
    run_pipeline(
        lang=args.lang,
        base_dir=args.base_dir,
        article_limit=args.article_limit,
        word_limit=args.word_limit,
        transliterate=args.transliterate,
        transliteration_scheme=args.transliteration_scheme
    )
