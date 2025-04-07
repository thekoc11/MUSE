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
            if limit > 0 and i >= limit-1:
                break
    
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


def visualize_multiple_embeddings(models, lang_names, output_path, n_words=1000, label_percentage=30, 
                                 colors=None, marker_sizes=None, random_seed=42):
    """
    Create a t-SNE visualization comparing word embeddings across multiple languages
    
    Args:
        models: List of fastText models to compare
        lang_names: List of language names corresponding to each model
        output_path: Path to save the visualization
        n_words: Number of most frequent words to include from each language
        label_percentage: Percentage of words to label (0-100)
        colors: List of colors for each language
        marker_sizes: List of marker sizes for each language
        random_seed: Random seed for reproducibility
    """
    if len(models) != len(lang_names):
        raise ValueError("Number of models must match number of language names")
    
    if len(models) < 2:
        raise ValueError("At least 2 models are needed for comparison")
    
    # Set default colors if not provided
    if colors is None:
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
        # Ensure we have enough colors
        if len(models) > len(colors):
            logging.warning(f"Not enough default colors for {len(models)} models")
            # Repeat colors if needed
            colors = colors * (len(models) // len(colors) + 1)
        # Trim to the number of models
        colors = colors[:len(models)]
    
    # Set default marker sizes if not provided
    if marker_sizes is None:
        marker_sizes = [5] * len(models)
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Collect words and vectors from each model
    all_words = []
    all_vectors = []
    all_langs = []
    
    for i, (model, lang) in enumerate(zip(models, lang_names)):
        # Get most frequent words for this model
        model_n_words = min(n_words, len(model.words))
        words = sorted(model.words, key=lambda w: model.get_word_id(w))[:model_n_words]
        
        # Skip models with too few words
        if len(words) < 2:
            logging.warning(f"Not enough words in {lang} model, skipping")
            continue
        
        # Get word vectors
        vectors = [model.get_word_vector(word) for word in words]
        
        # Add to collections
        all_words.extend(words)
        all_vectors.extend(vectors)
        all_langs.extend([lang] * len(words))
    
    # Convert to numpy array for t-SNE
    all_vectors_np = np.array(all_vectors)
    
    logging.info(f"Applying t-SNE to {len(all_words)} total word vectors across {len(models)} languages")
    
    # Apply t-SNE to reduce dimensions
    tsne = TSNE(n_components=2, random_state=random_seed)
    vectors_2d = tsne.fit_transform(all_vectors_np)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(vectors_2d, columns=['x', 'y'])
    df['word'] = all_words
    df['language'] = all_langs
    
    # Determine which points to label (randomly select label_percentage% of points per language)
    df['show_label'] = False
    
    for lang in df['language'].unique():
        # Get indices of points for this language
        lang_indices = df.index[df['language'] == lang].tolist()
        
        # Calculate number of points to label
        n_to_label = max(1, int(len(lang_indices) * label_percentage / 100))
        
        # Randomly select points to label
        label_indices = np.random.choice(lang_indices, size=n_to_label, replace=False)
        df.loc[label_indices, 'show_label'] = True
    
    # Create plot
    fig = go.Figure()
    
    # Add traces for each language
    for i, lang in enumerate(df['language'].unique()):
        lang_df = df[df['language'] == lang]
        
        # Points with labels
        labeled_df = lang_df[lang_df['show_label']]
        fig.add_trace(go.Scatter(
            x=labeled_df['x'],
            y=labeled_df['y'],
            mode='markers+text',
            text=labeled_df['word'],
            textposition='top center',
            marker=dict(
                size=marker_sizes[i],
                color=colors[i],
                opacity=0.7
            ),
            textfont=dict(size=8),
            name=lang
        ))
        
        # Points without labels
        unlabeled_df = lang_df[~lang_df['show_label']]
        fig.add_trace(go.Scatter(
            x=unlabeled_df['x'],
            y=unlabeled_df['y'],
            mode='markers',
            marker=dict(
                size=marker_sizes[i],
                color=colors[i],
                opacity=0.7
            ),
            name=lang,
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Comparing Word Embeddings Across Languages (t-SNE)',
        template='plotly_white',
        legend=dict(
            title="Languages",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save the plot
    fig.write_html(output_path)
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path)
    logging.info(f"Multi-language word embeddings visualization saved to {output_path}")


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


def compare_embeddings(lang_pair, base_dir='data', n_words=1000, label_percentage=20):
    """
    Compare word embeddings between two languages
    
    Args:
        lang_pair: A string in the format 'lang1-lang2' (e.g., 'en-hi' or 'en-hi_latin')
        base_dir: Base directory for data files
        n_words: Number of most frequent words to include from each language
        label_percentage: Percentage of words to label
    """
    # Parse language pair
    langs = lang_pair.split('-')
    if len(langs) != 2:
        raise ValueError(f"Invalid language pair format: {lang_pair}. Expected format: 'lang1-lang2'")
    
    lang1, lang2 = langs
    logging.info(f"Comparing embeddings for language pair: {lang1} and {lang2}")
    
    # Check if models exist and load them
    models = []
    lang_names = []
    
    for lang in [lang1, lang2]:
        # Check for model file
        model_path = os.path.join(base_dir, f'wiki.{lang}.bin')
        
        if not os.path.exists(model_path):
            logging.warning(f"Model file not found: {model_path}")
            # Try running the pipeline to create it
            logging.info(f"Running pipeline to create model for {lang}")
            
            # Determine if this is a transliterated model
            transliterate = False
            base_lang = lang
            
            if '_latin' in lang:
                transliterate = True
                base_lang = lang.replace('_latin', '')
                
            success = run_pipeline(
                lang=base_lang,
                base_dir=base_dir,
                transliterate=transliterate
            )
            
            if not success:
                logging.error(f"Failed to create model for {lang}")
                return False
            
            # Verify model was created
            if not os.path.exists(model_path):
                logging.error(f"Model still not found after pipeline: {model_path}")
                return False
        
        # Load the model
        logging.info(f"Loading model: {model_path}")
        model = fasttext.load_model(model_path)
        models.append(model)
        
        # Use nice display names for the languages
        display_name = lang
        if lang == 'en':
            display_name = 'English'
        elif lang == 'hi':
            display_name = 'Hindi'
        elif lang == 'hi_latin':
            display_name = 'Hindi (Latin)'
        elif lang == 'sa':
            display_name = 'Sanskrit'
        elif lang == 'sa_latin':
            display_name = 'Sanskrit (Latin)'
        elif lang == 'es':
            display_name = 'Spanish'
            
        lang_names.append(display_name)
    
    # Define output path
    output_path = os.path.join(base_dir, f'comparison_{lang1}_vs_{lang2}.html')
    
    # Visualize the comparison
    visualize_multiple_embeddings(
        models=models,
        lang_names=lang_names,
        output_path=output_path,
        n_words=n_words,
        label_percentage=label_percentage
    )
    
    logging.info(f"Comparison visualization saved to {output_path}")
    return True


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train fastText embeddings on Wikipedia data')
    parser.add_argument('--lang', type=str, help='Language code (e.g., en, hi, sa)')
    parser.add_argument('--base_dir', type=str, default='data', help='Base directory for data')
    parser.add_argument('--article_limit', type=int, default=10_000, help='Number of articles to process')
    parser.add_argument('--word_limit', type=int, default=200_000, help='Number of words to keep')
    parser.add_argument('--transliterate', action='store_true', help='Transliterate text to Latin script')
    parser.add_argument('--transliteration_scheme', type=str, default='iast', 
                        choices=['iast', 'itrans', 'hk', 'slp1'], 
                        help='Transliteration scheme to use (iast, itrans, hk, slp1)')
    parser.add_argument('--visualize', type=str, help='Visualize language pair embeddings (format: lang1-lang2, e.g., en-hi, en-hi_latin)')
    parser.add_argument('--n_words', type=int, default=1000, help='Number of words to include in visualization')
    parser.add_argument('--label_percentage', type=int, default=20, help='Percentage of words to label (0-100)')

    args = parser.parse_args()
    
    if args.visualize:
        # Visualize language pair
        compare_embeddings(
            lang_pair=args.visualize,
            base_dir=args.base_dir,
            n_words=args.n_words,
            label_percentage=args.label_percentage
        )
    elif args.lang:
        # Run normal pipeline
        run_pipeline(
            lang=args.lang,
            base_dir=args.base_dir,
            article_limit=args.article_limit,
            word_limit=args.word_limit,
            transliterate=args.transliterate,
            transliteration_scheme=args.transliteration_scheme
        )
    else:
        parser.print_help()
