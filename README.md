# Cross-Lingual Word Embedding Alignment: English to Hindi

## 1. Introduction

This report presents the implementation and evaluation of a supervised cross-lingual word embedding alignment system for English and Hindi languages using the Procrustes method. Cross-lingual word embeddings enable mapping words from different languages into a shared semantic space, which is fundamental for multilingual NLP tasks such as machine translation, cross-lingual information retrieval, and knowledge transfer.

<!-- TODO: A section need to be added for environment setup. We need to explain what mamba is and how it is different from miniconda but still essentially the same. Then, we need to give directions to install mamba and create a new mamba environment called "muse", with python=3.8.20. For this, we may need to search the web, and then write this part on the basis of our search results. Then we need to provide instructions on how to use mamba to create environment using the `environment.yaml` file -->

## 1. Environment Setup

Reproducible computational environments are crucial for research. This project utilizes the Conda package and environment management system, specifically leveraging Mamba for enhanced performance.

### 1.1 Mamba vs. Conda

*   **Conda:** A widely used open-source package management and environment management system that runs on Windows, macOS, and Linux. It allows users to create isolated environments containing specific versions of Python and other packages. Miniconda is a minimal installer for Conda.
*   **Mamba:** A reimplementation of the Conda package manager in C++. It offers significantly faster dependency solving and package downloading/installation compared to the standard `conda` command, primarily due to parallel processing and more efficient algorithms. Mamba is largely command-line compatible with Conda and uses the same environment structure and package sources (channels like `conda-forge`). It's typically installed within an existing Miniconda or Anaconda setup.

For this project, using Mamba within a Miniconda distribution is recommended for faster environment setup and package management.

### 1.2 Installation and Usage

The recommended way to manage the environment for this project is using **Mamba** with the provided `environment.yaml` file. Mamba is included in the **Miniforge** distribution, which is the recommended Conda installer for this setup.

1.  **Install Miniforge:** If you don't have a Conda distribution with Mamba installed, download and install the Miniforge installer appropriate for your operating system from the official repository: [https://github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge). Follow the instructions provided by the installer. This sets up a base Conda environment that includes Mamba and defaults to the `conda-forge` channel.

2.  **Verify Mamba Installation (Optional):** After installation, open a new terminal window (or Anaconda Prompt on Windows) and run `mamba --version` to confirm it's installed correctly.

3.  **Create Environment from File (Primary Method):** Navigate to the root directory of this project in your terminal (the directory containing the `environment.yaml` file) and create the project environment using the following Mamba command:
    ```bash
    mamba env create -f environment.yaml
    ```
    This command reads the `environment.yaml` file (which specifies the environment name `muse_env`, channels, and all necessary dependencies including Python 3.8 and packages like PyTorch, Faiss, etc.) and installs everything required, ensuring a consistent and reproducible setup.

4.  **Activate the Environment:** Before running any project code, activate the newly created environment:
    ```bash
    conda activate muse_env
    ```
    *(Note: The environment name `muse_env` is defined within the `environment.yaml` file).*

## 2. Methodology

### 2.1 Procrustes Alignment

The Procrustes method is a linear transformation approach that finds the optimal orthogonal mapping between two sets of embeddings. Given source embeddings X and target embeddings Y, along with a bilingual dictionary that maps words from the source language to the target language, the objective is to find an orthogonal matrix W that minimizes:

$\min_{W \in \mathcal{O}_d} \|WX - Y\|_F$

Where $\mathcal{O}_d$ is the space of orthogonal matrices of dimension d×d. This constraint ensures that the mapping preserves distances and angles between word vectors, maintaining the structural properties of the original embedding space.

The solution to this optimization problem is given by:
$W = UV^T$, where $UΣV^T$ is the singular value decomposition of $YX^T$.

### 2.2 Iterative Refinement

The alignment process is iterative, following these steps:
1. Initial dictionary-based alignment using Procrustes
2. Building a new dictionary based on the current alignment
3. Realigning using the new dictionary
4. Repeating steps 2-3 for a fixed number of iterations

This process helps improve the quality of the alignment by iteratively refining the set of word pairs used for training.

## 3. Implementation

### 3.1 Data Preparation

<!-- TODO: Describe comprehensively the steps taken in `fastTest_model.py`. Explain that this was done without any transliteration for english and espanol, and both with and without transliteration for Hindi. The vectorization largely is done as specfied in `paper.txt`(cite sections of the paper as reference), for all the languages except for transliterated Hindi. Then address the "Data Preparation" Section in `assgn.txt`, i.e. the original assignment. English and non-transliterated Hindi follow the exact criteria mentioned in the section. For addressing the point c., inform that the following command was run:

```bash
cd data/
wget https://dl.fbaipublicfiles.com/arrival/wordsim.tar.gz
wget https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
``` 

After this, as a preliminary analysis, we show `comparison_en_vs_hi` as a baseline of what the separation between the languages looks like
-->

The foundation of cross-lingual alignment lies in high-quality monolingual word embeddings. The process for generating these embeddings for English (en) and Hindi (hi) was managed by the `fastText_model.py` script, which largely follows the methodology described in the MUSE paper (Conneau et al., 2017, Section 3.1) while also addressing the requirements of the assignment (`assgn.txt`, Data Preparation section). Embeddings for Spanish (es) and IAST-transliterated Hindi (hi_latin) were also prepared using the same script for later experiments (detailed in Section 5), but the primary focus here is on English and standard Devanagari Hindi.

The pipeline implemented in `fastText_model.py` involved the following steps for each language:

1.  **Wikipedia Data Acquisition:** The latest Wikipedia dump (`*-latest-pages-articles-multistream.xml.bz2`) was downloaded for the respective language using the `download_wiki_dump` function.
2.  **Text Extraction and Tokenization:** Text content was extracted from the downloaded dump using `gensim`'s `WikiCorpus`. The `extract_wiki_text` function processed the articles, applying a simple whitespace tokenizer (`simple_tokenize`) that converted text to lowercase and filtered tokens to be between 2 and 100 characters long. Following the assignment's guideline (`assgn.txt`, Data Prep a), text was extracted from the first 10,000 articles (`--article_limit=10000` argument in `fastText_model.py`).
3.  **Transliteration (for Hindi - Optional):** For generating the `hi_latin` embeddings used in the unsupervised experiments (Section 5), the extracted Hindi text (`hi_wiki_text.txt`) was transliterated from Devanagari to the Latin script using the IAST scheme via the `indic-transliteration` library (`transliterate_text` function). The standard Hindi embeddings (`wiki.hi.vec`) used for the supervised experiments in Section 4 were trained directly on the Devanagari text without transliteration.
4.  **fastText Model Training:** A fastText model was trained on the processed text (either original or transliterated) using the `train_fasttext_model` function. Key parameters were set to align with the MUSE paper's specifications (Section 3.1):
    *   `model='skipgram'`
    *   `dim=300` (embedding dimension)
    *   `minCount=5` (minimum word frequency)
    *   `epoch=5`, `lr=0.05` (default training parameters in the script)
    The trained model was saved in the standard `.bin` format (e.g., `wiki.en.bin`, `wiki.hi.bin`).
5.  **Vocabulary Filtering and Vector Export:** The embeddings for the top `N` most frequent words were exported to the standard `.vec` text format using the `filter_top_words` function. For the supervised experiments described in Section 4, the vocabulary was limited to the top 100,000 words, aligning with the assignment requirement (`assgn.txt`, Data Prep b). Although the MUSE paper (Section 3.1) used 200,000 words, and the `fastText_model.py` script defaults to this (`--word_limit=200000`), the 100k limit was chosen for the primary supervised analysis. The 200k limit was utilized for the unsupervised experiments discussed later (Section 5). The resulting files (e.g., `data/wiki.en.vec`, `data/wiki.hi.vec`) containing the 100k vocabulary were used in the subsequent supervised alignment experiments.

To fulfill the assignment requirement (`assgn.txt`, Data Prep c) of obtaining bilingual lexicons and evaluation data from the MUSE dataset, the necessary files were downloaded using `wget`:

```bash
cd data/ # Assuming execution from the project root
wget https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
wget https://dl.fbaipublicfiles.com/arrival/wordsim.tar.gz
tar -xzf dictionaries.tar.gz
tar -xzf wordsim.tar.gz
cd ..
```
This provided the training and evaluation dictionaries (e.g., `en-hi.txt`, `en-hi.5000-6500.txt`) and word similarity datasets used throughout the project.

Finally, to establish a visual baseline before alignment, the `compare_embeddings` function within `fastText_model.py` was used to generate a t-SNE plot comparing the raw English and Hindi embeddings (using the top 1000 words from each for clarity).

![Baseline Comparison English vs Hindi Embeddings](data/comparison_en_vs_hi.png)

As shown in the figure above (`comparison_en_vs_hi.png`), the initial embedding spaces for English (blue) and Hindi (red) are largely separated, illustrating the need for an alignment process to map them into a shared semantic space. This visualization serves as the starting point against which the effectiveness of the alignment (shown later in Section 4.1) can be qualitatively assessed.

### 3.2 Orthogonality Verification

To ensure the mapping preserved distances and angles, a verification function was implemented that checks whether $W^TW$ is close to the identity matrix:

```python
def verify_orthogonality(W):
    WtW = torch.mm(W.t(), W)
    I = torch.eye(W.shape[1], device=W.device)
    error = torch.norm(WtW - I).item()
    is_orthogonal = error < 1e-4
    return is_orthogonal, error
```

## 4. Experimental Setup
<!--- TODO: Need to mention that only English and Non-transliterated Hindi were used for these experiments. Ensure that this section DOES NOT MISS anything from the workflow defined in `muse_wrapper.py`. Also inform `run_supervised.sh` was created for convenience and explain each option in it --->
The experiments described in this section focused exclusively on aligning English and non-transliterated Hindi embeddings.

The core supervised alignment process and the ablation study on dictionary size were managed by the `muse_wrapper.py` script. This script automates the following workflow:
1.  **Configuration:** Parses command-line arguments to set up experiment parameters, including language identifiers, embedding paths, vocabulary sizes, normalization methods, and refinement settings.
2.  **Dictionary Size Iteration:** Loops through a predefined list of bilingual dictionary sizes: 1,000, 2,000, 5,000, 10,000, and 20,000 word pairs.
3.  **Experiment Execution per Size:** For each dictionary size:
    *   **Dictionary Preparation:** Creates a training dictionary of the specified size by randomly sampling from a larger combined dictionary file (`en-hi.txt`), ensuring that pairs present in the validation set (`en-hi.5000-6500.txt`) are excluded.
    *   **Model Training:** Initializes the MUSE model components (source and target embeddings, mapping layer).
    *   **Alignment and Refinement:** Performs the alignment using the Procrustes method followed by a specified number of iterative refinement steps (`n_refinement`). In each refinement step beyond the first, a new dictionary is built based on the current alignment using a specified method (e.g., CSLS KNN).
    *   **Orthogonality:** Verifies if the learned mapping matrix `W` is orthogonal ($W^T W \approx I$). If the deviation exceeds a threshold and orthogonalization is enabled (`map_beta > 0`), it applies an iterative orthogonalization procedure.
    *   **Evaluation:** Calculates standard word translation evaluation metrics (Precision@1 and Precision@5 using both Nearest Neighbor and CSLS retrieval) on the designated evaluation dictionary. Additionally, it computes the average cosine similarity between the mapped source vectors and the target vectors for the word pairs in the evaluation dictionary.
    *   **Best Model Saving:** Tracks performance across refinement iterations and saves the mapping matrix (`best_mapping_dico<size>.pth`) corresponding to the iteration that yielded the highest average cosine similarity.
4.  **Result Aggregation & Visualization:** After running experiments for all dictionary sizes, the script aggregates the final metrics from the best model for each size into a JSON file (`all_results.json`) and generates interactive HTML plots (`precision_results.html`, `mapping_quality.html`) visualizing the relationship between dictionary size and performance metrics.

To facilitate running these experiments, particularly with different hyperparameter configurations, the `run_supervised.sh` script was created. This Bash script provides several conveniences:
*   **Base Command:** Defines the fundamental command to execute `muse_wrapper.py` with the source (`en`), target (`hi`), and their respective embedding file paths (`data/wiki.en.vec`, `data/wiki.hi.vec`).
*   **CPU/GPU Control:** Allows specifying `--cpu` as an argument to force execution on the CPU by appending `--cuda False` to the base command; otherwise, it defaults to using the GPU.
*   **Experiment Runner:** Includes a function `run_experiment` that takes an experiment name and the full command. It executes the command, simultaneously displaying the output to the console and saving it (both standard output and standard error) to a log file named `experiment_logs/<experiment_name>.log`.
*   **Hyperparameter Testing Menu:** Presents an interactive menu prompting the user to select which set of experiments to run. This allows easily testing variations of hyperparameters used by `muse_wrapper.py`, such as:
    *   `--normalize_embeddings`: Type of embedding normalization (e.g., `center`, `renorm`, `center,renorm`).
    *   `--n_refinement`: Number of refinement iterations (e.g., 3, 10; default is 5).
    *   `--dico_build`: Strategy for building the dictionary in refinement steps (e.g., `S2T|T2S` which uses pairs found from source-to-target and target-to-source searches).
    *   `--dico_threshold`: Similarity threshold for including pairs in the refinement dictionary.
    *   `--map_beta`: Strength of the orthogonality constraint regularization.
    *   `--dico_method`: Method for finding nearest neighbors when building the refinement dictionary (e.g., `csls_knn_5`, `csls_knn_15`).
The script offers options to run a baseline configuration, specific tests focusing on one hyperparameter, a `combined_best` configuration using potentially optimal settings derived from individual tests, or all predefined experiments sequentially.

### 4.1 Ablation Study on Dictionary Size

As per the assignment requirements, an ablation study was first conducted to assess the impact of bilingual lexicon size on alignment quality. The experiments were run with dictionary sizes of:
- 1,000 word pairs
- 2,000 word pairs
- 5,000 word pairs
- 10,000 word pairs
- 20,000 word pairs

The models for each dictionary size were selected based on the refinement iteration that yielded the highest average cosine similarity on the evaluation set (`en-hi.5000-6500.txt`). The key performance metrics for these best models are summarized below:

| Dictionary Size | Best Iter | Avg Cosine Sim | P@1 (NN) | P@1 (CSLS) | P@5 (NN) | P@5 (CSLS) |
|-----------------|-----------|----------------|----------|------------|----------|------------|
| 1,000           | 3         | 0.3841         | 31.30%   | 37.19%     | 49.25%   | 54.45%     |
| 2,000           | 2         | 0.3853         | 31.44%   | 37.81%     | 49.59%   | 56.10%     |
| 5,000           | 2         | 0.3862         | 32.19%   | 38.84%     | 49.45%   | 56.58%     |
| 10,000          | 1         | 0.3878         | 32.12%   | 38.63%     | 50.07%   | 56.51%     |
| 20,000          | 1         | 0.3893         | 32.60%   | 38.77%     | 50.21%   | 57.26%     |

![Overall Metrics vs Dictionary Size](plots/all_metrics_by_dict_size_20250407_122416.png)

As depicted in the table and the figure above (`all_metrics_by_dict_size_20250407_122416.png`), increasing the initial dictionary size generally leads to improvements in both precision (especially P@1 CSLS) and average cosine similarity. The gains appear to diminish somewhat with larger dictionary sizes, suggesting potential saturation, particularly between 10k and 20k pairs for this dataset and configuration.

The iterative refinement process plays a crucial role. Analyzing the metrics across the 10 refinement iterations for each dictionary size reveals important dynamics:

*   **1k Dictionary:** Average cosine similarity peaks at iteration 3 (0.3841), coinciding with the peak P@1 (CSLS) of 37.19%. Both metrics show a general upward trend followed by a plateau or slight decline in later iterations.
    ![Avg Cosine Similarity (1k)](plots/avg_cosine_similarity_dict_size_1000_refinement_20250407_122458.png)
    ![P@1 CSLS (1k)](plots/precision_at_1_csls_dict_size_1000_refinement_20250407_122458.png)

*   **2k Dictionary:** Average cosine similarity peaks early at iteration 2 (0.3853), while P@1 (CSLS) also peaks at iteration 2 (37.81%). The trends are similar to the 1k case, stabilizing after the peak.
    ![Avg Cosine Similarity (2k)](plots/avg_cosine_similarity_dict_size_2000_refinement_20250407_122509.png)
    ![P@1 CSLS (2k)](plots/precision_at_1_csls_dict_size_2000_refinement_20250407_122509.png)

*   **5k Dictionary:** Similar to the 2k case, average cosine similarity peaks at iteration 2 (0.3862), and P@1 (CSLS) also peaks at iteration 2 (38.84%).
    ![Avg Cosine Similarity (5k)](plots/avg_cosine_similarity_dict_size_5000_refinement_20250407_122516.png)
    ![P@1 CSLS (5k)](plots/precision_at_1_csls_dict_size_5000_refinement_20250407_122516.png)

*   **10k Dictionary:** Average cosine similarity reaches its maximum at iteration 1 (0.3878), and P@1 (CSLS) also peaks at iteration 1 (38.63%). Both metrics show a slight decline in subsequent iterations.
    ![Avg Cosine Similarity (10k)](plots/avg_cosine_similarity_dict_size_10000_refinement_20250407_122523.png)
    ![P@1 CSLS (10k)](plots/precision_at_1_csls_dict_size_10000_refinement_20250407_122523.png)

*   **20k Dictionary:** Average cosine similarity again peaks early at iteration 1 (0.3893). P@1 (CSLS) also achieves its best result at iteration 1 (38.77%). Interestingly, after iteration 1, both the average cosine similarity and P@1 (CSLS) show a noticeable downward trend throughout the remaining refinement steps.
    ![Avg Cosine Similarity (20k)](plots/avg_cosine_similarity_dict_size_20000_refinement_20250407_122530.png)
    ![P@1 CSLS (20k)](plots/precision_at_1_csls_dict_size_20000_refinement_20250407_122530.png)

A notable contrast is evident when comparing the average cosine similarity refinement trends for the 1k and 20k dictionary sizes (compare ![Avg Cosine Similarity (1k)](plots/avg_cosine_similarity_dict_size_1000_refinement_20250407_122458.png) with ![Avg Cosine Similarity (20k)](plots/avg_cosine_similarity_dict_size_20000_refinement_20250407_122530.png)). The 1k experiment shows gradual improvement over the first few iterations, suggesting that refinement successfully bootstraps a better alignment from a weaker starting point. Conversely, the 20k experiment peaks at iteration 1 and then declines. This could be because the initial Procrustes alignment with 20k pairs is already quite strong. Subsequent refinement iterations, while potentially finding more pairs based on the current (already good) alignment, might introduce noisier or less globally optimal pairs, causing the performance on the fixed evaluation set to degrade slightly. The system might be overfitting subtly to the characteristics of the pairs generated during refinement, especially when starting from a large, high-quality initial dictionary.

![Word Pair Alignment after Training](data/word_pair_alignment.png)

The figure above (`word_pair_alignment.png`) visualizes the alignment of English (blue) and Hindi (red) word embeddings after the supervised training process (using the best model from the 20k dictionary size experiment). Qualitatively, the alignment appears to have brought the embeddings from the two languages closer compared to their initial state (in the Data Preparation section with `comparison_en_vs_hi.png`). While the clouds are not perfectly overlapping, the increased intermingling suggests that the alignment process has been partially successful in mapping words with similar meanings closer together in the shared embedding space.


<!-- TODO: At this point, read `combined_best.log` and extract insights from that. First, we comment on overall statistics per dictionary size in the ablation study. Here we attach `all_metrics_by_dict_size_20250407_122416.png` and then discuss about the implications of varying dictionary sizes. We are purposefully storing the model with the highest average cosine similarity for every dict size.

After that we talk about individual refinement statistics. For each dictionary size, we discuss briefly about the contour of each of precisions and cosine similarity. Then we discuss about the correlation coefficient between precision and cosine similarity for each dictionary size. In this part, I want to highlight the differences in cosine similarity in refinement iterations for dict sizes; What I'm trying to say is if you observe the avg cosine similarity for dict size 1k, it's pretty much a flat line, but for 20k it first increases, and then reduces for the subsequent refinement iterations. I also want to discuss about the correlation between cosine similarity and precision.

In the end, we attach `word_pair_alignment.png`, showing that the separation between the two languages is a bit less well separated that what we showed in the "Data Preparation" section -->




These findings have practical implications for developing multilingual NLP systems, particularly for low-resource language pairs. Even with limited bilingual resources, effective cross-lingual embeddings can be constructed through careful optimization of the alignment process.

<!-- TODO: Add an additional section addressing the "Optional Extra Credit" pat of `assgn.txt`. For this part explain that, initially,  the same Data that was used for supervised Procrustes alignment learning was used; however, all P@1, P@5 and P@10 were zero with that config. That's where the usage of espanol comes in; Doing english to espanol translation in the first try failed, the result was all zero precisions again. So, I used the exact specifications and re-created the environment given in the paper (Elaborate these specifications and environment details by reading `paper.txt`). I realized that my discriminator was overfitting; it was essentially not getting "fooled" enough. By making the discriminator weaker (read and explain the args in weaker_disc_20250406_121320.log), I was able to achieve much more realistic results (realistic with reference to the paper) (pull results from weaker_disc_20250406_121320.log)
Now that we know that the unsupervised adverserially trained discriminator works well with a weak discriminator, I tried using hindi embeddings with these settings. However, the resulting precisions were still zero. The reason(s) were: (search the web for the reasons for why MUSE will struggle with translating Roman English to Devnagiri hindi, and give a brief synopsis). 
To mitigate that, the Devnagiri hindi was transliterated using IAST to roman. However, somehow, the 10k article limit at that point gave a total vocabulary of only around 60k unique words, which was significantly lower than the 200k I were using for english, so the constraint of using only 10000 wikipaedia articles was relaxed, and instead, 116035 articles were analyzed, which yielded a vocabulary of around 260k unique words. Top 200k were then used for adversarial training. We needed an evaluation set as well in the IAST-transliterated Hindi. So, using `transliterate_dict_new`, the supervised evaluation dict itself was transliterated through IAST. 
After this, give a brief discussion about the results of this approach. Pull the results from `hindi_latin_weak_disc_20250407_174830.log` 

Then inform the `run_unsupervised` exists for convenience, and explain each option in the file --->

<!-- TODO: Add section on unsupervised alignment experiments here -->

### 5. Unsupervised Alignment Experiments

Unsupervised alignment using adversarial training combined CSLS refinement, similar to the MUSE paper, was explored. Aligning English and Hindi, a distant language pair with different scripts, presented significant challenges compared to closer pairs or supervised methods.

**Initial Attempts and Strategy:**

The core issue with different scripts (Devanagari vs. Latin) is that the initial embedding spaces are structurally very different. An adversarial discriminator can easily distinguish between embeddings drawn from these spaces with high accuracy from the start. If the discriminator is too effective, it provides no useful gradient signal back to the mapping function (W), preventing the mapping from learning to align the spaces. Initial attempts using standard parameters for En-Hi resulted in zero precision for this reason.

The strategy therefore involved several steps:

1.  **Initial Attempts and Sanity Check:** The very first attempt focused on the target pair: English-Hindi (using original Devanagari embeddings). As anticipated due to the script difference and linguistic distance, this yielded zero precision even with standard MUSE parameters. To establish whether the unsupervised approach was viable at all under more favorable conditions, a sanity check was performed using English-Spanish (`en-es`), a closer language pair sharing the Latin script. However, even this initial En-Es run using standard parameters failed to produce meaningful alignment.

2.  **Successful Spanish Baseline:** Only after weakening the discriminator parameters (`dis_dropout=0.3`, `dis_input_dropout=0.3`, `map_beta=0.01`) compared to the original paper specifications did the English-Spanish alignment succeed. This successful run achieved a reasonable Precision@1 (CSLS) of 42.69% after 5 adversarial epochs and 5 refinement steps (documented in `unsupervised_logs/weaker_disc_20250406_121320.log`). This established a crucial baseline, confirming the general approach could work under less challenging conditions once parameters were appropriately tuned to prevent discriminator overfitting.

3.  **Hindi Transliteration and Vocabulary Expansion:** To mitigate the script difference issue, the Hindi text corpus was transliterated from Devanagari to Latin script (specifically ITRANS) using the `transliterate_dict_new.py` script (which leverages the `indic_transliteration` library). However, processing only the initial 10,000 Wikipedia articles yielded a vocabulary size significantly smaller than the 200k used for English after transliteration. To ensure comparable vocabulary sizes, the constraint was relaxed, and 116,035 Hindi articles were processed, yielding ~260k unique transliterated words. The top 200k words were then used to train the `data/wiki.hi_latin.vec` embeddings via `fastText_model.py`. The evaluation dictionary (`data/dictionaries/en-hi.5000-6500.txt`) was also transliterated using the same script to create `data/dictionaries/en-hi_latin.5000-6500_iast.txt`.

4.  **Parameter Tuning for En-Hi (Latin):** With the Spanish baseline established and Hindi transliterated to Latin script with a comparable vocabulary size, unsupervised alignment was then run between English and the *transliterated* Hindi (`en-hi_latin`). Even with the same script, the linguistic distance required further adjustments. Based on findings for distant pairs and the need to prevent the discriminator from becoming too strong too quickly, parameters were further adjusted (see `unsupervised_logs/hindi_latin_weak_disc_20250407_174830.log` and experiment 14 in `run_unsupervised.sh`):
    *   Further increased discriminator dropout (`dis_dropout=0.4`).
    *   Expanded the vocabulary considered by the discriminator (`dis_most_frequent=100000`).
    *   Used a higher learning rate for the mapping (`map_optimizer sgd,lr=0.2`).
    *   Allowed for longer training (25 adversarial epochs with early stopping patience 5, 8 refinement steps).

**Results (Unsupervised En-Hi Latin):**

The adversarial training phase stopped early after 13 epochs due to lack of improvement on the unsupervised validation metric. The refinement process ran for the full 8 iterations.
*   The best **Precision@1 (CSLS)** achieved was **34.91%** (after refinement iteration 6).
*   The best **Mean Cosine (CSLS)** was **0.586** (after refinement iteration 2).

These results, while non-zero, are significantly lower than the En-Es baseline and supervised methods, highlighting the persistent difficulty of unsupervised alignment for distant language pairs, even after addressing the script difference and tuning parameters.

**Experimentation Framework (`run_unsupervised.sh`):**

To facilitate these experiments, the `run_unsupervised.sh` script was created. This script acts as a convenient wrapper around the `unsupervised.py` script. It allows:
*   Selecting predefined experiment configurations via a menu (e.g., testing different learning rates, training durations, discriminator settings, or the specific Hindi-Latin setup described above).
*   Specifying whether to run on CPU or GPU.
*   Automatically logging the command, start/end times, and full output (stdout & stderr) to timestamped files in the `unsupervised_logs/` directory for reproducibility and analysis.
*   Attempting GPU memory cleanup between runs.
This script was used to run the Spanish baseline (option 1, although the log filename reflects a manual run) and the Hindi-Latin experiment (option 14).

Overall, while unsupervised methods offer a promising direction when parallel data is unavailable, achieving high-quality alignment for distant language pairs like English-Hindi requires careful strategy (transliteration, vocabulary matching, significant parameter tuning) and still lags considerably behind supervised performance.

These findings...
