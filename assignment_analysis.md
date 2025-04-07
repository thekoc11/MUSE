# Cross-Lingual Word Embedding Alignment: English to Hindi

## 1. Introduction

This report presents the implementation and evaluation of a supervised cross-lingual word embedding alignment system for English and Hindi languages using the Procrustes method. Cross-lingual word embeddings enable mapping words from different languages into a shared semantic space, which is fundamental for multilingual NLP tasks such as machine translation, cross-lingual information retrieval, and knowledge transfer.

<!-- TODO: A section need to be added for environment setup. We need to explain what mamba is and how it is different from miniconda but still essentially the same. Then, we need to give directions to install mamba and create a new mamba environment called "muse", with python=3.8.20. For this, we may need to search the web, and then write this part on the basis of our search results. Then we need to provide instructions on how to use mamba install to install pytorch, with the appropriate cuda version, and plotly. Finally, we run like `pip install -r requirements.txt` -->

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

<!--- TODO: The whole document should be written in a passive voice and third person, much like a typical research article / report --->

## 3. Implementation

### 3.1 Data Preparation

<!-- TODO: Describe comprehensively the steps taken in `fastTest_model.py`. Explain that this was done without any transliteration for english and espanol, and both with and without transliteration for Hindi. The vectorization largely is done as specfied in `paper.txt`(cite sections of the paper as reference), for all the languages except for transliterated Hindi. Then address the "Data Preparation" Section in `assgn.txt`, i.e. the original assignment. English and non-transliterated Hindi follow the exact criteria mentioned in the section. For addressing the point c., inform that the following command was run:

```bash
cd data/
wget https://dl.fbaipublicfiles.com/arrival/wordsim.tar.gz
wget https://dl.fbaipublicfiles.com/arrival/dictionaries.tar.gz
``` -->


### 3.2 Dictionary Construction
<!-- TODO: Remove this section. I tried doing this to improve performance, and it did improve performance marginally, but I don't have the before and after numbers to justify this section -->
I implemented a method to combine multiple dictionary files (both en-hi and hi-en) to create larger training dictionaries. This process involved:
1. Loading dictionaries from multiple sources
2. Filtering out validation pairs
3. Converting word pairs to embedding indices
4. Sampling subsets of various sizes for my ablation study

### 3.3 Orthogonality Verification

To ensure the mapping preserved distances and angles, I implemented a verification function that checks whether $W^TW$ is close to the identity matrix:

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
### 4.1 Ablation Study on Dictionary Size

As per the assignment requirements, I first conducted an ablation study to assess the impact of bilingual lexicon size on alignment quality. The experiments were run with dictionary sizes of:
- 1,000 word pairs
- 2,000 word pairs
- 5,000 word pairs
- 10,000 word pairs
- 20,000 word pairs

For each size, I performed 5 refinement iterations to improve the alignment quality.

### 4.2 Enhancement Experiments

After identifying the optimal dictionary size (20,000 pairs), I conducted additional experiments to further improve the alignment quality:

1. **Dictionary Building Methods**:
   - **S2T&T2S**: The default approach using intersection of source-to-target and target-to-source dictionaries
   - **S2T|T2S**: Using the union of source-to-target and target-to-source dictionaries

2. **Similarity Measures**:
   - **csls_knn_10**: Default Cross-domain Similarity Local Scaling with 10 nearest neighbors
   - **csls_knn_15**: CSLS with 15 nearest neighbors for improved hubness mitigation

3. **Embedding Normalization**:
   - **center,renorm**: Centering embeddings and renormalizing them to unit length

4. **Combined Approach**:
   - Integration of the best performing techniques: S2T|T2S with csls_knn_15 and embedding normalization

## 5. Results and Analysis

### 5.1 Alignment Quality Metrics

I evaluated the alignment using multiple metrics:

1. **Word Translation Accuracy**:
   - Precision@1: Percentage of source words for which the correct translation is the top candidate
   - Precision@5: Percentage of source words for which the correct translation is among the top 5 candidates

2. **Similarity Methods**:
   - Nearest Neighbor (NN): Simple vector similarity
   - Cross-domain Similarity Local Scaling (CSLS): Addresses the hubness problem in high-dimensional spaces

3. **Semantic Similarity**:
   - Average cosine similarity between aligned word pairs

### 5.2 Ablation Study Results: Dictionary Size Impact

| Dictionary Size | P@1 (NN) | P@1 (CSLS) | P@5 (NN) | P@5 (CSLS) | Avg. Cosine Similarity |
|-----------------|----------|------------|----------|------------|------------------------|
| 1,000           | 29.73%   | 36.03%     | 45.68%   | 54.18%     | 0.4730                 |
| 2,000           | 30.82%   | 36.44%     | 45.27%   | 52.88%     | 0.4742                 |
| 5,000           | 30.14%   | 35.82%     | 46.58%   | 54.52%     | 0.4745                 |
| 10,000          | 30.14%   | 36.37%     | 45.41%   | 54.93%     | 0.4744                 |
| 20,000          | 30.89%   | 37.74%     | 46.99%   | 54.25%     | 0.4755                 |

Key observations from the ablation study:

1. **Positive Correlation**: There is a positive correlation between dictionary size and alignment quality, with the 20,000 pair dictionary achieving the best overall performance.

2. **Diminishing Returns**: The improvement from 1,000 to 20,000 pairs is modest: only +1.71 percentage points for P@1 (CSLS).

3. **Quality vs. Quantity**: Most of the performance is achieved with just 1,000 word pairs, suggesting that quality of pairs may be more important than quantity.

4. **CSLS Advantage**: CSLS consistently outperforms nearest neighbor search across all dictionary sizes, with an average improvement of 6-7 percentage points in Precision@1.

Based on these results, I identified the 20,000 word pair dictionary as optimal and used it for further enhancement experiments.

### 5.3 Enhancement Experiments on 20,000 Word Pair Dictionary

| Configuration | P@1 (NN) | P@1 (CSLS) | P@5 (NN) | P@5 (CSLS) | Avg. Cosine Similarity |
|---------------|----------|------------|----------|------------|------------------------|
| Baseline (20k) | 30.89%   | 37.74%     | 46.99%   | 54.25%     | 0.4755                 |
| S2T\|T2S Build | 30.21%   | 37.47%     | 47.26%   | 55.14%     | 0.4717                 |
| CSLS KNN 15    | 30.96%   | 37.40%     | 46.92%   | 54.79%     | 0.4752                 |
| Combined Best  | 32.40%   | 37.81%     | 49.38%   | 56.16%     | 0.3847                 |

![Precision Results](precision_results.html)

Key findings from enhancement experiments:

1. **Dictionary Building Method**: 
   - The S2T|T2S (union) approach showed better results for Precision@5 metrics compared to the default approach
   - This method creates a larger dictionary by including word pairs from both translation directions

2. **CSLS Parameter Tuning**:
   - Increasing the number of nearest neighbors in CSLS from 10 to 15 slightly improved results
   - This modification helps further address the hubness problem in the embedding space

3. **Combined Approach**:
   - The best results were achieved by combining S2T|T2S with csls_knn_15 and normalize_embeddings "center,renorm"
   - This configuration improved Precision@1 (NN) by +1.51 percentage points
   - Precision@5 (NN) improved by +2.39 percentage points
   - Precision@5 (CSLS) improved by +1.91 percentage points

### 5.4 Cross-Lingual Semantic Similarity Analysis

Cosine similarity between word pairs serves as a direct measure of semantic alignment in the shared embedding space. I computed the average cosine similarity between ground truth word pairs from the evaluation dictionary to assess how well the semantic relationships are preserved across languages.

#### 5.4.1 Impact of Dictionary Size on Semantic Similarity

The average cosine similarity shows a slight upward trend with increasing dictionary size:
- 1,000 pairs: 0.4730
- 2,000 pairs: 0.4742 (+0.0012)
- 5,000 pairs: 0.4745 (+0.0003)
- 10,000 pairs: 0.4744 (-0.0001)
- 20,000 pairs: 0.4755 (+0.0011)

The total improvement from 1,000 to 20,000 pairs is only +0.0025, indicating that:
- Values around 0.47-0.48 represent moderate semantic alignment between English and Hindi
- The marginal improvement with larger dictionaries suggests that even small dictionaries can achieve reasonable semantic alignment
- There is a saturation effect in semantic similarity that mirrors the pattern observed in translation accuracy

#### 5.4.2 Semantic Similarity in Enhancement Experiments

Interestingly, the cosine similarity patterns in the enhancement experiments reveal important insights:

1. **S2T|T2S Union Approach**: 
   - Average cosine similarity (0.4717) is slightly lower than the baseline (0.4755)
   - However, this approach yields better Precision@5 scores, suggesting that the union method may sacrifice some direct similarity for better overall alignment structure

2. **CSLS KNN 15**:
   - Maintains nearly identical cosine similarity (0.4752) to the baseline
   - This confirms that adjusting the CSLS parameter primarily affects retrieval performance rather than the underlying semantic alignment

3. **Combined Best Approach**:
   - Shows a notably lower average cosine similarity (0.3847)
   - Despite this, it achieves the highest precision scores across all metrics
   - This apparent contradiction suggests that the normalization process ("center,renorm") changes the scale of cosine similarities while improving the overall alignment quality

#### 5.4.3 Relationship Between Cosine Similarity and Translation Accuracy

The data reveals an important insight: higher average cosine similarity does not always correlate with better translation accuracy. This can be explained by:

1. **Different Optimization Objectives**:
   - Procrustes optimization maximizes overall alignment rather than individual pair similarities
   - Translation accuracy depends on relative rankings of cosine similarities, not their absolute values

2. **Distribution of Similarities**:
   - Average cosine similarity can mask the distribution of similarities across word pairs
   - A lower average with better relative rankings can yield superior translation performance

3. **Embedding Normalization Effects**:
   - The "center,renorm" procedure in the Combined Best approach changes the distribution of cosine similarities
   - This appears to improve the structural properties of the embedding space, enhancing retrieval performance despite lowering raw similarity scores

This analysis demonstrates that while cosine similarity provides a direct measure of semantic alignment, translation accuracy metrics offer a more practical evaluation of the embedding quality for downstream applications.

### 5.5 Refinement Process Analysis

The refinement iterations showed different patterns depending on the dictionary building method:

1. **Default Approach (S2T&T2S)**:
   - Initial alignment with manual dictionary
   - Automatic construction of a new dictionary (~3,100 pairs)
   - Modest growth through iterations, eventually stabilizing around 3,200 pairs

2. **Union Approach (S2T|T2S)**:
   - Initial alignment with manual dictionary
   - Larger automatic dictionary construction (~7,100 pairs)
   - Continued growth to approximately 8,900 pairs before stabilizing

The union approach identified substantially more reliable word mappings, contributing to its improved performance, especially for Precision@5 metrics.

## 6. Conclusions

My implementation and evaluation of the supervised cross-lingual word embedding alignment system for English and Hindi using the Procrustes method reveals several important insights:

1. **Dictionary Size Impact**: Larger bilingual dictionaries generally improve alignment quality, but with diminishing returns. The 20,000 word pair dictionary provided the best balance of performance and efficiency.

2. **CSLS Superiority**: The consistent outperformance of CSLS over nearest neighbor search confirms the importance of addressing the hubness problem in cross-lingual embedding spaces.

3. **Enhancement Strategies**: Several strategies can further improve the baseline performance:
   - Using the union approach for dictionary building rather than intersection
   - Fine-tuning CSLS parameters with more nearest neighbors
   - Applying appropriate embedding normalization techniques

4. **Combined Optimizations**: The best performance was achieved by combining multiple enhancement strategies, resulting in meaningful improvements across all precision metrics.

5. **Orthogonality Preservation**: All mappings maintained orthogonality (error < 1e-4), confirming that the Procrustes solution successfully preserves the structural relationships in the embedding space.

These findings have practical implications for developing multilingual NLP systems, particularly for low-resource language pairs. Even with limited bilingual resources, effective cross-lingual embeddings can be constructed through careful optimization of the alignment process.

<!-- TODO: Add an additional section addressing the "Optional Extra Credit" pat of `assgn.txt`. For this partm explain that, initially,  the same Data that was used for supervised Procrustes alignment learning was used; however, all P@1, P@5 and P@10 were zero with that config. That's where the usage of espanol comes in; Doing english to espanol translation in the first try failed, the result was all zero precisions again. So, I used the exact specifications and re-created the environment given in the paper (Elaborate these specifications and environment details by reading `paper.txt`). I realized that my discriminator was overfitting; it was essentially not getting "fooled" enough. By making the discriminator weaker (read and explain the args in weaker_disc_20250406_121320.log), I was able to achieve much more realistic results (realistic with reference to the paper) (pull results from weaker_disc_20250406_121320.log)
Now that we know that the unsupervised adverserially trained discriminator works well with a weak discriminator, I tried using hindi embeddings with these settings. However, the resulting precisions were still zero. The reason(s) were: (search the web for the reasons for why MUSE will struggle with translating Roman English to Devnagiri hindi, and give a brief synopsis). 
To mitigate that, the Devnagiri hindi was transliterated using IAST to roman. However, somehow, the 10k article limit at that point gave a total vocabulary of only around 60k unique words, which was significantly lower than the 200k I were using for english, so the constraint of using only 10000 wikipaedia articles was relaxed, and instead, 116035 articles were analyzed, which yielded a vocabulary of around 260k unique words. Top 200k were then used for adversarial training. We needed an evaluation set as well in the IAST-transliterated Hindi. So, using `transliterate_dict_new`, the supervised evaluation dict itself was transliterated through IAST. 
After this, give a brief discussion about the results of this approach. Pull the results from `hindi_latin_weak_disc_20250406_214958.log` 

Then inform the `run_unsupervised` exists for convenience, and explain each option in the file --->
