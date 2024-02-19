### Detecting sports language/terminology in politics

- How to detect metaphors
    - edit distance + semantic similarity, "someone", "something", etc.
        - validate semantic thresh 
        - check spellings
        - t test [1]

- Metaphors can mean different things
    - WSD
        - gloss-bert [2]

- Sport analogies
    - MASK
        - compare MASK replacement probabilities with target probabilities
            - average values
            - comments with negative difference
    - word2vec [1]
        - cosine score low : How to fix?
            - seeds -> not changing values
            - stop words -> not changing values
            - epochs -> not changing values
            - embeddings norm value -> not changing values (maybe slightly higher is better?)
            - context window -> not changing values
            - min word frequency -> not changing values
            - 20 context length -> not changing values
            - ignore UNK -> not changing values
            - POS tagging -> not changing values
            - train longer -> 1000 epochs
        - validate
            - intrinsic (https://arxiv.org/pdf/1901.09785.pdf)
                - king, queen (analogy) [1]
                - target words -> cosine, euclid [1]
                - cluster into categories (k-means) [1]
                - outlier detection [1]
            - extrinsic
                - POS [1]
                - NER [1]
                - chunking [2]
        - embedding bias (sports terms <-> political terms?)
        - What similarity values are acceptable?
            - “While differences in word association are measurable and are often significant, small differences in cosine similarity are not reliable, especially for small corpora. If the intention of a study is to learn about a specific corpus, we recommend that practitioners test the statistical confidence of similarities based on word embeddings by training on multiple bootstrap samples”
                - bootstrap [2]
        - cosine vs euclidean [2]
        - T test [1]
    - BERT embeddings
        - static embeddings using contextual embedding (https://aclanthology.org/2021.acl-long.408.pdf) [1]

- How to calculate lm probabilites?
    - analyze
    - calculate with generation
    - multiple samples
    - log probs with backoff

- Autoencoders?

- How to interpret Shapley values

- Style transfer -> LM + paraphrasing?


### Analyzing sports language in politics

- Month wise metaphor frequency in political comments
    - fix datetime


### Data

- sports subs comments
- political subs comments
- random subs comments
- NOW dataset -> online news, sports etc. articles across 10 years
- political comments across years, months