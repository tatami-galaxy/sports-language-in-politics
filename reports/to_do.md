### Readings
- statistical significance tests
    - unpaired t test [2]

### Detecting sports language/terminology in politics

- How to detect metaphors
    - edit distance + semantic similarity, "someone", "something", etc.
        - vectorize code
        - validate semantic thresh 
        - check spellings

- Metaphors can mean different things
    - WSD [1]
        - t5-large-word-sense-disambiguation not working, find code/paper
        - other models [1]
    - other methods [1]

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
            - POS tagging -> [1]
        - validate -> random mappings, other methods from papers [1]
        - embedding bias (sports terms <-> political terms?) [2]
        - What similarity values are acceptable?
            - “While differences in word association are measurable and are often significant, small differences in cosine similarity are not reliable, especially for small corpora. If the intention of a study is to learn about a specific corpus, we recommend that practitioners test the statistical confidence of similarities based on word embeddings by training on multiple bootstrap samples”
        - T tests? [2]

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
    - fix datetime  [3]


### Data

- sports subs comments
- political subs comments
- random subs comments
- NOW dataset -> online news, sports etc. articles across 10 years
- political comments across years, months