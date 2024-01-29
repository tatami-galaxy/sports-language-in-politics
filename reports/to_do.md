### Readings
- No country for old members
- Political homophily
- Quantifying political polarization
- Immigration framing
- statistical significance tests
    - unpaired t test [2]

### Detecting sports language/terminology in politics

- Metaphors can mean different things
    - how to select metaphors?
        - frequency/TF-IDF using NOW dataset
            - filter sports articles first -> measure freqeuncy of metaphors from rest
            - how to select topk?
                - does it matter? -> try different topk
                - dont filter for now 

- Edit Metaphor csv 
    - re-verify metaphors
    - check spellings
    - remove duplicates

- How to detect metaphors
    - edit distance + semantic similarity
        - store edit, semantic values, comments / comment ids
        - "someone", "something", etc.  [1]
            - for the ones at the beginning or end, just remove something/someone. in the middle 10 edit distance  [1]
        - vectorize code
        - validate semantic thresh  [3]

- Sport analogies [1]
    - MASK [1]
    - embeddings [1]
        - train for longer
        - multiple subs?
        - plot + calc distances
        - increase vocab

- Analyze data 
    - build classifier to remove sports and gaming subs 
        - ngram, bert etc.
    - use classifier to filter remaining data

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