### Papers
- No country for old members
- Political homophily
- Quantifying political polarization

### Detecting sports language/terminology in politics

- Metaphors can mean different things
    - how to select metaphors?
        - frequency/TF-IDF using NOW dataset
            - filter sports articles first -> measure freqeuncy of metaphors from rest

- Edit Metaphor csv [1]

- How to detect metaphors
    - edit distance + semantic similarity
        - vectorize code [2]
        - edit distance threshold based on n-gram
        - validate semantic thresh

- Analyze data 
    - build classifier to remove sports and gaming subs 
    - use classifier to filter remaining data [1]

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
    - fix datetime  [2]


### Data

- sports subs comments
- political subs comments
- random subs comments
- NOW dataset -> online news, sports etc. articles across 10 years
- political comments across years, months