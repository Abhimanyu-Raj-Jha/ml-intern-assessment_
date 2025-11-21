Evaluation & Theoretical Design Choices

1. Mathematical Foundation

This implementation is based on the Chain Rule of Probability, approximating the joint probability of a sentence $P(w_1...w_n)$ using the Markov Assumption.

The Markov Assumption

We assume that the probability of a word depends only on the previous $n-1$ words. For this Trigram model ($n=3$):


$$P(w_i | w_1...w_{i-1}) \approx P(w_i | w_{i-2}, w_{i-1})$$

2. Smoothing & Interpolation

To handle sparsity (the zero-probability problem), I moved beyond simple Laplace smoothing to Linear Interpolation.

Linear Interpolation

Instead of relying solely on the Trigram count, we compute a weighted sum of Trigram, Bigram, and Unigram probabilities:


$$\hat{P}(w_3|w_1,w_2) = \lambda_1 P_{tri}(w_3|w_1,w_2) + \lambda_2 P_{bi}(w_3|w_2) + \lambda_3 P_{uni}(w_3)$$


Where $\sum \lambda_i = 1$.

Why? This utilizes the "Backoff" concept: when specific trigram evidence is weak, the model falls back to the more robust bigram or unigram statistics, ensuring better generalization on unseen test data.

3. Metrics & Evaluation

I implemented a quantitative evaluation script (src/evaluate.py) to measure:

Perplexity ($PP$): The inverse probability of the test set, normalized by the number of words. It measures how "surprised" the model is by real data. Lower is better.

Recall@K: Often used in recommendation systems (and autocomplete), this measures the percentage of time the true next word appears in the model's top $K$ predictions.

4. Deep Learning Extension (Task 2)

I implemented Scaled Dot-Product Attention, the core mathematical operation in Transformers.

Dense Representation: Unlike the discrete N-gram counts, attention operates on dense vectors (Query, Key, Value), allowing for capturing semantic relationships beyond simple adjacency.

5. Sampling Strategy

The generation engine supports:

Greedy Search (Temperature=0): Deterministic, picks the maximum likelihood token.

Stochastic Sampling: Uses the probability distribution to sample tokens, increasing diversity.