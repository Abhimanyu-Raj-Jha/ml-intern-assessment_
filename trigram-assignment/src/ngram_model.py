import random
import math
import re
import pickle
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

class TrigramModel:
    def __init__(self, alpha: float = 1.0):
        """
        Trigram Language Model.
        
        Theory Implemented:
        - Markov Assumption: P(w_i | w_1...w_{i-1}) approx P(w_i | w_{i-2}, w_{i-1})
        - Laplace Smoothing: Add-alpha to handle zero counts.
        - Linear Interpolation: Weighted average of Trigram, Bigram, and Unigram models.
        """
        self.alpha = alpha
        
        # Sparse mappings for N-gram counts
        self.trigram_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.bigram_model_counts: Dict[str, Counter] = defaultdict(Counter) # w2 -> w3
        self.bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)   # (w1, w2) count
        self.unigram_counts: Dict[str, int] = defaultdict(int)              # w2 count
        self.total_word_count = 0
        
        self.vocab: set = set()

    def preprocess(self, text: str) -> List[str]:
        """Tokenizes text: lowercase, removes punctuation."""
        return re.findall(r'\b\w+\b', text.lower())

    def train(self, corpus: List[str]):
        print(f"Training on {len(corpus)} sentences...")
        for sentence in corpus:
            tokens = self.preprocess(sentence)
            if not tokens: continue
                
            # Padding for Trigram context: <START>, <START> ... <END>
            padded = ['<START>', '<START>'] + tokens + ['<END>']
            
            self.vocab.update(tokens)
            self.vocab.add('<END>')
            self.vocab.add('<UNK>') 
            self.total_word_count += len(tokens)

            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
                
                # Count occurrences for Maximum Likelihood Estimation (MLE)
                self.trigram_counts[(w1, w2)][w3] += 1
                self.bigram_counts[(w1, w2)] += 1
                
                self.bigram_model_counts[w2][w3] += 1
                self.unigram_counts[w2] += 1

    def _get_word(self, word: str) -> str:
        """Handles Out-Of-Vocabulary (OOV) words by mapping to <UNK>."""
        return word if word in self.vocab else '<UNK>'

    def score(self, w1: str, w2: str, w3: str) -> float:
        """
        Calculates Probability using Linear Interpolation.
        P_hat(w3 | w1,w2) = λ1*P(tri) + λ2*P(bi) + λ3*P(uni)
        """
        w1 = self._get_word(w1.lower())
        w2 = self._get_word(w2.lower())
        w3 = self._get_word(w3.lower())
        
        # Hyperparameters for Interpolation (must sum to 1.0)
        # These allow the model to "Backoff" to simpler models if data is sparse
        l1, l2, l3 = 0.7, 0.2, 0.1
        
        # 1. Trigram Probability (with Laplace Smoothing)
        tri_num = self.trigram_counts[(w1, w2)][w3] + self.alpha
        tri_den = self.bigram_counts[(w1, w2)] + (self.alpha * len(self.vocab))
        p_tri = tri_num / tri_den
        
        # 2. Bigram Probability
        bi_num = self.bigram_model_counts[w2][w3] + self.alpha
        bi_den = self.unigram_counts[w2] + (self.alpha * len(self.vocab))
        p_bi = bi_num / bi_den
        
        # 3. Unigram Probability
        uni_num = self.unigram_counts[w3] + self.alpha
        uni_den = self.total_word_count + (self.alpha * len(self.vocab))
        p_uni = uni_num / uni_den
        
        return (l1 * p_tri) + (l2 * p_bi) + (l3 * p_uni)

    def predict_next(self, w1: str, w2: str, k: int = 3) -> List[Tuple[str, float]]:
        """Returns top-k likely words (used for Recall@K evaluation)."""
        w1, w2 = self._get_word(w1.lower()), self._get_word(w2.lower())
        
        # Try Trigram context first
        candidates = self.trigram_counts[(w1, w2)]
        
        # Backoff to Bigram
        if not candidates:
            candidates = self.bigram_model_counts[w2]
            
        # Backoff to Unigram (Top frequent words)
        if not candidates:
            top_uni = sorted(self.unigram_counts.items(), key=lambda x: x[1], reverse=True)[:k]
            total = self.total_word_count or 1
            return [(w, c/total) for w, c in top_uni]

        total_count = sum(candidates.values())
        probs = [(w, count / total_count) for w, count in candidates.items()]
        probs.sort(key=lambda x: x[1], reverse=True)
        
        return probs[:k]

    def generate_sentence(self, max_length: int = 20, temperature: float = 1.0) -> str:
        """
        Generates text using Sampling.
        temp=0 -> Greedy (Deterministic)
        temp>0 -> Weighted Random (Stochastic)
        """
        if not self.trigram_counts: return "Model not trained."

        current_bigram = ('<START>', '<START>')
        sentence = []
        
        for _ in range(max_length):
            candidates = self.trigram_counts[current_bigram]
            if not candidates:
                candidates = self.bigram_model_counts[current_bigram[1]]
            
            if not candidates: break
                
            words = list(candidates.keys())
            counts = list(candidates.values())
            
            if temperature == 0:
                next_word = words[counts.index(max(counts))]
            else:
                next_word = random.choices(words, weights=counts, k=1)[0]
            
            if next_word == '<END>': break
                
            sentence.append(next_word)
            current_bigram = (current_bigram[1], next_word)
            
        return ' '.join(sentence)

    def save_model(self, path: str):
        with open(path, 'wb') as f: pickle.dump(self, f)

    @staticmethod
    def load_model(path: str) -> 'TrigramModel':
        with open(path, 'rb') as f: return pickle.load(f)