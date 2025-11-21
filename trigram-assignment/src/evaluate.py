import os
import sys
import math
import random

# Add src to path
sys.path.append(os.path.dirname(__file__))
from ngram_model import TrigramModel

def calculate_perplexity(model, test_sentences):
    """
    Calculates Perplexity (PP) on test data.
    PP = exp(-1/N * sum(log P(w_i | context)))
    """
    log_prob_sum = 0
    N = 0
    
    for sentence in test_sentences:
        tokens = model.preprocess(sentence)
        padded = ['<START>', '<START>'] + tokens + ['<END>']
        
        for i in range(len(padded) - 2):
            w1, w2, w3 = padded[i], padded[i+1], padded[i+2]
            prob = model.score(w1, w2, w3)
            log_prob_sum += math.log(prob)
            N += 1
            
    if N == 0: return float('inf')
    return math.exp(-log_prob_sum / N)

def calculate_recall_at_k(model, test_sentences, k=3):
    """
    Calculates Recall@K: How often is the actual next word 
    present in the model's top K predictions?
    """
    correct_predictions = 0
    total_predictions = 0
    
    for sentence in test_sentences:
        tokens = model.preprocess(sentence)
        if len(tokens) < 3: continue
        
        for i in range(len(tokens) - 2):
            w1, w2 = tokens[i], tokens[i+1]
            actual_next = tokens[i+2]
            
            # Get model predictions
            predictions = model.predict_next(w1, w2, k=k)
            predicted_words = [p[0] for p in predictions]
            
            if actual_next in predicted_words:
                correct_predictions += 1
            total_predictions += 1
            
    if total_predictions == 0: return 0.0
    return correct_predictions / total_predictions

def main():
    # Simple dummy data or load Sherlock if available
    try:
        from demo import download_and_clean_data
        corpus = download_and_clean_data()
        # Split 80/20
        split_idx = int(len(corpus) * 0.8)
        train_data = corpus[:split_idx]
        test_data = corpus[split_idx:]
        print(f"\nData Split: {len(train_data)} Train, {len(test_data)} Test")
    except:
        print("Using dummy data for evaluation.")
        train_data = ["the quick brown fox", "jumped over the lazy dog"]
        test_data = ["the quick brown fox"]

    model = TrigramModel(alpha=0.01)
    model.train(train_data)
    
    print("\n--- Evaluation Metrics ---")
    
    # 1. Perplexity
    pp = calculate_perplexity(model, test_data)
    print(f"Perplexity (Lower is better): {pp:.2f}")
    
    # 2. Recall@3 (Accuracy of Autocomplete)
    recall = calculate_recall_at_k(model, test_data, k=3)
    print(f"Recall@3 (Higher is better):  {recall:.2%}")
    
    print("--------------------------")

if __name__ == "__main__":
    main()