DesibleAI Intern Assessment - Trigram Model & Attention

A comprehensive NLP submission containing a production-grade Trigram Language Model and a Scaled Dot-Product Attention implementation.

Project Structure

ml-intern-assessment/
├── demo.py                     
├── evaluation.md               
├── requirements.txt            
└── trigram-assignment/
    ├── src/
    │   ├── ngram_model.py      
    │   └── attention 
    |        └──attention.py      
    └── tests/
        └── test_ngram.py       


Quick Start

1. Setup

pip install -r requirements.txt


2. Run Trigram Demo (Task 1)

Downloads "Sherlock Holmes" and launches the autocomplete engine.

python demo.py


3. Run Attention Demo (Task 2)

Demonstrates the scaled dot-product attention mechanism.

python trigram-assignment/src/attention/attention.py


4. Run Tests

pytest trigram-assignment/tests/test_ngram.py


Key Features

Robustness: Handles <UNK> tokens and uses 3-level Backoff (Tri->Bi->Uni).

Flexibility: Supports both Deterministic (Greedy) and Stochastic text generation.

Modern Architecture: Includes attention.py implementing Transformer-style attention.