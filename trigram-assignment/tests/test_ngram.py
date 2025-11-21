import pytest
import sys
import os
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ngram_model import TrigramModel

@pytest.fixture
def model():
    return TrigramModel(alpha=1.0)

@pytest.fixture
def simple_corpus():
    return [
        "I love machine learning",
        "I love coding in python",
        "machine learning is fun"
    ]

def test_training_and_counts(model, simple_corpus):
    model.train(simple_corpus)
    # Check trigram count: (i, love) -> machine
    assert model.trigram_counts[('i', 'love')]['machine'] == 1
    # Check bigram backoff count: (love) -> coding
    assert model.bigram_model_counts['love']['coding'] == 1

def test_autocomplete_logic(model, simple_corpus):
    model.train(simple_corpus)
    
    # Context: "I love" -> should suggest "machine" and "coding"
    suggestions = model.predict_next('i', 'love', k=5)
    words = [w for w, p in suggestions]
    
    assert 'machine' in words
    assert 'coding' in words
    assert len(suggestions) <= 5

def test_backoff_logic(model, simple_corpus):
    model.train(simple_corpus)
    
    # Context: "we love" (trigram "we love" is unseen)
    # Should backoff to bigram "love" -> suggests "machine", "coding"
    suggestions = model.predict_next('we', 'love')
    words = [w for w, p in suggestions]
    
    # Since "love" was followed by "machine" and "coding" in training, 
    # backoff should find them.
    assert 'machine' in words or 'coding' in words

def test_generation_robustness(model, simple_corpus):
    model.train(simple_corpus)
    sent = model.generate_sentence()
    assert isinstance(sent, str)
    assert len(sent) > 0

def test_save_load(model, simple_corpus):
    model.train(simple_corpus)
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        model.save_model(tmp.name)
        loaded_model = TrigramModel.load_model(tmp.name)
        
    # Verify loaded model works
    assert loaded_model.vocab == model.vocab
    assert loaded_model.score('i', 'love', 'machine') == model.score('i', 'love', 'machine')
    os.unlink(tmp.name)