# 7_model_saver.py

import pickle

def save_model(model, filename='trained_model.pkl'):
    """Saves model to pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved as {filename}")
