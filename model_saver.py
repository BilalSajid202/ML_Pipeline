# 7_model_saver.py

import pickle

class ModelSaver:
    """Class to save and load machine learning models using pickle."""

    def __init__(self, filename='trained_model.pkl'):
        """
        Initialize ModelSaver with a default filename.
        
        Args:
            filename (str): Path to save or load the model.
        """
        self.filename = filename

    def save(self, model):
        """
        Save the given model to the specified pickle file.
        
        Args:
            model: Trained model object to save.
        """
        with open(self.filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✅ Model saved as {self.filename}")

    def load(self):
        """
        Load and return a model from the specified pickle file.
        
        Returns:
            Loaded model object.
        """
        with open(self.filename, 'rb') as f:
            model = pickle.load(f)
        print(f"\n✅ Model loaded from {self.filename}")
        return model
