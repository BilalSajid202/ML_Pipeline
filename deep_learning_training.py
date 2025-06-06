# deep_learning_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# Helper: Dynamic device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper: Convert pandas to tensors
def prepare_tensor_data(df, target_column):
    X = df.drop(columns=[target_column]).values.astype(np.float32)
    y = df[target_column].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y).astype(np.int64)
    return train_test_split(X, y_encoded, test_size=0.2, random_state=42), le

# DNN Model Class
class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64], num_classes=2):
        super(DNNClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], num_classes)
        )

    def forward(self, x):
        return self.model(x)

class DNNTrainer:
    def __init__(self, df, target_column='None', epochs=None):
        self.df = df
        self.target_column = target_column if target_column else df.columns[-1]
        self.epochs = epochs if epochs else 20

    def train(self):
        (X_train, X_test, y_train, y_test), label_encoder = prepare_tensor_data(self.df, self.target_column)
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))

        model = DNNClassifier(input_dim, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"📦 Training DNN for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        print("✅ DNN Classification Report")
        print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_.astype(str)))
        return model

# TabNet Classifier Trainer
class TabNetTrainer:
    def __init__(self, df, target_column=None, epochs=None):
        self.df = df
        self.target_column = target_column if target_column else df.columns[-1]
        self.epochs = epochs if epochs else 50

    def train(self):
        (X_train, X_test, y_train, y_test), label_encoder = prepare_tensor_data(self.df, self.target_column)
        model = TabNetClassifier(verbose=1)
        print(f"📦 Training TabNet for {self.epochs} epochs...")
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=self.epochs,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128
        )

        preds = model.predict(X_test)
        print("✅ TabNet Classification Report")
        print(classification_report(y_test, preds, target_names=label_encoder.classes_.astype(str)))
        return model

# FT-Transformer Trainer (only if user has `rtdl` library installed)
try:
    from rtdl import FTTransformer
    import torch_optimizer

    class FTTransformerTrainer:
        def __init__(self, df, target_column=None, epochs=None):
            self.df = df
            self.target_column = target_column if target_column else df.columns[-1]
            self.epochs = epochs if epochs else 30

        def train(self):
            (X_train, X_test, y_train, y_test), label_encoder = prepare_tensor_data(self.df, self.target_column)

            train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
            test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=256)

            model = FTTransformer(
                d_numerical=X_train.shape[1],
                d_out=len(np.unique(y_train)),
                n_layers=3,
                d_token=192,
                n_heads=8
            ).to(device)

            optimizer = torch_optimizer.AdamP(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            print(f"📦 Training FT-Transformer for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model(inputs)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss:.4f}")

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    output = model(inputs)
                    _, preds = torch.max(output, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())

            print("✅ FT-Transformer Classification Report")
            print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_.astype(str)))
            return model
except ImportError:
    print("⚠️ FT-Transformer skipped: `rtdl` not installed.")

