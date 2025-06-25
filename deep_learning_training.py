# deep_learning_training.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper: Prepare tensor data and auto-detect task type
def prepare_tensor_data(df, target_column):
    X = df.drop(columns=[target_column]).values.astype(np.float32)
    y = df[target_column].values

    if np.issubdtype(df[target_column].dtype, np.number) and len(np.unique(y)) > 20:
        task_type = 'regression'
        y_tensor = y.astype(np.float32)
        label_encoder = None
    else:
        task_type = 'classification'
        le = LabelEncoder()
        y_tensor = le.fit_transform(y).astype(np.int64)
        label_encoder = le

    return train_test_split(X, y_tensor, test_size=0.2, random_state=42), label_encoder, task_type

# DNN Model Class
class DNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64], num_classes=2, task_type='classification'):
        super(DNNClassifier, self).__init__()
        self.task_type = task_type
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], 1 if task_type == 'regression' else num_classes)
        )

    def forward(self, x):
        return self.model(x)

# DNN Trainer
class DNNTrainer:
    def __init__(self, df, target_column='None', epochs=None):
        self.df = df
        self.target_column = target_column if target_column else df.columns[-1]
        self.epochs = epochs if epochs else 20

    def train(self):
        (X_train, X_test, y_train, y_test), label_encoder, task_type = prepare_tensor_data(self.df, self.target_column)
        self.task_type = task_type

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if task_type == 'regression' else torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if task_type == 'regression' else torch.long)

        train_dataset = TensorDataset(torch.tensor(X_train), y_train_tensor)
        test_dataset = TensorDataset(torch.tensor(X_test), y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train)) if task_type == 'classification' else 1

        model = DNNClassifier(input_dim, num_classes=num_classes, task_type=task_type).to(device)
        criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print(f"📦 Training DNN ({task_type}) for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss:.4f}")

        # Evaluation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze().cpu().numpy()
                preds.extend(outputs)
                true.extend(labels.numpy())

        print(f"\n✅ DNN {task_type.capitalize()} Report")
        if task_type == 'regression':
            mae = mean_absolute_error(true, preds)
            mse = mean_squared_error(true, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(true, preds)
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
        else:
            preds = np.array(preds).argmax(axis=1) if len(np.array(preds).shape) > 1 else np.array(preds)
            print(classification_report(true, preds, target_names=label_encoder.classes_.astype(str)))

        return model

# TabNet Trainer
class TabNetTrainer:
    def __init__(self, df, target_column=None, epochs=None):
        self.df = df
        self.target_column = target_column if target_column else df.columns[-1]
        self.epochs = epochs if epochs else 50

    def train(self):
        (X_train, X_test, y_train, y_test), label_encoder, task_type = prepare_tensor_data(self.df, self.target_column)
        self.task_type = task_type

        model_cls = TabNetRegressor if task_type == 'regression' else TabNetClassifier
        model = model_cls(verbose=1)

        print(f"📦 Training TabNet ({task_type}) for {self.epochs} epochs...")
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=self.epochs,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128
        )

        preds = model.predict(X_test)
        print(f"\n✅ TabNet {task_type.capitalize()} Report")
        if task_type == 'regression':
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, preds)
            print(f"MAE: {mae:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R2 Score: {r2:.4f}")
        else:
            print(classification_report(y_test, preds, target_names=label_encoder.classes_.astype(str)))

        return model

# FTTransformer Trainer
try:
    from rtdl import FTTransformer
    import torch_optimizer

    class FTTransformerTrainer:
        def __init__(self, df, target_column=None, epochs=None):
            self.df = df
            self.target_column = target_column if target_column else df.columns[-1]
            self.epochs = epochs if epochs else 30

        def train(self):
            (X_train, X_test, y_train, y_test), label_encoder, task_type = prepare_tensor_data(self.df, self.target_column)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if task_type == 'regression' else torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if task_type == 'regression' else torch.long)

            train_dataset = TensorDataset(torch.tensor(X_train), y_train_tensor)
            test_dataset = TensorDataset(torch.tensor(X_test), y_test_tensor)

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=256)

            model = FTTransformer(
                d_numerical=X_train.shape[1],
                d_out=1 if task_type == 'regression' else len(np.unique(y_train)),
                n_layers=3,
                d_token=192,
                n_heads=8
            ).to(device)

            optimizer = torch_optimizer.AdamP(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss() if task_type == 'regression' else nn.CrossEntropyLoss()

            print(f"📦 Training FT-Transformer ({task_type}) for {self.epochs} epochs...")
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs).squeeze()
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss:.4f}")

            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    output = model(inputs).squeeze().cpu().numpy()
                    all_preds.extend(output)
                    all_labels.extend(labels.numpy())

            print(f"\n✅ FT-Transformer {task_type.capitalize()} Report")
            if task_type == 'regression':
                mae = mean_absolute_error(all_labels, all_preds)
                mse = mean_squared_error(all_labels, all_preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(all_labels, all_preds)
                print(f"MAE: {mae:.4f}")
                print(f"MSE: {mse:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"R2 Score: {r2:.4f}")
            else:
                print(classification_report(all_labels, np.array(all_preds).argmax(axis=1), target_names=label_encoder.classes_.astype(str)))
            return model

except ImportError:
    print("⚠️ FT-Transformer skipped: `rtdl` not installed.")
