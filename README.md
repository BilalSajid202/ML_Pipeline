# 🔍 ML Workflow Pipeline – Interactive & Modular Python Package

This project provides a **professional, extensible Python package** for building complete machine learning workflows—from raw data to trained models—guided via an intuitive **Jupyter Notebook interface**. It supports both classical ML and deep learning models.

---

## 📁 Folder Structure

```
ML_Workflow/
│
├── file_handler.py              # Load/save data from file or MySQL
├── data_understanding.py        # Explore shape, types, nulls, stats
├── preprocessing.py             # Clean, scale, and encode data
├── data_visualization.py        # Plot histograms, heatmaps, etc.
├── data_labeling.py             # Label encoding & target column handler
├── model_training.py            # Classical ML training & evaluation
├── deep_learning_training.py    # Deep learning model training (DNN, TabNet, FTTransformer)
├── model_saver.py               # Save/load trained models (.pkl)
├── ML_Workflow_Interface.ipynb  # 🎯 Guided Jupyter Notebook interface
└── README.md
```

---

## 🎯 Project Goal

To create a **modular, user-driven machine learning pipeline** that:

- Loads data from files or a MySQL database  
- Supports full preprocessing with visualization  
- Offers both classical ML and deep learning model training  
- Auto-detects classification/regression problems  
- Supports dynamic target column labeling and one-hot decoding  
- Saves trained models for deployment  

---

## 🛠️ Features

✅ Load data from CSV, Excel, or MySQL  
✅ Optional MySQL storage via SQLAlchemy + PyMySQL  
✅ Interactive preprocessing (null handling, encoding, scaling)  
✅ Rich visualization: histograms, boxplots, heatmaps, etc.  
✅ Label target with label encoding or one-hot (with auto-recovery)  
✅ Train both classical and deep learning models  
✅ Save models using a class-based `.pkl` saver  
✅ Modular architecture: plug in new models or steps easily  
✅ Global memory tracking of user choices (e.g., target column)  

---

## 🤖 Supported Models

### Supervised Learning

#### 📊 Classical ML
- **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, LightGBM  
- **Regression**: Linear Regression, Decision Tree, Random Forest, SVR

#### 🧠 Deep Learning
- DNNTrainer (Fully Connected Deep Network)  
- TabNetTrainer (Tabular Attention-based Network)  
- FTTransformerTrainer (Requires `rtdl` and `torch`)

> You choose which model to use interactively in the notebook.

---

## 📦 Outputs

- Cleaned, encoded, and scaled DataFrame
- Trained model (classical or deep learning)
- `.pkl` file for deployment (`model_saver.py`)
- Optional logs or metrics printed in notebook
- Optional Gantt chart if task scheduling is involved

---

## 💻 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ML_Workflow.git
cd ML_Workflow
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Includes:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `sqlalchemy`, `pymysql`, `lightgbm`, `torch`, `rtdl`, etc.

### 3. Launch the Notebook Interface

```bash
jupyter notebook ML_Workflow_Interface.ipynb
```

### 4. Follow the Guided Steps

- Step 1: Load or store data (file/DB)  
- Step 2–4: Explore and preprocess  
- Step 5: Label target column (user-specified and stored globally)  
- Step 6: Train ML model (classification/regression, auto-detected or manual)  
- Step 7–9: Train deep learning model (DNN/TabNet/FTTransformer on demand)  
- Step 10: Save model  

---

## 🧪 Example Usage (Notebook Snippets)

```python
# Load & explore
df = load_data()
display_data_summary(df)

# Preprocess
preprocessed_df = preprocess_pipeline(df, scale_method='minmax')

# Label
labeler = DataLabeler(preprocessed_df)
target_column = input("Enter target column: ")
labeled_df = labeler.label_data(target_column)

# Train ML
model = MLModelPipeline(
    df=labeled_df,
    target_column=target_column,
    problem_type="classification",  # or auto
    model_choice="lightgbm"
).run()

# Train Deep Learning (choose one interactively)
trainer = DNNTrainer(labeled_df, target_column, epochs=30)
model = trainer.train()

# Save
ModelSaver("final_model.pkl").save(model)
```

---

## 🧰 Optional: MySQL Database Integration

- Load or store datasets via MySQL
- Use `.env` or prompted credentials
- Secure, SQLAlchemy-backed connection

---

## 🚧 Future Enhancements

- Unsupervised learning (KMeans, PCA, Isolation Forest)
- Streamlit or Gradio interface
- Docker + MLFlow support
- YAML-based config runner for automation

---

## 🤝 Contributing

Pull requests are welcome! Feel free to fork, enhance, or suggest improvements. All contributions will be reviewed with ❤️.

---

## 📜 License

MIT License – free for personal or commercial use.

---


