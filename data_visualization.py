# 4_visualization.py

import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df):
    """Plots histograms for each numeric column"""
    df.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

def plot_heatmap(df):
    """Plots correlation heatmap"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()

def plot_boxplots(df):
    """Draws boxplots for outlier detection"""
    for col in df.select_dtypes(include='number').columns:
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}')
        plt.show()
