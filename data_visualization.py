import seaborn as sns
import matplotlib.pyplot as plt

def plot_distributions(df):
    """Plots histograms for each numeric column."""
    df.hist(bins=20, figsize=(15, 10), edgecolor='black')
    plt.suptitle("Distributions of Numeric Features", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_heatmap(df):
    """Plots a correlation heatmap for numeric features."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    """Draws boxplots for outlier detection on numeric features."""
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}', fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_all_graphs(df):
    """Wrapper function to run all basic visualizations."""
    print("🔍 Plotting Distributions...")
    plot_distributions(df)
    
    print("🔍 Plotting Correlation Heatmap...")
    plot_heatmap(df)
    
    print("🔍 Plotting Boxplots...")
    plot_boxplots(df)
    

