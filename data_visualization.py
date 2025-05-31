import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame.
        """
        self.df = df

    def plot_distributions(self):
        """
        Plot histograms for each numeric column in the DataFrame.
        Helps visualize the distribution (skewness, spread) of each numeric feature.
        """
        self.df.hist(bins=20, figsize=(15, 10), edgecolor='black')
        plt.suptitle("Distributions of Numeric Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
        plt.show()

    def plot_heatmap(self):
        """
        Plot a correlation heatmap for numeric features.
        Shows pairwise correlation coefficients, useful to identify relationships.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self):
        """
        Draw boxplots for all numeric columns.
        Useful for detecting outliers and spread within each feature.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot: {col}', fontsize=12)
            plt.tight_layout()
            plt.show()

    def plot_all_graphs(self):
        """
        Wrapper to run all visualizations sequentially.
        Prints progress messages for user clarity.
        """
        print("🔍 Plotting Distributions...")
        self.plot_distributions()
        
        print("🔍 Plotting Correlation Heatmap...")
        self.plot_heatmap()
        
        print("🔍 Plotting Boxplots...")
        self.plot_boxplots()
