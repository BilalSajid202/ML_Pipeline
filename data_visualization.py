import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

class DataVisualizer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame.
        Creates the output directory if it doesn't exist.
        """
        self.df = df
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_distributions(self):
        """
        Plot and save histograms for each numeric column in the DataFrame.
        """
        self.df.hist(bins=20, figsize=(15, 10), edgecolor='black')
        plt.suptitle("Distributions of Numeric Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.output_dir, "distributions.png")
        plt.savefig(path)
        print(f"📊 Saved distribution plot to {path}")
        plt.show()

    def plot_heatmap(self):
        """
        Plot and save a correlation heatmap for numeric features.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap", fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "correlation_matrix.png")
        plt.savefig(path)
        print(f"📊 Saved heatmap to {path}")
        plt.show()

    def plot_boxplots(self):
        """
        Plot and save boxplots for each numeric column.
        """
        numeric_cols = self.df.select_dtypes(include='number').columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot: {col}', fontsize=12)
            plt.tight_layout()
            path = os.path.join(self.output_dir, f"boxplot_{col}.png")
            plt.savefig(path)
            print(f"📊 Saved boxplot for {col} to {path}")
            plt.show()

    def plot_all_graphs(self):
        """
        Generate and save all visualizations sequentially.
        """
        print("🔍 Plotting and saving Distributions...")
        self.plot_distributions()

        print("🔍 Plotting and saving Correlation Heatmap...")
        self.plot_heatmap()

        print("🔍 Plotting and saving Boxplots...")
        self.plot_boxplots()
