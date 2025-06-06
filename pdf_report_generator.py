from fpdf import FPDF
import pandas as pd
import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, df, target_column, model_name="Trained Model"):
        self.df = df
        self.target_column = target_column
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.section_counter = 1

    def _add_page_number(self):
        self.pdf.set_y(-15)
        self.pdf.set_font("Arial", 'I', 10)
        self.pdf.cell(0, 10, f'Page {self.pdf.page_no()}', align='C')

    def _add_horizontal_line(self):
        self.pdf.set_draw_color(180, 180, 180)
        self.pdf.set_line_width(0.4)
        self.pdf.line(10, self.pdf.get_y() + 2, 200, self.pdf.get_y() + 2)
        self.pdf.ln(5)

    def _add_title_page(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 24)
        self.pdf.cell(0, 20, "ML Workflow Report", ln=True, align='C')
        self.pdf.ln(10)
        self.pdf.set_font("Arial", '', 14)
        self.pdf.cell(0, 10, f"Model: {self.model_name}", ln=True, align='C')
        self.pdf.cell(0, 10, f"Generated on: {self.timestamp}", ln=True, align='C')

    def _add_section_title(self, title):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 18)
        self.pdf.cell(0, 10, f"{self.section_counter}. {title}", ln=True)
        self._add_horizontal_line()
        self.section_counter += 1

    def _add_text_block(self, text, font="Arial", size=12):
        self.pdf.set_font(font, '', size)
        self.pdf.multi_cell(0, 8, text)
        self.pdf.ln(3)

    def _add_dataframe_summary(self):
        self._add_section_title("Dataset Summary")
        self._add_text_block(f"Rows: {self.df.shape[0]}\nColumns: {self.df.shape[1]}")

        self._add_section_title("Data Types")
        dtypes = self.df.dtypes.astype(str)
        dtype_text = "\n".join([f"{col:<25} {dtype}" for col, dtype in dtypes.items()])
        self._add_text_block(dtype_text, font="Courier", size=11)

        self._add_section_title("Missing Values")
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_text = "\n".join([f"{col:<25} {count}" for col, count in missing.items()])
        else:
            missing_text = "No missing values found."
        self._add_text_block(missing_text, font="Courier", size=11)

        self._add_section_title("Statistical Summary")
        stats = self.df.describe().round(2).T
        stats_text = stats.to_string()
        self._add_text_block(stats_text, font="Courier", size=10)

    def _add_image(self, image_path, title):
        if os.path.exists(image_path):
            self._add_section_title(title)
            self.pdf.image(image_path, w=180)
        else:
            self._add_text_block(f"⚠️ Image not found: {image_path}")

    def _add_graphs(self):
        output_folder = "outputs"
        if not os.path.exists(output_folder):
            self._add_text_block("⚠️ Output folder not found.")
            return

        png_files = [f for f in os.listdir(output_folder) if f.lower().endswith(".png")]
        if not png_files:
            self._add_text_block("⚠️ No PNG images found in output folder.")
            return

        for img_file in sorted(png_files):
            base_title = os.path.splitext(img_file)[0].replace('_', ' ').title()
            img_path = os.path.join(output_folder, img_file)
            self._add_image(img_path, base_title)

    def _add_model_evaluation(self):
        self._add_section_title("Model Evaluation Summary")
        eval_file = "outputs/evaluation_summary.txt"
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                lines = f.read().strip().splitlines()

            formatted = []
            for line in lines:
                if ':' in line:
                    key, val = line.split(":", 1)
                    formatted.append(f"• {key.strip()}: {val.strip()}")
                elif line.strip().startswith("["):
                    formatted.append(line)
                else:
                    formatted.append(line.strip())

            # Format confusion matrix nicely if detected
            matrix_lines = [line for line in formatted if line.startswith("[")]
            normal_lines = [line for line in formatted if not line.startswith("[")]

            self._add_text_block("\n".join(normal_lines), font="Courier", size=11)
            if matrix_lines:
                self._add_text_block("Confusion Matrix:", font="Arial", size=12)
                self._add_text_block("\n".join(matrix_lines), font="Courier", size=10)
        else:
            self._add_text_block("⚠️ Evaluation file not found.")


    import os
    if os.path.exists("D:/Tasks/ML_Pipeline/ML_Report.pdf"):
        os.remove("D:/Tasks/ML_Pipeline/ML_Report.pdf")


    def generate_pdf(self, output_path="D:/Tasks/ML_Pipeline/ML-Report.pdf"):
        print("🚀 Running updated ReportGenerator...")
        self._add_title_page()
        self._add_dataframe_summary()
        self._add_graphs()
        self._add_model_evaluation()

        # Page numbers (skip title page)
        for i in range(2, self.pdf.page_no() + 1):
            self.pdf.page = i
            self._add_page_number()

        self.pdf.output(output_path)
        print(f"\n✅ Report saved at: {output_path}")
