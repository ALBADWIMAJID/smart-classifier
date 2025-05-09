import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# === إعداد المسارات ===
dataset_dir = "dataset"
csv_path = os.path.join(dataset_dir, "dataset_summary.csv")
plot_path = os.path.join(dataset_dir, "dataset_bar_plot.png")
pdf_path = os.path.join(dataset_dir, "summary_report.pdf")

summary = []

# === التحقق من صلاحية الصورة ===
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

# === تحليل كل فئة في مجلد dataset ===
for category in sorted(os.listdir(dataset_dir)):
    cat_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(cat_path):
        continue

    total = len(os.listdir(cat_path))
    valid = 0
    for fname in os.listdir(cat_path):
        fpath = os.path.join(cat_path, fname)
        if is_valid_image(fpath):
            valid += 1

    summary.append({
        "Category": category,
        "Valid Images": valid,
        "Total Files": total,
        "Corrupted": total - valid
    })

# === إنشاء DataFrame وحفظ CSV ===
df = pd.DataFrame(summary).sort_values(by="Valid Images", ascending=False)
df.to_csv(csv_path, index=False)
print(f"📄 Summary saved to: {csv_path}")

# === رسم بياني ===
plt.figure(figsize=(12, 6))
plt.bar(df["Category"], df["Valid Images"], color="skyblue")
plt.xticks(rotation=90)
plt.title("Valid Images per Category")
plt.xlabel("Category")
plt.ylabel("Valid Images")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.savefig(plot_path)
plt.close()
print(f"📊 Plot saved to: {plot_path}")

# === إنشاء تقرير PDF ===
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "📊 Dataset Summary Report", ln=True, align="C")

pdf.set_font("Arial", "", 12)
pdf.ln(10)
pdf.cell(0, 10, "This report shows the number of valid and corrupted images in each category.", ln=True)

pdf.ln(5)
pdf.image(plot_path, x=10, w=190)

pdf.ln(10)
pdf.set_font("Arial", "B", 12)
pdf.cell(60, 10, "Category", 1)
pdf.cell(40, 10, "Valid", 1)
pdf.cell(40, 10, "Total", 1)
pdf.cell(40, 10, "Corrupted", 1)
pdf.ln()

pdf.set_font("Arial", "", 11)
for _, row in df.iterrows():
    pdf.cell(60, 10, row["Category"], 1)
    pdf.cell(40, 10, str(row["Valid Images"]), 1)
    pdf.cell(40, 10, str(row["Total Files"]), 1)
    pdf.cell(40, 10, str(row["Corrupted"]), 1)
    pdf.ln()

pdf.output(pdf_path)
print(f"✅ PDF report saved to: {pdf_path}")
