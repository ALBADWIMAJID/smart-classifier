
# 🧠 Custom Smart Classifier

A deep learning project using **MobileNetV2** and **Streamlit** to classify real-world images into 39 everyday categories (animals, devices, furniture, etc.).

📌 Built and fine-tuned as part of a graduation project by **Аль-Бадви Маджед Башир**, ВолгГТУ, ИВТ-363.

---

## 🚀 Features

- ✅ Fine-tuned MobileNetV2 model for multi-class image classification
- 🖼️ Upload or paste URL of an image via Streamlit UI
- 📈 Visualize training accuracy/loss curves
- 📊 Dataset analysis (CSV + PNG + PDF)
- 📦 Includes full training, fine-tuning & evaluation scripts

---

## 📂 Project Structure


├── app.py # Streamlit web interface

├── analyze_dataset.py # Dataset summary generator

├── fine_tune_mobilenet.py # Main training script using MobileNetV2

├── unfreeze_model.py # Extra fine-tuning with frozen layers

├── download_images_full.py # Downloads images from Bing

├── models/

│ ├── mobilenet_finetuned.h5

│ ├── class_names.json

│ ├── fine_tune_plot.png

│ ├── fine_tune_results.csv

│ └── best_model.h5

├── requirements.txt


└── README.md



---

## 💻 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/ALBADWIMAJID/smart-classifier.git
cd smart-classifier
2. Install dependencies

pip install -r requirements.txt
3. Run the Streamlit app

streamlit run app.py
🌐 Try it online (Streamlit Cloud)

(⚠️ Add your actual link after deployment)

📊 Example Predictions
Example	Prediction
phone (86.07%)
cat (99.92%)
cow (98.72%)

🧪 Dataset
Source: Bing + OpenImages (custom script)

Classes: 39

Images per class: ~300–1000

Total: ~6200 images

📈 Training Results
Model: MobileNetV2 (pre-trained on ImageNet)

Final accuracy: ~87% validation

Loss and accuracy plots saved in models/fine_tune_plot.png

👨‍🎓 Author
Аль-Бадви Маджед Башир
ИВТ-363, ВолгГТУ
2025

📄 License
This project is for educational and demonstration purposes only.

