import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import json, os, time

# === إعدادات ===
model_path = "models/custom_image_classifier.h5"  # ← النموذج الحالي لديك
labels_path = "models/class_names.json"           # ← أسماء الفئات موجودة
history_path = "models/unfreeze_history.json"     # ← حفظ سجل التدريب الجديد
dataset_dir = "dataset"                           # ← مجلد الصور المصنّفة


img_height, img_width = 224, 224
batch_size = 32
epochs = 5  # تدريب خفيف فقط على الطبقات العميقة

# === تحميل النموذج المدرب سابقًا ===
model = load_model(model_path)

# === فتح طبقات base_model للتدريب ===
base_model = model.layers[1]  # الطبقة الثانية هي MobileNetV2
base_model.trainable = True  # فتح كل الطبقات (أو جزئية لاحقًا)

# ✅ يمكنك تحديد عدد الطبقات المفتوحة فقط مثلاً:
# for layer in base_model.layers[:-20]:
#     layer.trainable = False

# === إعادة تجميع النموذج بمعدل تعلم صغير ===
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === تحميل أسماء الفئات ===
with open(labels_path, "r") as f:
    class_names = json.load(f)
num_classes = len(class_names)

# === تجهيز البيانات مجددًا ===
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# === التدريب ===
print("🔁 Starting fine-tuning on base model layers...")
start = time.time()
history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=epochs)
end = time.time()

# === حفظ النموذج والنتائج ===
model.save(model_path)
with open(history_path, "w") as f:
    json.dump(history.history, f)

print(f"✅ Fine-tuned model saved to: {model_path}")
print(f"🕒 Total fine-tuning time: {end - start:.2f} seconds")

# === رسم الدقة الجديدة ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy (fine-tuned)")
plt.plot(history.history['val_accuracy'], label="Val Accuracy (fine-tuned)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fine-Tuning MobileNetV2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
