import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os, json, time

# إعداد المسارات
dataset_dir = "dataset"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "mobilenet_finetuned.h5")
labels_path = os.path.join(model_dir, "class_names.json")
history_path = os.path.join(model_dir, "fine_tune_history.json")

# إعداد البيانات
img_height, img_width = 224, 224
batch_size = 32
epochs = 5  # عدد دورات fine-tuning

# تحميل أسماء الفئات
with open(labels_path, "r") as f:
    class_names = json.load(f)

# إعداد البيانات مع Augmentation + preprocess_input
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
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

# بناء النموذج مع MobileNetV2
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = True  # 🔓 Unfreeze

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(class_names), activation='softmax')
])

# تجميع النموذج بمعدل تعلم منخفض
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# التدريب
print("🔁 Starting fine-tuning on MobileNetV2...")
start = time.time()
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)
end = time.time()

# حفظ النموذج
model.save(model_path)
print(f"✅ Fine-tuned model saved to {model_path}")
print(f"🕒 Fine-tuning time: {end - start:.2f} seconds")

# حفظ سجل التدريب
with open(history_path, "w") as f:
    json.dump(history.history, f)

# رسم منحنى الدقة
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("📈 Fine-Tuned MobileNetV2 Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
