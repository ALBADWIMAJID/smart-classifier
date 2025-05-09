from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
from PIL import Image
import os

# قائمة الفئات
categories = [
    "cat", "dog", "lion", "elephant", "rabbit", "giraffe", "tiger", "bear", "cow", "zebra",
    "car", "motorcycle", "bicycle", "truck", "bus", "airplane", "train", "helicopter", "ship", "scooter",
    "laptop", "phone", "chair", "table", "television", "book", "backpack", "umbrella", "bottle", "clock",
    "keyboard", "monitor", "microwave", "fridge", "bed", "sofa", "fan", "lamp", "printer"
]

max_images = 1000  # العدد المطلوب لكل فئة
output_dir = "dataset"

# التحقق من صلاحية الصورة
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

# تنظيف الصور التالفة
def clean_folder(folder):
    removed = 0
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not is_valid_image(fpath):
            os.remove(fpath)
            removed += 1
    return removed

# تنزيل الصور
for category in categories:
    print(f"\n📂 Category: {category}")
    save_path = os.path.join(output_dir, category)
    os.makedirs(save_path, exist_ok=True)

    existing = len(os.listdir(save_path))
    remaining = max_images - existing
    if remaining <= 0:
        print(f"✅ Already has {existing} images.")
        continue

    print(f"🔽 Bing: downloading {remaining} images...")
    bing = BingImageCrawler(storage={"root_dir": save_path})
    bing.crawl(keyword=category, max_num=remaining)

    # تحقق بعد Bing
    existing = len(os.listdir(save_path))
    remaining = max_images - existing
    if remaining > 0:
        print(f"🔁 Switching to Google for remaining {remaining} images...")
        google = GoogleImageCrawler(storage={"root_dir": save_path})
        google.crawl(keyword=category, max_num=remaining)

    # تنظيف الصور التالفة
    removed = clean_folder(save_path)
    total = len(os.listdir(save_path))
    print(f"🧹 Removed {removed} corrupted images. Total valid: {total}")

print("\n✅ All categories processed successfully.")
