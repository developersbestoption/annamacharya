import os
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

# Optional: set tesseract path manually on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ====== Paths ======
IMAGE_FOLDER = r"C:\surekha\Lab Manual\R23"  # Folder containing images
OUTPUT_FOLDER = r"C:\surekha\Lab Manual\R23\Summaries"  # Folder to save summaries

# Create output folder if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====== Preprocess image ======
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Grayscale
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    return img

# ====== OCR ======
def extract_text(img):
    return pytesseract.image_to_string(img).strip()

# ====== Simple summarization ======
def summarize_text(text, num_sentences=3):
    sentences = text.split('.')
    summary = '. '.join(sentences[:num_sentences]).strip()
    if summary and not summary.endswith('.'):
        summary += '.'
    return summary

# ====== Process all images in folder ======
def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessing image: {filename}")

            img = preprocess_image(image_path)
            text = extract_text(img)
            if not text:
                print("⚠️ No text detected.")
                continue

            summary = summarize_text(text)
            print("📝 Extracted Text:\n", text)
            print("🧠 Summary:\n", summary)

            summary_file = os.path.join(OUTPUT_FOLDER, f"summary_{os.path.splitext(filename)[0]}.txt")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"💾 Summary saved to: {summary_file}")

# ====== Main ======
if __name__ == "__main__":
    if not os.path.exists(IMAGE_FOLDER):
        print(f"❌ Image folder not found: {IMAGE_FOLDER}")
    else:
        process_images(IMAGE_FOLDER)
      
