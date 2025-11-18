import subprocess
import sys
import os
from PIL import Image, ImageEnhance, ImageFilter

# 🔹 Auto-install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install & import required libraries
try:
    from googletrans import Translator
except ImportError:
    install("googletrans==4.0.0-rc1")
    from googletrans import Translator

try:
    import pytesseract
except ImportError:
    install("pytesseract")
    import pytesseract

try:
    import pyttsx3
except ImportError:
    install("pyttsx3")
    import pyttsx3

# Optional: set tesseract path manually on Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ========== STEP 0: Load local image ==========
IMAGE_PATH = r'C:\surekha\Lab Manual\R23\Tesser.jpeg'

# ========== STEP 1: Preprocess Image ==========
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')  # Grayscale
    img = img.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    preprocessed_path = "preprocessed_image.png"
    img.save(preprocessed_path)
    return preprocessed_path

# ========== STEP 2: OCR - Extract text ==========
def extract_text(image_path):
    print("\n🔍 Extracting text from image...")
    text = pytesseract.image_to_string(Image.open(image_path))
    print("\n📝 Extracted Text:\n", text)
    return text.strip()

# ========== STEP 3: Detect language ==========
def detect_language(text):
    print("\n🧐 Detecting language...")
    translator = Translator()
    detected = translator.detect(text)
    confidence = detected.confidence if detected.confidence is not None else 1.0
    print(f"\n✅ Detected Language: {detected.lang} (confidence: {confidence:.2f})")
    return detected.lang

# ========== STEP 4: Translate into multiple languages ==========
def translate_text_multi(text, target_languages=['en', 'es', 'fr', 'de']):
    translator = Translator()
    translations = {}
    print("\n🌍 Translating text into multiple languages...")
    for lang in target_languages:
        translated = translator.translate(text, dest=lang)
        translations[lang] = translated.text
        print(f"\n✅ {lang} translation:\n{translated.text}")
    return translations

# ========== STEP 5: Simple Text Summarization ==========
def summarize_text(text, num_sentences=2):
    sentences = text.split('.')
    summary = '. '.join(sentences[:num_sentences]).strip()
    if not summary.endswith('.'):
        summary += '.'
    return summary

# ========== STEP 6: Text-to-Speech ==========
def speak_text(text, lang='en'):
    print("\n🔊 Reading summary aloud...")
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ========== STEP 7: Save Summary ==========
def save_summary(summary, lang_code):
    filename = rf"C:\surekha\Lab Manual\R23\summary_{lang_code}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"💾 Summary for '{lang_code}' saved to '{filename}'")

# ========== MAIN ==========
if __name__ == "__main__":
    print("🚀 OCR + Multi-Language Translation + Summarization + TTS + Save Summary Starting...\n")

    try:
        if not os.path.exists(IMAGE_PATH):
            raise FileNotFoundError(f"❌ Image file not found at path: {IMAGE_PATH}")

        # Preprocess image
        preprocessed_path = preprocess_image(IMAGE_PATH)

        # OCR
        extracted_text = extract_text(preprocessed_path)

        if extracted_text:
            # Detect original language
            detected_lang = detect_language(extracted_text)

            # Define target languages (ISO codes)
            target_languages = ['en', 'es', 'fr', 'de']  # English, Spanish, French, German

            # Translate into multiple languages
            translations = translate_text_multi(extracted_text, target_languages)

            # Summarize, speak, and save for each language
            for lang, translated_text in translations.items():
                print(f"\n✂️ Summarizing text in {lang}...")
                summary = summarize_text(translated_text)
                print(f"\n🧠 Summary ({lang}):\n{summary}")
                speak_text(summary, lang)
                save_summary(summary, lang)

        else:
            print("⚠️ No text detected in the image.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
