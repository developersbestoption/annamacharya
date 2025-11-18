# ---------------- Sentiment Analysis ----------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report
except ModuleNotFoundError:
    import os
    os.system("pip install scikit-learn")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score, classification_report

# ---------------- SAMPLE DATASET ----------------
# Example dataset: text + sentiment (1=positive, 0=negative)
texts = [
    "I love this product, it is amazing!",
    "This is the worst experience I have ever had.",
    "Absolutely fantastic, I am very happy!",
    "I hate it, very disappointing.",
    "Best purchase ever, highly recommend it.",
    "Not good, I will never buy this again.",
    "Excellent quality and great service.",
    "Terrible, completely useless.",
]

labels = [1, 0, 1, 0, 1, 0, 1, 0]

# ---------------- VECTORIZE TEXT ----------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ---------------- TRAIN NAIVE BAYES ----------------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------------- PREDICT & EVALUATE ----------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- TEST WITH NEW TEXT ----------------
new_texts = [
    "I am extremely satisfied with this product!",
    "This is a horrible product, very bad experience."
]

new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
  
