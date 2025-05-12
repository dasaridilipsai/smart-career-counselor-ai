import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load cleaned data
df = pd.read_csv("career_data_cleaned.csv")

# Combine qualification and skills into one text feature
df["input_text"] = df["qualification"].astype(str) + " " + df["skills"].astype(str)
X = df["input_text"]
y = df["career"]

# Convert text to numerical vectors using TF-IDF with bi-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("✅ Model Evaluation Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "career_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved as career_model.pkl and vectorizer.pkl")
