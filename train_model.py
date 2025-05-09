import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Load the cleaned dataset
df = pd.read_csv("career_data_cleaned.csv")

# Combine qualification and skills into one input text field
df['input'] = df['qualification'].astype(str) + ' ' + df['skills'].astype(str)

# Define features and target
X = df['input']
y = df['career']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform input data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a classifier
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, "career_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Optional: Evaluate
score = model.score(X_test_vectorized, y_test)
print(f"✅ Model trained successfully. Accuracy: {score:.2f}")
