import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load and clean
df = pd.read_csv('email.csv')

df.dropna(subset=['text', 'Category'], inplace=True)
df = df[df['text'].str.strip().astype(bool)]

df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['Category'], inplace=True)
df['Category'] = df['Category'].astype(int)

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

df['text'] = df['text'].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Category'], test_size=0.2, random_state=42)

# Vectorize
vect = TfidfVectorizer(stop_words='english')
X_train_vec = vect.fit_transform(X_train)

# Train
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(vect, 'vectorizer.pkl')

print("Model and vectorizer saved.")