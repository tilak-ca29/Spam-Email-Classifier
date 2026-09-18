from flask import Flask, render_template, request

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv(
    r"C:\Users\ADMIN\Spam-Email-Classifier\SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

print("Dataset loaded successfully!")
print(f"Total messages: {len(data)}")


# ==========================================
# CONVERT LABELS
# spam = 1
# ham = 0
# ==========================================

data["label"] = data["label"].map({
    "spam": 1,
    "ham": 0
})


# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.20,
    random_state=42,
    stratify=data["label"]
)


# ==========================================
# TEXT → NUMBERS
# ==========================================

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train_vectors, y_train)


# ==========================================
# MODEL EVALUATION
# ==========================================

y_pred = model.predict(X_test_vectors)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print()
print("========== MODEL PERFORMANCE ==========")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1-Score : {f1 * 100:.2f}%")
print("=======================================")
print()


# ==========================================
# WEBSITE ROUTE
# ==========================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    probability = None

    if request.method == "POST":

        message = request.form.get("message", "").strip()

        if message:

            # Convert message into numerical features
            message_vector = vectorizer.transform([message])

            # Make prediction
            prediction_value = model.predict(message_vector)[0]

            # Get probabilities
            probabilities = model.predict_proba(message_vector)[0]

            # Convert prediction to text
            if prediction_value == 1:
                prediction = "SPAM"
                probability = round(probabilities[1] * 100, 2)
            else:
                prediction = "HAM"
                probability = round(probabilities[0] * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)