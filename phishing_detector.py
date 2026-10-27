import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# Extract basic URL and keyword-related features
class EmailFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        suspicious_words = [
            "urgent", "verify", "password", "prize",
            "winner", "bank details", "suspended",
            "click here", "account blocked", "claim"
        ]

        for email in X:
            text = str(email).lower()

            url_count = len(
                re.findall(r"https?://\S+|www\.\S+", text)
            )

            keyword_count = sum(
                text.count(word) for word in suspicious_words
            )

            exclamation_count = text.count("!")
            length = len(text)

            features.append([
                url_count,
                keyword_count,
                exclamation_count,
                length
            ])

        return pd.DataFrame(
            features,
            columns=[
                "url_count",
                "suspicious_keyword_count",
                "exclamation_count",
                "email_length"
            ]
        ).values


def main():

    # Load the dataset
    try:
        data = pd.read_csv("phishing_emails.csv")
    except FileNotFoundError:
        print("Error: phishing_emails.csv was not found.")
        return

    # Validate the dataset
    if not {"email_text", "label"}.issubset(data.columns):
        print("Dataset must contain email_text and label columns.")
        return

    data = data.dropna(subset=["email_text", "label"])
    data["label"] = data["label"].astype(str).str.strip()
    data["email_text"] = data["email_text"].astype(str)

    if set(data["label"].unique()) != {"Safe", "Phishing"}:
        print("Labels must be exactly Safe and Phishing.")
        return

    if data["label"].value_counts().min() < 2:
        print("At least two examples of each label are required.")
        return

    X = data["email_text"]
    y = data["label"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Combine text analysis with URL/keyword features
    text_features = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    feature_extractor = FeatureUnion([
        ("text", text_features),
        ("email_features", EmailFeatures())
    ])

    model = Pipeline([
        ("features", feature_extractor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model on held-out test data
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n===== PHISHING EMAIL DETECTOR =====")
    print(f"Dataset size: {len(data)} emails")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    # Display and save confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=["Phishing", "Safe"],
        cmap="Blues"
    )

    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved as confusion_matrix.png")
    plt.show()

    # Test a new email
    print("\n===== TEST A NEW EMAIL =====")
    email = input("Paste an email to analyze: ").strip()

    if email:
        prediction = model.predict([email])[0]
        print(f"Prediction: {prediction}")

        if prediction == "Phishing":
            print("Warning: This email may be suspicious.")
        else:
            print(
                "No strong phishing indicators detected. "
                "This does not guarantee the email is safe."
            )


if __name__ == "__main__":
    main()
