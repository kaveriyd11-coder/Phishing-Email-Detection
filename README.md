# Phishing Email Detection Model

## Project Overview

The Phishing Email Detection Model is a machine learning project developed using Python and Scikit-learn. It analyzes email text and classifies messages as either Phishing or Safe.

## Objectives

- Detect potentially suspicious email messages.
- Analyze email text and identify suspicious keywords.
- Extract URL-related and other email features.
- Train a machine learning classification model.
- Evaluate model performance using accuracy and a confusion matrix.

## Features

- Email text analysis using TF-IDF.
- Suspicious keyword detection.
- URL count extraction.
- Email length and exclamation mark analysis.
- Logistic Regression classification.
- Accuracy score and classification report.
- Confusion matrix visualization.
- Interactive testing of new email messages.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Project Files

- `phishing_emails.csv` — Sample email dataset.
- `phishing_detector.py` — Python source code for the detection model.
- `confusion_matrix.png` — Visualization of model predictions.
- `README.md` — Project documentation.

## Installation

Install the required Python libraries using:

```bash
python -m pip install pandas scikit-learn matplotlib
```

## How to Run

1. Open the project folder in Visual Studio Code.
2. Open the terminal.
3. Run the following command:

```bash
python phishing_detector.py
```

4. Review the accuracy score and classification report.
5. View the confusion matrix.
6. Enter an email message when prompted to test the model.

## Model Evaluation

The model uses a training dataset and a separate test dataset to evaluate its predictions.

The current sample dataset produced approximately 84.62% accuracy on the test set.

**Note:** This project uses a small demonstration dataset. The reported accuracy does not guarantee reliable detection of real-world phishing emails.

## Learning Outcomes

Through this project, I learned about:

- Basic machine learning concepts.
- Text feature extraction using TF-IDF.
- URL and suspicious keyword feature analysis.
- Logistic Regression classification.
- Model training and testing.
- Accuracy measurement and confusion matrix visualization.

## Conclusion

This project demonstrates how machine learning can be used to analyze email text and identify potentially suspicious messages. It provides a basic foundation for further development of phishing email detection systems.
