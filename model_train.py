# Import necessary libraries
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from DATAMANIPULATION.data_analysis import read_files
from main import read_dirfiles

# Step 1: Load the cleaned dataset, selecting only the relevant columns
df = read_files(read_dirfiles(dir=Path("CLEANDATA"), ext="*.csv"))

# Combine 'sender', 'subject', and 'body' columns into a single text column
df['combined_text'] = df['sender'] + ' ' + df['subject'] + ' ' + df['body']

# Remove rows with missing values
df = df.dropna(subset=['combined_text', 'label'])
df = df[df['combined_text'].str.strip() != '']

# Step 2: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['label'], test_size=0.2, random_state=42)

# Step 3: Apply TF-IDF Vectorization to the combined text
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features based on your data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Initialize and train a Logistic Regression model
log_reg_model = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg_model.fit(X_train_tfidf, y_train)

# Step 5: Make predictions on the test set
y_pred = log_reg_model.predict(X_test_tfidf)

# -- Graphing Section Starts Here --
# Define the plotting functions

def classification_report():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = [f1, recall, precision, accuracy]  # Accuracy last to be on top
    metric_names = ['F1-Score', 'Recall', 'Precision', 'Accuracy']

    plt.figure(figsize=(10, 6))
    bars = plt.barh(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.xlim(0, 1)  # Set x-axis limit from 0 to 1
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add value labels to the center of the bars
    for bar in bars:
        plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.4f}', ha='center', va='center', color='white', fontsize=12)

    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='green', label='Precision-Recall curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()

# Generate predictions with probabilities
y_pred_proba = log_reg_model.predict_proba(X_test_tfidf)[:, 1]

"""
Phishing Detection:

The logistic regression model will then use these features to learn patterns during training. 

For example, certain terms in the subject (e.g., "urgent", "password", etc.) or the body might be common in phishing emails.

Similarly, the sender field could also have useful information (e.g., suspicious domains or known phishing email addresses).
The model then uses these learned patterns to predict whether new, unseen emails (from the test set) are phishing or non-phishing.
"""


"""
How to interpret the graphs:

1. Confusion Matrix
What it means: The confusion matrix shows the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). This helps you understand how well the model is distinguishing between phishing and non-phishing emails.
2. ROC Curve (Receiver Operating Characteristic Curve)
What it means: The ROC curve shows the trade-off between the true positive rate (recall) and the false positive rate (1-specificity) for different thresholds. It helps evaluate the model's ability to distinguish between classes across thresholds.
AUC (Area Under the Curve): AUC is a single value that summarizes the performance of the model. A value of 1.0 is perfect, and 0.5 is random guessing.
3. Precision-Recall Curve
What it means: The precision-recall curve shows the trade-off between precision and recall across thresholds. It's particularly useful for imbalanced datasets where the positive class (phishing) is less frequent.

"""

"""
1. Confusion Matrix
What It Shows:
The confusion matrix gives a breakdown of how well your model classified the test data into True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
Interpretation:
True Positives (TP): Emails correctly classified as phishing.

True Negatives (TN): Emails correctly classified as non-phishing.

False Positives (FP): Non-phishing emails incorrectly classified as phishing (type I error).

False Negatives (FN): Phishing emails incorrectly classified as non-phishing (type II error).

A good model will have high values for TP and TN (along the diagonal) and low values for FP and FN.

Example:
If the model shows high FP, that means it's flagging many non-phishing emails as phishing, which could lead to unnecessary actions (e.g., emails being blocked).
If FN is high, phishing emails might be slipping through undetected.


2. ROC Curve (Receiver Operating Characteristic Curve)
What It Shows:
The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate for various threshold settings.
The AUC (Area Under the Curve) gives you a single number to represent the performance of the classifier. AUC ranges from 0 to 1.
Interpretation:
A perfect classifier would have an AUC of 1.0, indicating the ability to perfectly distinguish phishing from non-phishing.
Closer to the top-left corner indicates a better model, with a high true positive rate and a low false positive rate.
AUC near 0.5 suggests the model is performing no better than random guessing.
Example:
An AUC of 0.90 indicates strong model performance, while an AUC of 0.60 suggests there’s room for improvement in distinguishing between phishing and non-phishing emails.


3. Precision-Recall Curve
What It Shows:
The Precision-Recall curve plots Precision (the proportion of correctly predicted phishing emails out of all predicted phishing emails) against Recall (the proportion of phishing emails correctly identified out of all actual phishing emails).
Interpretation:
A good model should balance Precision and Recall.
High precision and low recall indicate that the model is conservative: it detects phishing emails carefully but might miss some actual phishing emails (high FN).
Low precision and high recall mean the model catches most phishing emails but often misclassifies non-phishing emails as phishing (high FP).
Example:
A flat precision-recall curve means the model performs consistently across different thresholds, while a steep drop means the model may be sensitive to threshold changes (leading to poor performance on some thresholds).


Summary:
Confusion Matrix: Understand the raw performance of your model in terms of correctly or incorrectly classified emails.

ROC Curve: Measure the overall effectiveness of your model’s ability to differentiate between phishing and non-phishing.

Precision-Recall Curve: Evaluate your model’s balance between catching all phishing emails (recall) while minimizing false alarms (precision).


By analyzing these graphs, you can evaluate your model's strengths and weaknesses, identify areas for improvement, and better understand how it makes decisions.


"""



