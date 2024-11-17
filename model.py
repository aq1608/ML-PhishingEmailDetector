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
from pickle import load

def train_model():
    """
    Phishing Detection:

    The logistic regression model will then use these features to learn patterns during training. 

    For example, certain terms in the subject (e.g., "urgent", "password", etc.) or the body might be common in phishing emails.

    Similarly, the sender field could also have useful information (e.g., suspicious domains or known phishing email addresses).
    The model then uses these learned patterns to predict whether new, unseen emails (from the test set) are phishing or non-phishing.
    """

    df = read_files(read_dirfiles(dir=Path("CLEANDATA"), ext="*.csv"))
    df['combined_text'] = df['sender'] + ' ' + df['subject'] + ' ' + df['body']

    df = df.dropna(subset=['combined_text', 'label'])
    df = df[df['combined_text'].str.strip() != '']

    X_train, X_test, y_train, y_test = train_test_split(df['combined_text'], df['label'], test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    log_reg_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    log_reg_model.fit(X_train_tfidf, y_train)

    y_pred = log_reg_model.predict(X_test_tfidf)
    y_pred_proba = log_reg_model.predict_proba(X_test_tfidf)[:, 1]

    return y_pred, y_test, y_pred_proba

# -- Graphs functions --

def classification_report(y_test, y_pred):
    """
    Shows:
        -   Accuracy Score
        -   Precision Score
        -   Recall Score
        -   F1 Score
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = [f1, recall, precision, accuracy]  
    metric_names = ['F1-Score', 'Recall', 'Precision', 'Accuracy']

    plt.figure(figsize=(10, 6))
    bars = plt.barh(metric_names, metrics, color=['blue', 'orange', 'green', 'red'])
    plt.xlim(0, 1) 
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Scores')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    for bar in bars:
        plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2, 
                f'{bar.get_width():.4f}', ha='center', va='center', color='white', fontsize=12)

    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """
    Confusion Matrix
    
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
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba):
    """
    ROC Curve (Receiver Operating Characteristic Curve)

    What It Shows:
    The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate for various threshold settings.
    The AUC (Area Under the Curve) gives you a single number to represent the performance of the classifier. AUC ranges from 0 to 1.

    Interpretation:
    A perfect classifier would have an AUC of 1.0, indicating the ability to perfectly distinguish phishing from non-phishing.
    Closer to the top-left corner indicates a better model, with a high true positive rate and a low false positive rate.
    AUC near 0.5 suggests the model is performing no better than random guessing.

    Example:
    An AUC of 0.90 indicates strong model performance, while an AUC of 0.60 suggests there’s room for improvement in distinguishing between phishing and non-phishing emails.
    """
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
    """
    Precision-Recall Curve
    
    What It Shows:
    The Precision-Recall curve plots Precision (the proportion of correctly predicted phishing emails out of all predicted phishing emails) against Recall (the proportion of phishing emails correctly identified out of all actual phishing emails).
    
    Interpretation:
    A good model should balance Precision and Recall.
    High precision and low recall indicate that the model is conservative: it detects phishing emails carefully but might miss some actual phishing emails (high FN).
    Low precision and high recall mean the model catches most phishing emails but often misclassifies non-phishing emails as phishing (high FP).
    
    Example:
    A flat precision-recall curve means the model performs consistently across different thresholds, while a steep drop means the model may be sensitive to threshold changes (leading to poor performance on some thresholds).
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, color='green', label='Precision-Recall curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()


# ------------------------------ Testing models ------------------------------

def read_pkl():
    """
    Reads pickled files for model testing
    
    Returns Logistic Regression and TF-IDF Vectorizer values
    """
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        log_reg_model = load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = load(vectorizer_file)
    
    return log_reg_model, tfidf_vectorizer

def check_email(sender, subject, body, log_reg_model, tfidf_vectorizer):
    """
    Test for non-phishing:

    Subject: Your Order Confirmation - Thank You for Shopping with Us!

    From: support@onlinestore.com
    To: shuangxrong08@gmail.com
    Date: November 17, 2024

    Hi Shuang,

    Thank you for your recent purchase from OnlineStore.com! We’ve received your order and are processing it right now. Below are the details of your order:

    Order Number: 123456789
    Order Date: November 17, 2024

    Items Purchased:

    Wireless Headphones - $79.99
    Bluetooth Speaker - $49.99
    Total: $129.98 (including tax and shipping)

    Estimated Delivery Date: November 22, 2024

    If you have any questions or need to make any changes to your order, please don't hesitate to reach out to us. We're happy to assist!

    Best regards,
    The OnlineStore Team
    www.onlinestore.com
    Customer Support: support@onlinestore.com | Phone: (123) 456-7890

    Test for phishing:

    Email: myetherevvalliet@gmail.com
    Subject: Urgent: Update Your Account Information
    Body: Dear Valued Customer,

    We noticed unusual activity in your account. To ensure your security, we need you to verify your account information immediately. Click the link below to secure your account:
    [Update Your Account Now] http://www.myetherevvalliet.com/
    Failure to verify your information within 24 hours will result in account suspension.
    Thank you for your prompt attention to this matter.

    Sincerely,
    Customer Support Team
    MyEther Inc.

    Hash: b1b74ff5c67cfdc16f1cb30db9b885046c4c5d71af575a2d3de4a79918c1ce89

    """
    full_email_text = f"{sender} {subject} {body}"
    email_tfidf = tfidf_vectorizer.transform([full_email_text])
    prediction = log_reg_model.predict(email_tfidf)[0]
    result = "Phishing email" if prediction == 1 else "Non-phishing email"
    
    return result

