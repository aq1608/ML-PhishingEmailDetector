import pickle

# Load the saved model and vectorizer
with open('logistic_regression_model.pkl', 'rb') as model_file:
    log_reg_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

def check_email(sender, subject, body):
    # Combine sender, subject, and body for classification
    full_email_text = f"{sender} {subject} {body}"
    
    # Transform the input using the loaded vectorizer
    email_tfidf = tfidf_vectorizer.transform([full_email_text])
    
    # Predict whether the email is phishing (1) or non-phishing (0)
    prediction = log_reg_model.predict(email_tfidf)[0]
    
    # Interpret the result
    result = "Phishing email" if prediction == 1 else "Non-phishing email"
    
    return result



"""
Test for non-phishing:

Email: peterwon@gmail.com

Subject: Team Meeting Reminder

Body: Hi Team,

This is a reminder for our weekly team meeting scheduled for Thursday at 3 PM. We will discuss project updates and any challenges you might be facing.

Please be prepared to share your progress.

Best regards,
James

"""

"""
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