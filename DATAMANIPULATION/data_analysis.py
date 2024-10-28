from pandas import Series, to_datetime, read_csv, concat, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from numpy import arange

def label_distrbution(df):
    """
    This function visualizes the distribution of phishing vs legitimate emails
    (0 for legitimate, 1 for phishing).
    Plots a bar chart to show the count of phishing vs legitimate emails.
    """
    try:
        df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Phishing vs Legitimate Emails')
        plt.xlabel('Email Type')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()
    except Exception as e:
        print(e)

def sender_domain(df):
    """
    This function analyzes and visualizes the top 12 sender domains for phishing and legitimate emails.
    Extracts the domain from the 'sender' email addresses.
    Creates a grouped bar plot to visualize counts of phishing and legitimate sender domains
    Red for phishing, blue for legitimate
    """
    try:
        df = df.dropna(subset=['sender'])
        df = df[df['sender'].str.strip() != '']
        df['domain'] = df['sender'].apply(lambda x: x.split('@')[-1])

        # Separate DataFrames for phishing and legitimate emails
        phishing_domains = df[df['label'] == 1]['domain'].value_counts().head(12)
        legitimate_domains = df[df['label'] == 0]['domain'].value_counts().head(12)

        # Combine the top domains into a single DataFrame for plotting
        combined_domains = DataFrame({
            'Phishing': phishing_domains,
            'Legitimate': legitimate_domains
        }).fillna(0)  # Fill NaN with 0 for missing values
        
        # Create a bar plot for phishing and legitimate sender domains
        x = arange(len(combined_domains))  # the label locations
        width = 0.35  # the width of the bars
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2, combined_domains['Phishing'], width, label='Phishing', color='red', alpha=0.7)
        bars2 = plt.bar(x + width/2, combined_domains['Legitimate'], width, label='Legitimate', color='blue', alpha=0.7)

        # Adding titles and labels
        plt.title('Top Sender Domains: Phishing vs Legitimate')
        plt.xlabel('Domain')
        plt.ylabel('Count')
        plt.xticks(x, combined_domains.index, rotation=45)
        plt.legend()
        plt.tight_layout()  # Adjust layout to make room for the labels
        plt.show()
    
    except Exception as e:
        print(e)

def feature_frequency(df):
    """
    This function generates word clouds for the bodies of phishing and legitimate emails.
    Concatenates the text of phishing and legitimate emails separately.
    Creates two subplots showing word clouds for the most common words within the emails.
    """
    try:
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        phishing_text = ' '.join(df[df['label'] == 1]['body'])
        legit_text = ' '.join(df[df['label'] == 0]['body'])

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('Phishing Emails Word Cloud')
        plt.imshow(WordCloud().generate(phishing_text), interpolation='bilinear')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Legitimate Emails Word Cloud')
        plt.imshow(WordCloud().generate(legit_text), interpolation='bilinear')
        plt.axis('off')

        plt.show()
    except Exception as e:
        print(e)

def time_count(df):
    """
    This function visualizes the frequency of emails sent by hour of the day for both phishing and legitimate emails.
    Counts the number of emails sent during each hour for both categories.
    Plots a line graph comparing the counts for phishing and legitimate emails over 24 hours.
    Red for phishing, blue for legitimate
    """
    try:
        df['date'] = to_datetime(df['date'], format='%a, %d %b %Y %H:%M:%S %z', errors='coerce', utc=True)
        df = df.dropna(subset=['date'])
        df['time_sent'] = to_datetime(df['date']).dt.hour

        # Count emails sent by hour for phishing and legitimate emails
        phishing_hour_counts = df[df['label'] == 1]['time_sent'].value_counts().sort_index()
        legitimate_hour_counts = df[df['label'] == 0]['time_sent'].value_counts().sort_index()

        # Create a combined line plot
        plt.figure(figsize=(12, 6))
        plt.plot(phishing_hour_counts.index, phishing_hour_counts.values, marker='o', color='red', label='Phishing')
        plt.plot(legitimate_hour_counts.index, legitimate_hour_counts.values, marker='o', color='blue', label='Legitimate')

        # Adding titles and labels
        plt.title('Frequency of Emails Sent by Hour: Phishing vs Legitimate')
        plt.xlabel('Hour of Day')
        plt.ylabel('Count')
        plt.xticks(range(24))
        plt.grid()
        plt.legend()
        plt.tight_layout()  # Adjust layout
        plt.show()
        
    except Exception as e:
        print(e)


def sentiment_analysis(df):
    """This function analyzes the sentiment of the email bodies
    Uses SentimentIntensityAnalyzer to compute a sentiment score for each email body
    The sentiment score (compound) ranges from -1 (very negative) to +1 (very positive)
    In terms of tone, and emotional analysis
    Creates a bar plot to display average sentiment scores for phishing and legitimate emails
    """
    try:
        sia = SentimentIntensityAnalyzer()
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        df['sentiment'] = df['body'].apply(lambda x: sia.polarity_scores(x)['compound'])

        sns.barplot(x='label', y='sentiment', data=df)
        plt.title('Average Sentiment Scores by Email Type')
        plt.ylabel('Average Sentiment Score')
        plt.show()
    except Exception as e:
        print(e)

def analyze_sentiment(text):
    """Analyze sentiment of the given text using TextBlob."""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def visualise_sentiment2(df):
    """This function provides a more detailed visualization of sentiment analysis
    It applies the analyze_sentiment function to each email body, 
    which uses TextBlob to calculate both 
    polarity (how positive (+1) or negative (-1) the text is) and
    subjectivity[0-1] (how subjective or objective the text is)
    **Higher subjectivity means text contains personal opinion rather than factual information.
    """
    try:
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        # Apply the function to the content column
        df[['polarity', 'subjectivity']] = df['body'].apply(lambda x: analyze_sentiment(x)).apply(Series)

        # Visualize the sentiment analysis results
        plt.figure(figsize=(12, 6))

        # Polarity
        plt.subplot(1, 2, 1)
        sns.boxplot(x='label', y='polarity', data=df)
        plt.title('Polarity Scores by Email Type')
        plt.ylabel('Polarity Score')

        # Subjectivity
        plt.subplot(1, 2, 2)
        sns.boxplot(x='label', y='subjectivity', data=df)
        plt.title('Subjectivity Scores by Email Type')
        plt.ylabel('Subjectivity Score')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(e)
        

def read_files(dataset: list):
    """Combines data from different files into one DataFrame"""
    return concat(map(read_csv, dataset ), ignore_index=True)