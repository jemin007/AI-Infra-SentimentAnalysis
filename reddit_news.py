import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

# Initialize environment
load_dotenv()

class RedditSentimentAnalyzer:
    def __init__(self):
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        
        # Azure Blob Storage setup
        self.azure_conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        self.container_name = "reddit-sentiment-data"
        
        # Initialize NLTK
        nltk.download(['vader_lexicon', 'punkt'], quiet=True)
        self.sia = SentimentIntensityAnalyzer()
        
        # Local backup directory
        self.local_backup_dir = "reddit_data_backups"
        os.makedirs(self.local_backup_dir, exist_ok=True)

    def fetch_headlines(self, subreddit='politics', limit=1000):
        """Fetch headlines from Reddit with progress tracking"""
        headlines = set()
        try:
            for submission in self.reddit.subreddit(subreddit).new(limit=limit):
                headlines.add(submission.title)
                print(f"\rHeadlines collected: {len(headlines)}", end='')
            return list(headlines)
        except Exception as e:
            print(f"\nError fetching headlines: {e}")
            return []

    def analyze_sentiment(self, headlines):
        """Perform sentiment analysis on headlines"""
        results = []
        for headline in headlines:
            pol_score = self.sia.polarity_scores(headline)
            results.append({
                'headline': headline,
                'negative': pol_score['neg'],
                'neutral': pol_score['neu'],
                'positive': pol_score['pos'],
                'compound': pol_score['compound'],
                'label': self._categorize_sentiment(pol_score['compound'])
            })
        return pd.DataFrame(results)

    def _categorize_sentiment(self, compound_score):
        """Categorize sentiment based on compound score"""
        if compound_score >= 0.2:
            return "Positive"
        elif compound_score <= -0.2:
            return "Negative"
        return "Neutral"

    def _clean_headline(self, text):
        """Properly clean headlines for CSV"""
        return (
            str(text).replace('"', '""')  # Escape existing quotes
                  .replace('\n', ' ')     # Remove newlines
                  .replace('\r', ' ')     # Remove carriage returns
                  .strip()
        )

    def _generate_filename(self):
        """Generate consistent filename"""
        return "reddit_news.csv"

    def _validate_csv(self, filepath):
        """Validate CSV has exactly 6 columns"""
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.count('","') != 5:  # 6 columns should have 5 separators
                    print(f"Validation failed on line {i}: {line.strip()}")
                    return False
        return True

    def _save_locally(self, df, filename):
        """Save with proper CSV escaping"""
        local_path = os.path.join(self.local_backup_dir, filename)
        
        # Apply cleaning
        df['headline'] = df['headline'].apply(self._clean_headline)
        
        # Write manually with proper CSV formatting
        with open(local_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
            # Write header
            writer.writerow(['headline', 'negative', 'neutral', 'positive', 'compound', 'label'])
            # Write rows
            for _, row in df.iterrows():
                writer.writerow([
                    row['headline'],
                    row['negative'],
                    row['neutral'],
                    row['positive'],
                    row['compound'],
                    row['label']
                ])
        print(f"Local backup saved to: {local_path}")
        return local_path

    def save_to_blob(self, df):
        """Save results to Azure Blob Storage with validation"""
        filename = self._generate_filename()
        
        try:
            # 1. Save locally with strict formatting
            local_path = self._save_locally(df, filename)
            
            # 2. Validate the file
            if not self._validate_csv(local_path):
                raise ValueError("CSV validation failed - incorrect column count")
            
            # 3. Upload to Azure
            blob_service = BlobServiceClient.from_connection_string(self.azure_conn_str)
            container_client = blob_service.get_container_client(self.container_name)
            
            if not container_client.exists():
                container_client.create_container()
            
            with open(local_path, 'rb') as data:
                blob_client = container_client.get_blob_client(filename)
                blob_client.upload_blob(data, overwrite=True)
            
            print(f"Successfully uploaded validated file: {filename}")
            return True
            
        except Exception as e:
            print(f"\nError during save operation: {e}")
            return False

if __name__ == "__main__":
    analyzer = RedditSentimentAnalyzer()
    
    print("Fetching Reddit headlines...")
    headlines = analyzer.fetch_headlines(limit=500)  # Reduced limit for demo
    
    if headlines:
        print("\nAnalyzing sentiment...")
        sentiment_df = analyzer.analyze_sentiment(headlines)
        
        # Ensure proper column order and clean data
        required_columns = ['headline', 'negative', 'neutral', 'positive', 'compound', 'label']
        sentiment_df = sentiment_df[required_columns].dropna()
        
        print("\nSample results:")
        print(sentiment_df.head())
        
        print("\nSaving data...")
        if analyzer.save_to_blob(sentiment_df):
            print("Process completed successfully!")
        else:
            print("Process completed with errors")
    else:
        print("No headlines were fetched")