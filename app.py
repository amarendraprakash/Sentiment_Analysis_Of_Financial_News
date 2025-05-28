import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask import Flask, render_template, request

# --- NLTK VADER Lexicon Download Check ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("VADER lexicon not found. Attempting to download...")
    nltk.download('vader_lexicon')
    print("VADER lexicon downloaded.")
except LookupError:
    print("VADER lexicon not found. Attempting to download...")
    nltk.download('vader_lexicon')
    print("VADER lexicon downloaded.")


app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis
def analyze_sentiment(news_text):
    if isinstance(news_text, str):
        scores = analyzer.polarity_scores(news_text)
        compound_score = scores['compound']

        # Categorize sentiment based on compound score
        if compound_score >= 0.05:
            sentiment_category = 'Positive'
        elif compound_score <= -0.05:
            sentiment_category = 'Negative'
        else:
            sentiment_category = 'Neutral'
        return sentiment_category, scores
    return 'Neutral', {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

@app.route('/')
def index():
    try:
        # Load the CSV file with 'latin1' encoding to handle potential character issues
        df = pd.read_csv('/kaggle/sentiment-analysis-for-financial-news/all-data.csv', header=None, names=['Original_Sentiment', 'News_Text'], encoding='latin1')

        # Drop rows where 'News_Text' is NaN or empty
        df.dropna(subset=['News_Text'], inplace=True)
        df = df[df['News_Text'].astype(str).str.strip() != '']

        # Apply sentiment analysis
        df[['Predicted_Sentiment', 'VADER_Scores_Dict']] = df['News_Text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

        # Extract individual scores from the dictionary
        df['VADER_Negative'] = df['VADER_Scores_Dict'].apply(lambda x: x['neg'])
        df['VADER_Neutral'] = df['VADER_Scores_Dict'].apply(lambda x: x['neu'])
        df['VADER_Positive'] = df['VADER_Scores_Dict'].apply(lambda x: x['pos'])
        df['VADER_Compound'] = df['VADER_Scores_Dict'].apply(lambda x: x['compound'])

        # Clean up the intermediate dictionary column
        df.drop(columns=['VADER_Scores_Dict'], inplace=True)

        # Get overall sentiment distribution
        sentiment_distribution = df['Predicted_Sentiment'].value_counts().to_dict()

        # Get top 10 positive, negative, and neutral news examples
        positive_news = df[df['Predicted_Sentiment'] == 'Positive'].head(10).copy()
        negative_news = df[df['Predicted_Sentiment'] == 'Negative'].head(10).copy()
        neutral_news = df[df['Predicted_Sentiment'] == 'Neutral'].head(10).copy()

        # Convert to list of dictionaries for Jinja2 template
        positive_news = positive_news.to_dict('records')
        negative_news = negative_news.to_dict('records')
        neutral_news = neutral_news.to_dict('records')


        # Pass data to the HTML template
        return render_template(
            'index.html',
            sentiment_distribution=sentiment_distribution,
            positive_news=positive_news,
            negative_news=negative_news,
            neutral_news=neutral_news,
            total_news=len(df)
        )

    except FileNotFoundError:
        return render_template('index.html', error="all-data.csv not found. Please ensure it's in the same directory as app.py.")
    except Exception as e:
        # Log the full traceback for debugging purposes if debug is True
        if app.debug:
            import traceback
            traceback.print_exc()
        return render_template('index.html', error=f"An error occurred: {str(e)}. Please check your terminal for more details if running in debug mode.")

if __name__ == '__main__':
    app.run(debug=True)
