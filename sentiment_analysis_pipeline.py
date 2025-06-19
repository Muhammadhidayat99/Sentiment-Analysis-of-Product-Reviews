import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Data Loading & Initial Exploration
df = pd.read_csv('Data/product_reviews_mock_data.csv')
print("Shape of data:", df.shape)
print("Columns:", df.columns)
print(df.head())

# 2. Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['CleanReviewText'] = df['ReviewText'].astype(str).apply(preprocess_text)

# 3. Sentiment Analysis (Using TextBlob Polarity)
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        return 'positive'
    elif score < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['CleanReviewText'].apply(get_sentiment)
df['SentimentScore'] = df['CleanReviewText'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 4. EDA of Sentiments
# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title("Sentiment Distribution")
plt.show()

# WordClouds for positive & negative reviews
for sentiment in ['positive', 'negative']:
    text = " ".join(df[df['Sentiment'] == sentiment]['CleanReviewText'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment.capitalize()} Reviews')
    plt.show()

# Sentiment by Product
sentiment_product = df.groupby(['ProductID', 'Sentiment']).size().unstack().fillna(0)
sentiment_product.plot(kind='bar', stacked=True, figsize=(10,6), colormap='coolwarm')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution by Product')
plt.show()

# Sentiment over Time
df['ReviewDate'] = pd.to_datetime(df['ReviewDate'])
df['YearMonth'] = df['ReviewDate'].dt.to_period('M')
sentiment_time = df.groupby(['YearMonth', 'Sentiment']).size().unstack().fillna(0)
sentiment_time.plot(kind='line', figsize=(12,6))
plt.title('Sentiment Trends Over Time')
plt.ylabel('Number of Reviews')
plt.show()

# 5. Topic Modeling (LDA on Negative Reviews)
neg_reviews = df[df['Sentiment'] == 'negative']
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(neg_reviews['CleanReviewText'])

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

def display_topics(model, feature_names, no_top_words):
    for ix, topic in enumerate(model.components_):
        print("Topic %d:" % (ix+1), " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words-1:-1]]))

print("\nTop words per topic in negative reviews:")
display_topics(lda, vectorizer.get_feature_names_out(), 8)

# 6. Key Findings and Recommendations
def summarize_findings(df):
    total = len(df)
    pos = (df['Sentiment']=='positive').sum()
    neg = (df['Sentiment']=='negative').sum()
    neu = (df['Sentiment']=='neutral').sum()
    print(f"\nSentiment Breakdown: {pos/total:.1%} positive, {neg/total:.1%} negative, {neu/total:.1%} neutral")
    print("\nCommon positive feedback themes:")
    print(df[df['Sentiment']=='positive']['CleanReviewText'].sample(5, random_state=1).to_list())
    print("\nCommon negative pain points:")
    print(df[df['Sentiment']=='negative']['CleanReviewText'].sample(5, random_state=1).to_list())

summarize_findings(df)

print("\nActionable Recommendations:")
print("- Highlight common positive aspects (e.g., 'easy to use', 'excellent quality') in marketing.")
print("- Focus product improvement on major pain points (e.g., 'broke easily', 'customer service', 'missing features').")
print("- Consider addressing the most frequent negative topics identified by LDA in product updates or support documentation.")

# --- END OF PIPELINE ---