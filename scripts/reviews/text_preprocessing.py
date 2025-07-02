import pandas as pd
import re

class ReviewTextPreprocessor:
    
    def __init__(self, positive_threshold: int = 4, neutral_threshold: int = 3):
        self.positive_threshold = positive_threshold
        self.neutral_threshold = neutral_threshold
    
    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        
        text = str(text).lower() # converts to lower case
        text = re.sub(r'\s+', ' ', text) # removes white space
        text = re.sub(r'http\S+', '', text) # remove urls
        text = re.sub(r'@\w+', '', text) # remove @ mentions
        return text.strip()

    def preprocess_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        if 'text' not in df_processed.columns:
            return df_processed
        
        df_processed['text_clean'] = df_processed['text'].apply(self._clean_text)
        df_processed['text_length'] = df_processed['text'].str.len()
        df_processed['word_count'] = df_processed['text'].str.split().str.len()
        
        if 'stars' in df_processed.columns:
            df_processed['sentiment_binary'] = df_processed['stars'].apply(
                lambda x: 1 if x >= self.positive_threshold else 0  # 4-5 stars = positive, 1-3 = negative
            )
            df_processed['sentiment_multiclass'] = df_processed['stars'].apply(
                lambda x: 'negative' if x <= 2 else ('neutral' if x == self.neutral_threshold else 'positive')
            )
        
        return df_processed