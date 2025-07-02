import pandas as pd


class ReviewDataCleaner:
    
    def __init__(self, target_cities: list[str], min_review_length: int, samples_per_star: int, date_range: tuple):
        self.target_cities = target_cities
        self.min_review_length = min_review_length
        self.samples_per_star = samples_per_star
        self.date_range = date_range
        self.samples_by_stars = {star_rating: [] for star_rating in [1, 2, 3, 4, 5]}
    
    def _filter_cities(self, df_chunk: pd.DataFrame):
        if 'city' not in df_chunk.columns or df_chunk['city'].isnull().all():
            return df_chunk
        return df_chunk[df_chunk['city'].isin(self.target_cities)]

    def _filter_date_range(self, df_chunk: pd.DataFrame, 
                          start_date: str, 
                          end_date: str):
        df_chunk['date'] = pd.to_datetime(df_chunk['date'], errors='coerce')
        start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
        df_chunk = df_chunk[(df_chunk['date'] >= start_date) & (df_chunk['date'] <= end_date)]
        return df_chunk

    def _filter_tourism_business(self, categories_str):
        if pd.isna(categories_str) or categories_str is None:
            return False
        
        categories_lower = str(categories_str).lower()
        tourism_categories = [
            'restaurants', 'food', 'hotels', 'travel', 'attractions', 
            'tours', 'museums', 'parks', 'entertainment', 'nightlife',
            'bars', 'coffee', 'shopping', 'arts', 'beaches', 'landmarks'
        ]
        
        return any(cat in categories_lower for cat in tourism_categories)

    def _filter_tourism_businesses(self, df_chunk: pd.DataFrame):
        if 'categories' not in df_chunk.columns or df_chunk['categories'].isnull().all():
            return df_chunk
        return df_chunk[df_chunk['categories'].apply(self._filter_tourism_business)]

    def _filter_columns(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        desired_columns = ['text', 'stars', 'date', 'business_id', 'name']
        
        existing_columns = [col for col in desired_columns if col in df_chunk.columns]
        
        if existing_columns:
            df_chunk = df_chunk[existing_columns]
            essential_columns = ['text', 'stars']
            essential_existing = [col for col in essential_columns if col in df_chunk.columns]
            if essential_existing:
                df_chunk = df_chunk.dropna(subset=essential_existing)
        
        return df_chunk

    def _filter_review_length(self, df_chunk: pd.DataFrame):
        if 'text' not in df_chunk.columns:
            return df_chunk
        return df_chunk[df_chunk['text'].str.len() >= self.min_review_length]

    def _add_samples_by_star_rating(self, df_chunk: pd.DataFrame):
        if 'stars' in df_chunk.columns and len(df_chunk) > 0:
            for star_rating in [1, 2, 3, 4, 5]:
                if len(self.samples_by_stars[star_rating]) < self.samples_per_star:
                    star_reviews = df_chunk[df_chunk['stars'] == star_rating]
                    needed = self.samples_per_star - len(self.samples_by_stars[star_rating])
                    if len(star_reviews) > 0:
                        self.samples_by_stars[star_rating].extend(star_reviews.head(needed).to_dict('records'))

    def process_chunk(self, df_chunk: pd.DataFrame):
        
        df_chunk = self._filter_cities(df_chunk)
        df_chunk = self._filter_date_range(df_chunk, self.date_range[0], self.date_range[1])
        df_chunk = self._filter_tourism_businesses(df_chunk)
        df_chunk = self._filter_columns(df_chunk)
        df_chunk = self._filter_review_length(df_chunk)
        
        self._add_samples_by_star_rating(df_chunk)
        return df_chunk