import pandas as pd
from data_cleaning import ReviewDataCleaner


class BalancedSampler:
    
    def __init__(self,
                 sample_per_star_per_city: int = 5000,
                 min_review_length: int = 10,
                 date_range: tuple = ('2005-01-01', '2019-12-31'),
                 chunk_size: int = 100000,
                 progress_interval: int = 500000):
        
        self.sample_per_star_per_city = sample_per_star_per_city
        self.min_review_length = min_review_length
        self.date_range = date_range
        self.chunk_size = chunk_size
        self.progress_interval = progress_interval
        self.samples = []
        self.samples_by_city_star = {}
    
    def create_balanced_sample(self, review_filepath: str, business_filepath: str) -> pd.DataFrame:
        print("Loading business data for city lookup...")
        business_df = pd.read_csv(business_filepath)
        business_df = business_df.dropna(subset=['city'])
        business_df['business_id'] = business_df['business_id'].astype(str)
        print(f"Loaded {len(business_df)} businesses with city info.")
        print("Sample business_ids:", business_df['business_id'].head().tolist())

        print("Counting reviews per city...")
        city_counts = {}
        review_id_samples = []
        for chunk in pd.read_csv(review_filepath, chunksize=self.chunk_size):
            chunk['business_id'] = chunk['business_id'].astype(str)
            if len(review_id_samples) < 5:
                review_id_samples.extend(chunk['business_id'].head(5 - len(review_id_samples)).tolist())
            merged = chunk.merge(business_df[['business_id', 'city']], on='business_id', how='left')
            city_col = 'city_y' if 'city_y' in merged.columns else 'city'
            if city_col in merged.columns:
                valid_cities = merged[city_col].dropna()
                for city, count in valid_cities.value_counts().items():
                    city_counts[city] = city_counts.get(city, 0) + count
        print("Sample review business_ids:", review_id_samples)
        top_cities = [city for city, _ in sorted(city_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        print(f"Top 10 cities: {top_cities}")
        if not top_cities:
            print("No cities found after merging review and business data. Please check business_id formats and data integrity.")
            return pd.DataFrame()

        # Prepare sample storage
        for city in top_cities:
            for star in [1,2,3,4,5]:
                self.samples_by_city_star[(city, star)] = []

        print("Sampling reviews for each city and star rating...")
        for chunk in pd.read_csv(review_filepath, chunksize=self.chunk_size):
            chunk['business_id'] = chunk['business_id'].astype(str)
            chunk = chunk.dropna(subset=['text', 'stars'])
            chunk = chunk[chunk['text'].str.len() >= self.min_review_length]
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
            chunk = chunk[(chunk['date'] >= self.date_range[0]) & (chunk['date'] <= self.date_range[1])]
            merged = chunk.merge(business_df[['business_id', 'city', 'name']], on='business_id', how='left')
            city_col = 'city_y' if 'city_y' in merged.columns else 'city'
            name_col = 'name_y' if 'name_y' in merged.columns else 'name'
            if city_col not in merged.columns:
                continue
            merged = merged[merged[city_col].isin(top_cities)]
            for (city, star), group in merged.groupby([city_col, 'stars']):
                key = (city, int(star))
                if key in self.samples_by_city_star:
                    needed = self.sample_per_star_per_city - len(self.samples_by_city_star[key])
                    if needed > 0:
                        required_cols = ['text', 'stars', 'date', 'business_id']
                        if name_col in group.columns:
                            required_cols.append(name_col)
                        group_selected = group[required_cols].head(needed)
                        if name_col == 'name_y':
                            group_selected = group_selected.rename(columns={'name_y': 'name'})
                        self.samples_by_city_star[key].extend(group_selected.to_dict('records'))
            if sum(len(v) for v in self.samples_by_city_star.values()) % self.progress_interval == 0:
                print(f"Collected {sum(len(v) for v in self.samples_by_city_star.values()):,} samples so far...")
            if all(len(v) >= self.sample_per_star_per_city for v in self.samples_by_city_star.values()):
                print("Collected enough samples for all city/star buckets!")
                break

        all_samples = []
        for key, samples in self.samples_by_city_star.items():
            all_samples.extend(samples[:self.sample_per_star_per_city])
            print(f"{key[0]} - {key[1]} star: {len(samples[:self.sample_per_star_per_city])}")
        df_sample = pd.DataFrame(all_samples)
        print(f"\nFinal sample size: {len(df_sample):,} reviews")
        return df_sample