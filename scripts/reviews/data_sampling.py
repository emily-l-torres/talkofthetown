import pandas as pd
from data_cleaning import ReviewDataCleaner


class BalancedSampler:
    
    def __init__(self, 
                 target_cities: list[str] = None,
                 categories: list[str] = None,
                 sample_size: int = 100000,
                 min_review_length: int = 10,
                 date_range: tuple = ('2005-01-01', '2019-12-31'),
                 chunk_size: int = 10000,
                 progress_interval: int = 50000):
        
        self.target_cities = target_cities
        self.categories = categories
        self.sample_size = sample_size
        self.min_review_length = min_review_length
        self.date_range = date_range
        self.chunk_size = chunk_size
        self.progress_interval = progress_interval
        
        self.samples_per_star = sample_size // 5
        self.total_processed = 0
        
        # Create processor instance
        self.processor = ReviewDataCleaner(
            target_cities=target_cities or [],
            min_review_length=min_review_length,
            samples_per_star=self.samples_per_star,
            date_range=date_range
        )
    
    def _process_chunk(self, df_chunk: pd.DataFrame):
        return self.processor.process_chunk(df_chunk)
    
    def _print_progress(self):
        if self.total_processed % self.progress_interval == 0:
            current_total = sum(len(samples) for samples in self.processor.samples_by_stars.values())
            print(f"Processed {self.total_processed:,} rows, collected {current_total:,} samples")
    
    def _has_enough_samples(self) -> bool:
        return all(len(samples) >= self.samples_per_star for samples in self.processor.samples_by_stars.values())
    
    def _create_final_sample(self) -> pd.DataFrame:
        final_samples = []
        for star_rating, samples in self.processor.samples_by_stars.items():
            selected_samples = samples[:self.samples_per_star]
            final_samples.extend(selected_samples)
            print(f"{star_rating}-star reviews: {len(selected_samples):,}")
        
        df_sample = pd.DataFrame(final_samples)
        print(f"\nFinal sample size: {len(df_sample):,} reviews")
        
        return df_sample
    
    def create_balanced_sample(self, filepath: str) -> pd.DataFrame:
        print("Creating a balanced sample of reviews...")
        
        for chunk_num, df_chunk in enumerate(pd.read_csv(filepath, chunksize=self.chunk_size)):
            self._process_chunk(df_chunk)
            
            self.total_processed += len(df_chunk)
            self._print_progress()
            
            # Check if we have enough samples
            if self._has_enough_samples():
                print("Collected enough samples for all star ratings!")
                break
        
        return self._create_final_sample()