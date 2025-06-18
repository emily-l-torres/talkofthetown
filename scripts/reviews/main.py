from data_sampling import BalancedSampler
from text_preprocessing import ReviewTextPreprocessor

# python scripts/reviews/main.py <- run me!

def main():
    # Configuration
    input_file = "data/raw/yelp_academic_dataset_review.csv"
    target_cities = [
    'Las Vegas', 'Phoenix', 'Charlotte', 'Scottsdale', 'Pittsburgh',
    'Montreal', 'Mesa', 'Henderson', 'Tempe', 'Chandler', 'Cleveland'
    ]
    sample_size = 50000
    min_review_length = 20
    date_range = ('2005-01-01', '2019-12-31')
    
    # Step 1: Create balanced sample
    print("Creating balanced sample...")
    sampler = BalancedSampler(
        target_cities=target_cities,
        sample_size=sample_size,
        min_review_length=min_review_length,
        date_range=date_range
    )
    df_sample = sampler.create_balanced_sample(input_file)
    
    # Step 2: Preprocess text
    print("Preprocessing text...")
    preprocessor = ReviewTextPreprocessor()
    df_processed = preprocessor.preprocess_text(df_sample)
    
    # Step 3: Save results
    print("Saving results...")
    df_processed.to_csv('data/processed/yelp_academic_dataset_processed_reviews.csv', index=False)
    
    print(f"Done! Processed {len(df_processed)} reviews")
    print(f"Star distribution: {df_processed['stars'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()