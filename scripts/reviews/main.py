from data_sampling import BalancedSampler
from text_preprocessing import ReviewTextPreprocessor

# python scripts/reviews/main.py <- run me!

def main():
    # Configuration
    review_file = "data/raw/yelp_academic_dataset_review.csv"
    business_file = "data/raw/yelp_academic_dataset_business.csv"
    sample_per_star_per_city = 5000
    min_review_length = 20
    date_range = ('2005-01-01', '2019-12-31')
    
    # Step 1: Create balanced sample
    print("Creating balanced sample...")
    sampler = BalancedSampler(
        sample_per_star_per_city=sample_per_star_per_city,
        min_review_length=min_review_length,
        date_range=date_range
    )
    df_sample = sampler.create_balanced_sample(review_file, business_file)
    
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