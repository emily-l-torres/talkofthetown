# Talk of the Town

Our work in "Talk of the Town" analysis aims to assist cities in boosting their tourism by using Natural Language Processing modeling to understand thousands of city-specific Yelp reviews.

Cities put a lot of time, effort, and money into attracting visitors. Tourism can be an economic engine if harnessed correctly. Consumer feedback via reviews on Google, Yelp, TripAdvisor, and the like is a gold mine of information these cities could mine to identify the things most in need of improvement as well as the things that are already attracting tourists, but this data is not easily parsed or understood. 

By parsing reviews and analyzing the sentiments of the feedback, our analysis will go far beyond how many stars a site has; it will fully unpack what elements about each site people like or dislike, enabling cities to invest more in what works and discontinue or change what does not. 

More information on the problem we're solving and our approach is available <a href="https://github.com/emily-l-torres/talkofthetown/blob/main/documents/Group%206%20-%20Talk%20of%20the%20Town.pdf" target="_blank">here</a>.

## Research and Selection of Methods

Our project aims to extract reviewer sentiments for specific site features for tourism-heavy sites. Various approaches have been used to apply NLP methods to extract sentiments from Yelp reviews, often focused on restaurant reviews. Researchers have implemented various tactics, including topic modeling with lexicon-based and deep-learning-based approaches and sentiment analysis with BERT, LSTM, and other models. We explore the latest research <a href="https://github.com/emily-l-torres/talkofthetown/blob/main/documents/Literature%Review.pdf" target="_blank">here</a>.

Many of these efforts have graded their models based on precision, recall, accuracy and F1-score. Scalability and generalizability are less important concerns because the goal of extracting information from Yelp reviews is highly domain specific.

We experimented with various modeling approaches to this problem, as can be seen <a href="https://github.com/emily-l-torres/talkofthetown/blob/main/notebooks/preliminary_experimentation.ipynb" target="_blank">here</a>.

## Model Implementation

Our model relies on spaCy, ntlk, and TensorFlow libraries.

Various methods were used to preprocess our data. These steps can be explored in the <a href="https://github.com/emily-l-torres/talkofthetown/tree/main/scripts" target="_blank">scripts folder</a> of this repository. Our finalized CSVs used by our model our available as a zipped folder that can be downloaded <a href="https://github.com/emily-l-torres/talkofthetown/tree/main/data/data-20250703T212152Z-1-001.zip" target="_blank">here</a>.

Our model is implemented <a href="https://github.com/emily-l-torres/talkofthetown/blob/main/notebooks/Talk_of_the_Town.ipynb" target="_blank">here</a>.

## Set Up

All required libraries are listed in the requirements.txt file of this repository.
