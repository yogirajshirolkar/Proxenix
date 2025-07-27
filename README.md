Movie Review Analysis

This project intend to predict the sentiment for a number of movie reviews using the movie reviews dataset from IMDb along with their associated binary sentiment polarity labels.Analyze the textual documents and predict their sentiment or opinion based on the content of these documents to determine the movie review is positive or negative.

Sentiment analysis on movie reviews
Data
Movie reviews from IMDb(Internet movie database).The dataset obtained from http://ai.stanford.edu/~amaas/data/sentiment/. The data set contains 50,000 movie reviews labeled whether they are positive or negative based on the content. For changing the preprocessed txt data to csv format, used the function as in http://psyyz10.github.io/2017/06/Sentiment/.

Approach
A classification analysis on reviews to predict the sentiment positive or negative.The task is to predict the sentiment of 15,000 labeled movie reviews and use the remaining 35,000 reviews for training the supervised models.The techniques used include text preprocessing, normalization and in-depth analysis of models using python's built in packages and custom modules like text_normalizer and model_evaluation_utils. (Source Credit: Practical Machine Learning with Python: A Problem-Solver's Guide to Building Real-World Intelligent SystemsBook by Dipanjan Sarkar, Raghav Bali, and Tushar Sharma. https://github.com/dipanjanS/practical-machine-learning-with-python/tree/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment)

Text pre-processing and normalization
Normalizing movie review data includes creating functions to remove HTML tags,accented characters,expanding contractions,removing special characters,lemmatization to get the root word and removing stop words.Then using all these components and tie them together in the function called normalize_corpus which can be used to take a document corpus as input and return the same corpus with cleaned and normalized text documents. (Reference: practical-machine-learning-with-python/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text Normalization Demo.ipynb

Steps for supervised sentiment analysis
Prepare train and test datasets
Text pre-processing
Feature engineering
Model training
Model prediction and evaluation
Analysing topic models
The first step in this analysis is to combine the normalized train and test reviews and separate out these reviews in to positive and negative reviews. Second step is extract features from positive and negative reviews using TF-IDF feature vectorizer.

Topic Modeling
For topic modeling we use the NMF class from scikit-learn and pyLDAvis for building interactive visualizations of topic models. Also some utility functions from topic-model-utils module to display the results in a clean format.

Interactive visualization of positive review topics: The visualizations shows 10 models from positive and negative reviews.The visualizations are interactive(if using jupyter notebook)and you can click on any of the bubbles representing topics in the Intertopic Distance Map on the left and see the most relevant terms in each of the topics in the right bar chart. Multi-Dimension Scaling (MDS) is used in the plot on the left shows similar topics should be close to one another and dissimilar topics should be far apart.The size of each topic bubble is based on the frequency of that topic and its components in the overall corpus. The visualization on the right shows the top terms.When no topic is selected it shows the top 15 most salient topics in the corpus.The term's saliency is defined as a measure of how frequently the term appears the corpus and its distinguishing factor when used to distinguish between topics.The relevancy metric can be changed based on the slider on top of the bar chart.

Machine learning models and evaluation
Feature engineering techniques is based on the bag of words and TF-IDF models. For classification we will be using Logistic regression. We also tried the deep learning model deep neural network(DNN) based on word2vec and GloVe models.That involves steps like label encoding and word embeddings in feature engineering.

The logistic regression model got 90% accuracy And F1-score 91% on our BOW models The TF-IDF model got accuracy And F1-score 89% The DNN model accuracy and F1-score on word2vec features is 89% The DNN model accuracy and F1-score on GloVe features is 72%

Summary
For predicting the sentiment from movie reviews we use machine learning approches like Logistic regression and deep neural networks.The Linear regression on BOW features got the highest accuracy and F1-score 91% and DNN model on GloVe features got the lowest accuracy and F1-score 72%.
