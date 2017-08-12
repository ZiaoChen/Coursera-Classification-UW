import sframe

# Read Data
products = sframe.SFrame('data/')


def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation)


# Remove Punctuation
products['review_clean'] = products['review'].apply(remove_punctuation)

# Remove NA
products = products.fillna('review', "")

# Ignore reviews with rating = 3
products = products[products['rating'] != 3]

# Change Rating = 4 or 5 to be positive
products['sentiment'] = products['rating'].apply(lambda rating: +1 if rating > 3 else -1)

# Split into training set and testing set
train_data, test_data = products.random_split(.8, seed=1)

# Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform((train_data['review_clean']))
test_matrix = vectorizer.transform(
    (test_data['review_clean']))  # Test data must be transform in the same way as the training data

from sklearn.linear_model import LogisticRegression

sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])
params = sentiment_model.get_params()
print sum(1 for key in params if params[key] >= 0)

# Making Predictions
sample_test_data = test_data[10:13]
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
scores = sentiment_model.decision_function(sample_test_matrix)
print scores
prob = sentiment_model.predict_proba(sample_test_matrix)
print prob
