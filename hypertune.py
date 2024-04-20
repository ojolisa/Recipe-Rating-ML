import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Importing the data.
trainset = pd.read_csv("train.csv")

# Cleaning the data by filling missing values.
trainset['Recipe_Review'].fillna('', inplace=True)

X = trainset.drop('Rating', axis=1)
y = trainset['Rating']

# Splitting the training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorizing the text.
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_text_tfidf = tfidf_vectorizer.fit_transform(X_train['Recipe_Review'])
X_test_text_tfidf = tfidf_vectorizer.transform(X_test['Recipe_Review'])

# Dropping features.
columns_to_drop = ['RecipeName', 'CommentID', 'UserID',
                   'UserName', 'CreationTimestamp', 'Recipe_Review', 'UserReputation',
                   'ReplyCount', 'ThumbsUpCount', 'ThumbsDownCount', 'BestScore']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# Concatenating the vectorized text with the dataset.
X_train_text_tfidf_df = pd.DataFrame(
    X_train_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_text_tfidf_df = pd.DataFrame(
    X_test_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

X_train = pd.concat(
    [X_train_text_tfidf_df, X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat(
    [X_test_text_tfidf_df, X_test.reset_index(drop=True)], axis=1)

# Hyperparameter tuning using RandomizedSearchCV
rf_classifier = RandomForestClassifier(random_state=42)

# Define the parameter distribution
param_dist = {'n_estimators': range(80,95)}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=rf_classifier, param_distributions=param_dist, n_iter=50, cv=3, scoring='accuracy', random_state=42)

# Fit the model to the data
random_search.fit(X_train, y_train)

# Get the best parameters
best_params_random = random_search.best_params_
print("Best Parameters (Randomized Search): ", best_params_random)

# Evaluate the best model on the test set
best_rf_model_random = random_search.best_estimator_
y_pred = best_rf_model_random.predict(X_test)

# Print accuracy
print("Accuracy on Test Set (Best Model): {:.2f}%".format(accuracy_score(y_pred, y_test) * 100))
