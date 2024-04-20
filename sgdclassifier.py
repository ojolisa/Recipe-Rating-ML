import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Importing the data.
trainset = pd.read_csv("train.csv")

# Cleaning the data by filling missing values.
trainset['Recipe_Review'].fillna('', inplace=True)

X = trainset.drop('Rating', axis=1)
y = trainset['Rating']

# Splitting the training and test datasets.
train_data, test_data, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorizing the text using TfidfVectorizer.
tfidf_vectorizer = TfidfVectorizer(max_features=1000,stop_words = 'english')
X_train_text_tfidf = tfidf_vectorizer.fit_transform(
    train_data['Recipe_Review'])
X_test_text_tfidf = tfidf_vectorizer.transform(test_data['Recipe_Review'])

# Dropping unnecessary columns.
columns_to_drop = ['RecipeName', 'CommentID', 'UserID',
                   'UserName', 'CreationTimestamp', 'Recipe_Review', 'UserReputation',
                   'ReplyCount', 'ThumbsUpCount', 'ThumbsDownCount', 'BestScore']
train_data = train_data.drop(columns=columns_to_drop)
test_data = test_data.drop(columns=columns_to_drop)

# Concatenating the vectorized text with the dataset.
X_train = pd.concat(
    [pd.DataFrame(X_train_text_tfidf.toarray()), train_data.reset_index(drop=True)], axis=1)
X_test = pd.concat(
    [pd.DataFrame(X_test_text_tfidf.toarray()), test_data.reset_index(drop=True)], axis=1)

X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Training the model using SGDClassifier.
sgd_classifier = SGDClassifier(random_state=42)
#sgd_classifier.fit(X_train, y_train)

# cross validation score
scores = cross_val_score(sgd_classifier, X_train, y_train, cv=5)
print("Accuracy for each fold: ", scores)
print("Mean accuracy: ", scores.mean())


# Predicting.
'''y_pred = sgd_classifier.predict(X_test)

# Putting the prediction into csv.
submission_df = pd.DataFrame({'Rating': y_pred})
submission_df['ID'] = range(1, len(submission_df) + 1)
submission_df = submission_df[['ID', 'Rating']]

# Creating the submission csv.
submission_file_path = 'testpred.csv'
submission_df.to_csv(submission_file_path, index=False)

# Calculate and print accuracy score.
accuracy = accuracy_score(y_pred, y_test)
print(f"SGD Accuracy Score: {accuracy}")

rf_classifier = RandomForestClassifier(n_estimators=80,random_state=42,max_depth=200)
rf_classifier.fit(X_train,y_train)

y_pred = rf_classifier.predict(X_test)

# Calculate and print accuracy score.
accuracy = accuracy_score(y_pred, y_test)
print(f"RF Accuracy Score: {accuracy}")
'''