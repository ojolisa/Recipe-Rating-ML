# Importing the modules.
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Importing the data.
trainset = pd.read_csv("train.csv")

# Cleaning the data by filling missing values.
trainset['Recipe_Review'].fillna('', inplace=True)

X = trainset.drop('Rating', axis=1)
y = trainset['Rating']

# Splitting the training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.001, random_state=42)

# Vectorizing the text.
v = ["ideal", "good", "bad", "great", "wrong", "wonderful", "throw", "not", "disaster", "dry", "delicious", "loved", "love",
     "loves", "favorite", "simple", "tasty", "quick", "worst", "like", "liked", "nice", "fabulous", "bland", "don&#39;t", "perfect"]
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# tfidf_vectorizer = TfidfVectorizer(max_features=1000)
# tfidf_vectorizer = TfidfVectorizer(vocabulary=v)
X_train_text_tfidf = tfidf_vectorizer.fit_transform(X_train['Recipe_Review'])
X_test_text_tfidf = tfidf_vectorizer.transform(X_test['Recipe_Review'])

# Dropping features.
columns_to_drop = ['RecipeName', 'CommentID', 'UserID',
                   'UserName', 'Recipe_Review']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# print(X_train.columns())

# Concatenating the vectorized text with the dataset.
X_train_text_tfidf_df = pd.DataFrame(
    X_train_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_text_tfidf_df = pd.DataFrame(
    X_test_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tf_columns_to_drop = ["10", "12", "13", "15", "16", "18", "20", "25",
                      "2nd", "30", "34", "35", "350", "39", "40", "45", "50", "9x13", "14"]
X_train_text_tfidf_df.drop(columns=tf_columns_to_drop, inplace=True)
X_test_text_tfidf_df.drop(columns=tf_columns_to_drop, inplace=True)
X_train = pd.concat(
    [X_train_text_tfidf_df, X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat(
    [X_test_text_tfidf_df, X_test.reset_index(drop=True)], axis=1)

# print(X_train.shape, y_train.shape)
# print(X_train.columns,testset.columns)

# Training the model.
knn_classifier = KNeighborsClassifier(
    n_neighbors=39)
# rf_classifier.fit(X_train, y_train)

# cross validation score
scores = cross_val_score(knn_classifier, X_train, y_train, cv=5)
print("Accuracy for each fold: ", scores)
print("Mean accuracy: ", scores.mean())        

# Predicting.
'''y_pred = rf_classifier.predict(X_test)

# Putting the prediction into csv.
submission_df = pd.DataFrame({'Rating': y_pred})
submission_df['ID'] = range(1, len(submission_df) + 1)
submission_df = submission_df[['ID', 'Rating']]

# print(submission_df['Rating'].describe())

# Creating the submission csv.
submission_file_path = 'testpred.csv'
submission_df.to_csv(submission_file_path, index=False)

print(accuracy_score(y_pred, y_test))

columns_list = X_train.columns
columns_df = pd.DataFrame([columns_list])
columns_df.to_csv('columns_list.csv', index=False)
'''
