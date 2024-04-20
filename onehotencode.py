# Importing the modules.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

# Importing the data.
trainset = pd.read_csv("train.csv")

# Cleaning the data by filling missing values.
trainset['Recipe_Review'].fillna('', inplace=True)

X = trainset.drop('Rating', axis=1)
y = trainset['Rating']

# Splitting the training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# One-hot encoding the categorical features.
categorical_columns = ['RecipeNumber', 'RecipeCode']
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = onehot_encoder.fit_transform(
    X_train[categorical_columns])
X_test_cat_encoded = onehot_encoder.transform(X_test[categorical_columns])

# Dropping the original categorical features.
columns_to_drop = ['RecipeName', 'CommentID', 'UserID', 'RecipeNumber', 'RecipeCode',
                   'UserName', 'CreationTimestamp', 'UserReputation',
                   'ReplyCount', 'ThumbsUpCount', 'ThumbsDownCount', 'BestScore']
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

# Combining the one-hot encoded features with the rest of the dataset.
X_train = pd.concat([pd.DataFrame(X_train_cat_encoded, columns=onehot_encoder.get_feature_names_out(
)), X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([pd.DataFrame(X_test_cat_encoded, columns=onehot_encoder.get_feature_names_out(
)), X_test.reset_index(drop=True)], axis=1)

# Vectorizing the text (you can use a different method if needed).
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_text_tfidf = tfidf_vectorizer.fit_transform(X_train['Recipe_Review'])
X_test_text_tfidf = tfidf_vectorizer.transform(X_test['Recipe_Review'])

# Dropping the original text feature.
X_train = X_train.drop(columns=['Recipe_Review'])
X_test = X_test.drop(columns=['Recipe_Review'])

# Concatenating the vectorized text with the dataset.
X_train_text_tfidf_df = pd.DataFrame(
    X_train_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_text_tfidf_df = pd.DataFrame(
    X_test_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

X_train = pd.concat(
    [X_train_text_tfidf_df, X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat(
    [X_test_text_tfidf_df, X_test.reset_index(drop=True)], axis=1)

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# print(X_train.columns, X_train.columns)
# print(X_test.shape, X_test.shape)

# Training the model.
rf_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# cross validation score
scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("Accuracy for each fold: ", scores)
print("Mean accuracy: ", scores.mean())


# Predicting.
'''y_pred = rf_classifier.predict(X_test)

# Putting the prediction into csv.
submission_df = pd.DataFrame({'Rating': y_pred})
submission_df['ID'] = range(1, len(submission_df) + 1)
submission_df = submission_df[['ID', 'Rating']]

# Creating the submission csv.
submission_file_path = 'testpred.csv'
submission_df.to_csv(submission_file_path, index=False)

print(accuracy_score(y_pred, y_test))

columns_list = X_train.columns
columns_df = pd.DataFrame([columns_list])
columns_df.to_csv('columns_list.csv', index=False)
'''
