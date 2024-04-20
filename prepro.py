#Importing modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#Importing the data.
trainset = pd.read_csv("/kaggle/input/recipe-for-rating-predict-food-ratings-using-ml/train.csv")
testset = pd.read_csv("/kaggle/input/recipe-for-rating-predict-food-ratings-using-ml/test.csv")

#Exploratory Data Analysis
print("Shape of training data: ",trainset.shape)
print("Datatypes of each feature: ",trainset.dtypes)
print(trainset.describe())

train_cols_with_null = []
for col in trainset.columns:
    if trainset[col].isnull().sum()!=0:
        train_cols_with_null.append(col)

test_cols_with_null = []
for col in testset.columns:
    if testset[col].isnull().sum()!=0:
        test_cols_with_null.append(col)
        
# Finding numerical features
numerical_features = trainset.select_dtypes(include=['int', 'float']).columns.tolist()
print(numerical_features)

#Correlation Matrix
corr_matrix = trainset[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
        
#Cleaning the data by filling missing values.
for col in train_cols_with_null:
    trainset[col].fillna(" ",inplace=True)
for col in test_cols_with_null:
    testset[col].fillna(" ",inplace=True)

#Vectorizing the text.
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_text_tfidf = tfidf_vectorizer.fit_transform(trainset['Recipe_Review'])
X_test_text_tfidf = tfidf_vectorizer.transform(testset['Recipe_Review'])

# One-hot encoding the categorical features.
categorical_columns = ['RecipeNumber', 'RecipeCode']
onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = onehot_encoder.fit_transform(
    trainset[categorical_columns])
X_test_cat_encoded = onehot_encoder.transform(testset[categorical_columns])

#Dropping features.
columns_to_drop = ['RecipeName', 'CommentID', 'UserID',
                   'UserName','Recipe_Review', 'RecipeNumber', 'RecipeCode']
trainset = trainset.drop(columns=columns_to_drop)
testset = testset.drop(columns=columns_to_drop)

#Splitting into Features and target.
X_train = trainset.drop('Rating', axis=1)
y_train = trainset['Rating']
X_test = testset

#Concatenating the vectorized text with the dataset.
X_train_text_tfidf_df = pd.DataFrame(X_train_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_text_tfidf_df = pd.DataFrame(X_test_text_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
tf_columns_to_drop = ["10", "12", "13", "15", "16", "18", "20", "25",
                      "2nd", "30", "34", "35", "350", "39", "40", "45", "50", "9x13", "14"]
X_train_text_tfidf_df.drop(columns=tf_columns_to_drop, inplace=True)
X_test_text_tfidf_df.drop(columns=tf_columns_to_drop, inplace=True)
X_train = pd.concat([X_train_text_tfidf_df, X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_text_tfidf_df, X_test.reset_index(drop=True)], axis=1)

# Combining the one-hot encoded features with the rest of the dataset.
X_train = pd.concat([pd.DataFrame(X_train_cat_encoded, columns=onehot_encoder.get_feature_names_out(
)), X_train.reset_index(drop=True)], axis=1)
X_test = pd.concat([pd.DataFrame(X_test_cat_encoded, columns=onehot_encoder.get_feature_names_out(
)), X_test.reset_index(drop=True)], axis=1)

oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)