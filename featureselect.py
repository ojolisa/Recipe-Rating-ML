from sklearn.ensemble import RandomForestClassifier
import pandas as pd

trainset = pd.read_csv("train.csv")
X_train = trainset.drop(columns=['Rating', 'Recipe_Review','RecipeName','UserName','CommentID','UserID',], axis=1)
y_train = trainset['Rating']

# X is your feature matrix, y is the target variable
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get feature importances
feature_importances = model.feature_importances_

# Print or visualize feature importances
print("Feature Importances:")
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")