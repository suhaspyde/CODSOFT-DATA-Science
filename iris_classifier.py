import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the dataset
df = pd.read_csv('iris.csv')  # Ensure this file is in the same directory

# Step 2: Encode species as numeric labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Step 3: Split the data
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier()
}

# Step 5: Train the models and evaluate
accuracy_results = {}
best_accuracy = 0
best_model = None

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_results[model_name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_predictions = y_pred

# Step 6: Prepare confusion matrix for the best model
conf_matrix = confusion_matrix(y_test, best_predictions)

# Save the results for the Dash app
results = {
    'accuracy_results': accuracy_results,
    'conf_matrix': conf_matrix,
    'best_predictions': best_predictions,
    'classes': le.classes_
}

# Optionally, save the results to a file for use in Dash
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
