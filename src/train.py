import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from preprocess import preprocess_data

# Load and preprocess data
dataset_dir = "C:/Users/HP/Desktop/baby_cry_detection/data"
X, y = preprocess_data(dataset_dir)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model
os.makedirs("C:/Users/HP/Desktop/baby_cry_detection/models", exist_ok=True)
dump(model, "C:/Users/HP/Desktop/baby_cry_detection/models/baby_cry_model.pkl")
