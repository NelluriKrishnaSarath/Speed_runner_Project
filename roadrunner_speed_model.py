import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Read your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('dataset.csv')

# Define the features (independent variables) and target (dependent variable)
X = data[['Distance (km)', 'Terrain', 'Weather']]
y = data['Speed (km/h)']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=['Terrain', 'Weather'], drop_first=True)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model as a pickle (.pkl) file
joblib.dump(model, 'roadrunner_speed_model.pkl')
