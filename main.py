import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'GPA': np.random.uniform(0, 4, num_students),
    'Attendance': np.random.randint(0, 100, num_students), # Percentage
    'TestScores': np.random.randint(50, 100, num_students),
    'ExtracurricularActivities': np.random.randint(0, 5, num_students), # Number of activities
    'SocioeconomicStatus': np.random.choice(['Low', 'Medium', 'High'], num_students),
    'AtRisk': np.random.choice([0, 1], num_students, p=[0.7, 0.3]) # 30% at risk
}
df = pd.DataFrame(data)
# One-hot encode categorical feature
df = pd.get_dummies(df, columns=['SocioeconomicStatus'], drop_first=True)
# --- 2. Data Cleaning and Preprocessing (Minimal in this synthetic example) ---
# Check for missing values (unlikely in synthetic data, but good practice)
print("Missing values:\n", df.isnull().sum())
# --- 3. Predictive Modeling ---
# Split data into features (X) and target (y)
X = df.drop('AtRisk', axis=1)
y = df['AtRisk']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Logistic Regression model (simple model for demonstration)
model = LogisticRegression(max_iter=1000) # Increased max_iter to ensure convergence
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# --- 4. Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not At Risk', 'At Risk'], 
            yticklabels=['Not At Risk', 'At Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
output_filename = 'confusion_matrix.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
plt.figure(figsize=(8,6))
sns.countplot(x='AtRisk', data=df)
plt.title('Distribution of At-Risk Students')
plt.xlabel('At Risk (0=No, 1=Yes)')
plt.ylabel('Count')
output_filename = 'at_risk_distribution.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")