## Step 1: Import Libraries
# Import necessary Python libraries for data analysis and machine learning.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Dataset
# Load the dataset from a CSV file into a pandas DataFrame.
file_path = 'Customer-Churn-Prediction.csv'  # Replace with your dataset path
churn_data = pd.read_csv(file_path)

# Step 3: Initial Data Inspection
# Check the structure and preview the dataset.
print(churn_data.info())
print(churn_data.head())

# Step 4: Data Cleaning
# Convert 'TotalCharges' to numeric and handle missing values.
churn_data['TotalCharges'] = pd.to_numeric(churn_data['TotalCharges'], errors='coerce')
churn_data_cleaned = churn_data.dropna()

# Step 5: Encode Categorical Variables
# Convert categorical variables into dummy/indicator variables.
categorical_cols = churn_data_cleaned.select_dtypes(include=['object']).columns
churn_data_encoded = pd.get_dummies(churn_data_cleaned, columns=categorical_cols, drop_first=True)

# Step 6: Split Data into Features (X) and Target (y)
X = churn_data_encoded.drop(columns=['Churn_Yes'])
y = churn_data_encoded['Churn_Yes']  # Target variable (1 = Churn, 0 = No Churn)

# Step 7: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 8: Feature Selection
# Select the top 20 features using the Chi-Squared test.
selector = SelectKBest(score_func=chi2, k=20)
X_train_reduced = selector.fit_transform(X_train, y_train)
X_test_reduced = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]
print("Selected Features:", selected_features.tolist())

# Step 9: Train Models
# Train Logistic Regression and Random Forest models.
log_reg = LogisticRegression(max_iter=500, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

log_reg.fit(X_train_reduced, y_train)
rf_model.fit(X_train_reduced, y_train)

# Step 10: Evaluate Models
# Make predictions and evaluate both models on the test set.
log_reg_predictions = log_reg.predict(X_test_reduced)
rf_predictions = rf_model.predict(X_test_reduced)

log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

print("Logistic Regression Report:\n", classification_report(y_test, log_reg_predictions))
print("Random Forest Report:\n", classification_report(y_test, rf_predictions))

# Step 11: Visualize Key Insights
# Create simple plots to understand important features.
visualization_data = churn_data_encoded[selected_features.tolist() + ['Churn_Yes']]

# Plot tenure vs. churn
plt.figure(figsize=(8, 5))
sns.histplot(data=visualization_data, x='tenure', hue='Churn_Yes', kde=True, palette='Set1', bins=30)
plt.title("Distribution of Tenure by Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

# Plot monthly charges vs. churn
plt.figure(figsize=(8, 5))
sns.histplot(data=visualization_data, x='MonthlyCharges', hue='Churn_Yes', kde=True, palette='Set1', bins=30)
plt.title("Distribution of Monthly Charges by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

# Plot payment method (Electronic Check) vs. churn
plt.figure(figsize=(8, 5))
sns.countplot(data=visualization_data, x='PaymentMethod_Electronic check', hue='Churn_Yes', palette='Set2')
plt.title("Count of Electronic Check Payments by Churn")
plt.xlabel("Electronic Check Payment (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.legend(title="Churn", labels=["No", "Yes"])
plt.tight_layout()
plt.show()
 Customer-Churn-Analysis
