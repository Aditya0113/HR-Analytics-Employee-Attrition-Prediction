# ==============================================================================
# HR Analytics - Predict Employee Attrition Project
# Objective: Use analytics to understand the main causes of employee resignation
# and predict future attrition.
# ==============================================================================

# --- 0. Import Necessary Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap # For model interpretability
import os # For creating directories

# Suppress warnings for cleaner output during execution
import warnings
warnings.filterwarnings('ignore')

print("All required libraries loaded successfully! ✅")

# --- Setup for saving figures ---
output_figures_dir = 'hr_analytics_figures'
if not os.path.exists(output_figures_dir):
    os.makedirs(output_figures_dir)
    print(f"\nCreated output directory for figures: '{output_figures_dir}'")
else:
    print(f"\nOutput directory for figures already exists: '{output_figures_dir}'")

# --- 1. Data Loading and Initial Cleaning ---

# Define the dataset file name
dataset_file = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

# Load the dataset
try:
    df = pd.read_csv(dataset_file)
    print(f"\nDataset '{dataset_file}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: '{dataset_file}' not found.")
    print("Please ensure the dataset file is in the same directory as this script.")
    exit()

# Display initial information about the dataset
print("\n--- Initial Dataset Information ---")
print(df.head())
df.info()
print("\n--- Summary Statistics ---")
print(df.describe())
print("\n--- Missing Values Check ---")
print(df.isnull().sum()) # Check for any missing values

# Identify and drop irrelevant columns
# 'EmployeeCount', 'StandardHours', 'Over18' often contain only one unique value.
# 'EmployeeNumber' is a unique identifier and not predictive.
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df = df.drop(columns=columns_to_drop)
print(f"\nDropped uninformative columns: {columns_to_drop}")
print(f"New dataset shape after dropping columns: {df.shape}")

# Convert 'Attrition' target variable to numerical (Yes=1, No=0)
# This is crucial for machine learning models
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
print("\n'Attrition' column converted to numerical (Yes=1, No=0).")
print(f"Attrition value counts:\n{df['Attrition'].value_counts()}")

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Starting Exploratory Data Analysis (EDA) ---")

# 2.1 Attrition Rate by Department
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Department', hue='Attrition', palette='viridis')
plt.title('Attrition Rate by Department', fontsize=16)
plt.xlabel('Department', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '01_attrition_by_department.png'))
plt.close() # Close the figure to free memory
print(f"EDA Insight: Observe raw attrition numbers across departments. Sales and R&D show higher employee counts and thus more attrition cases in absolute terms. Figure saved to {output_figures_dir}/01_attrition_by_department.png")


# 2.2 Attrition Rate by Monthly Income (Salary Bands)
plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome', palette='coolwarm')
plt.title('Monthly Income vs. Attrition', fontsize=16)
plt.xlabel('Attrition (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Monthly Income', fontsize=12)
plt.xticks([0, 1], ['No Attrition', 'Attrition'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '02_monthly_income_vs_attrition.png'))
plt.close()
print(f"EDA Insight: Employees who left (Attrition=1) tend to have a lower median and interquartile range for Monthly Income compared to those who stayed. Figure saved to {output_figures_dir}/02_monthly_income_vs_attrition.png")


# 2.3 Attrition Rate by Years Since Last Promotion
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='YearsSinceLastPromotion', hue='Attrition', palette='rocket')
plt.title('Attrition Rate by Years Since Last Promotion', fontsize=16)
plt.xlabel('Years Since Last Promotion', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '03_attrition_by_years_since_last_promotion.png'))
plt.close()
print(f"EDA Insight: Attrition is noticeable even for those with 0-2 years since last promotion, indicating that recent promotions aren't a complete guarantee of retention for everyone. Figure saved to {output_figures_dir}/03_attrition_by_years_since_last_promotion.png")


# 2.4 Attrition by Job Role
plt.figure(figsize=(14, 7))
sns.countplot(data=df, y='JobRole', hue='Attrition', palette='crest')
plt.title('Attrition Rate by Job Role', fontsize=16)
plt.xlabel('Number of Employees', fontsize=12)
plt.ylabel('Job Role', fontsize=12)
plt.legend(title='Attrition', labels=['No', 'Yes'])
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '04_attrition_by_job_role.png'))
plt.close()
print(f"EDA Insight: Sales Representatives and Laboratory Technicians appear to have a higher propensity for attrition based on raw counts. Figure saved to {output_figures_dir}/04_attrition_by_job_role.png")


# 2.5 Correlation Heatmap for Numerical Features
plt.figure(figsize=(18, 14))
numerical_df = df.select_dtypes(include=np.number)
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=18)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '05_correlation_heatmap.png'))
plt.close()
print(f"EDA Insight: Observe correlations with 'Attrition'. Features like 'MonthlyIncome', 'JobLevel', 'TotalWorkingYears', and 'YearsAtCompany' show negative correlation, suggesting higher values reduce attrition likelihood. Figure saved to {output_figures_dir}/05_correlation_heatmap.png")
print("--- EDA Completed ---")

# --- 3. Data Preprocessing for Modeling ---
print("\n--- Starting Data Preprocessing ---")

# Separate target variable (y) and features (X)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# Identify categorical and numerical features for encoding and scaling
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"\nIdentified Categorical Features: {list(categorical_features)}")
print(f"Identified Numerical Features: {list(numerical_features)}")

# Apply One-Hot Encoding to categorical features
# This converts categorical text data into a numerical format suitable for ML models
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)

# Combine numerical features with the one-hot encoded categorical features
X_preprocessed = pd.concat([X[numerical_features], X_encoded], axis=1)

print(f"\nShape after One-Hot Encoding: {X_preprocessed.shape}")
print("First 5 rows of preprocessed features (before scaling numericals):")
print(X_preprocessed.head())

# Apply Standard Scaling to numerical features
# Scaling ensures all numerical features contribute equally to the model, preventing
# features with larger values from dominating.
scaler = StandardScaler()
X_preprocessed[numerical_features] = scaler.fit_transform(X_preprocessed[numerical_features])

print("\nNumerical features scaled using StandardScaler.")
print("First 5 rows of fully preprocessed features:")
print(X_preprocessed.head())

# Split the preprocessed data into training and testing sets
# Using an 80/20 split and stratify=y to maintain the proportion of attrition in both sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set shape (X_train, y_train): {X_train.shape}, {y_train.shape}")
print(f"Testing set shape (X_test, y_test): {X_test.shape}, {y_test.shape}")
print("--- Data Preprocessing Completed ---")

# --- 4. Build and Evaluate Classification Model (Logistic Regression) ---
print("\n--- Building and Evaluating Logistic Regression Model ---")

# Initialize the Logistic Regression model
# solver='liblinear' is a good choice for smaller datasets and handles L1/L2 regularization.
# random_state ensures reproducibility of results.
model = LogisticRegression(solver='liblinear', random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Logistic Regression model trained successfully!")

# Make predictions on the test set
y_pred = model.predict(X_test)
# Get predicted probabilities for the positive class (Attrition=1)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# --- Model Accuracy Report ---
print("\n--- Model Performance Metrics ---")

# Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix for better understanding
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attrition', 'Attrition'],
            yticklabels=['No Attrition', 'Attrition'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '06_confusion_matrix.png'))
plt.close()
print(f"Confusion Matrix Interpretation: Figure saved to {output_figures_dir}/06_confusion_matrix.png")
print(f"  - True Negatives (Correctly predicted 'No Attrition'): {conf_matrix[0,0]}")
print(f"  - False Positives (Incorrectly predicted 'Attrition'): {conf_matrix[0,1]} (Type I Error - costly for false alarms)")
print(f"  - False Negatives (Incorrectly predicted 'No Attrition'): {conf_matrix[1,0]} (Type II Error - missing at-risk employees)")
print(f"  - True Positives (Correctly predicted 'Attrition'): {conf_matrix[1,1]}")

# Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition'])
print("\nClassification Report:")
print(class_report)
print("--- Model Evaluation Completed ---")

# --- 5. Explain Model Predictions with SHAP Value Analysis ---
print("\n--- Starting SHAP Value Analysis for Model Interpretability ---")

# Using shap.Explainer for broader compatibility, especially with predict_proba outputs.
# We wrap model.predict_proba to explain the output probability of the positive class (Attrition=1).
explainer = shap.Explainer(model.predict_proba, X_train)

# Calculate SHAP values for the test set.
# This returns a shap.Explanation object.
shap_values_obj = explainer(X_test)

# For multi-output models like predict_proba, shap_values_obj.values
# will be a 3D array: (n_samples, n_features, n_classes).
# We want the SHAP values for the positive class (Attrition=1), which is the last dimension (index 1).
shap_values_for_plot = shap_values_obj.values[:, :, 1]

print("SHAP values calculated successfully using shap.Explainer! 🎉")

# 5.1 SHAP Summary Plot (Global Feature Importance - Bar Plot)
# Shows the average absolute impact of each feature on the model output.
print("\n--- SHAP Summary Plot (Overall Feature Importance - Bar Plot) ---")
plt.figure(figsize=(10, 6)) # Create a new figure for SHAP plot
shap.summary_plot(shap_values_for_plot, X_test, feature_names=X_test.columns, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Average Absolute Impact)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '07_shap_feature_importance_bar.png'))
plt.close()
print(f"SHAP Insight: This bar plot highlights the features with the greatest average impact on the model's prediction magnitude for attrition. Figure saved to {output_figures_dir}/07_shap_feature_importance_bar.png")


# 5.2 SHAP Summary Plot (Global Feature Impact & Direction - Dot Plot)
# Shows how each feature's value impacts the prediction (positive or negative contribution).
print("\n--- SHAP Summary Plot (Impact & Direction - Dot Plot) ---")
plt.figure(figsize=(12, 8)) # Create a new figure for SHAP plot
shap.summary_plot(shap_values_for_plot, X_test, feature_names=X_test.columns, show=False)
plt.title('SHAP Summary Plot (Feature Impact and Direction)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '08_shap_summary_dot_plot.png'))
plt.close()
print(f"SHAP Insight: Red dots indicate higher feature values, blue dots indicate lower. A feature pushing towards positive SHAP values (right) increases attrition likelihood, while pushing towards negative (left) decreases it. Figure saved to {output_figures_dir}/08_shap_summary_dot_plot.png")


# 5.3 SHAP Dependence Plot (How a single feature affects the prediction)
# Pick a few key features identified from the summary plots.
# Let's use 'MonthlyIncome' and 'JobSatisfaction' as examples.

print("\n--- SHAP Dependence Plots for Key Features ---")

# Dependence Plot for MonthlyIncome (interacting with JobLevel for context)
plt.figure(figsize=(10, 6)) # Create a new figure for SHAP plot
# For dependence_plot, you can pass the full shap_values_for_plot
shap.dependence_plot("MonthlyIncome", shap_values_for_plot, X_test, interaction_index="JobLevel", show=False)
plt.title('SHAP Dependence Plot: Monthly Income vs. Attrition', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '09_shap_dependence_monthly_income.png'))
plt.close()
print(f"SHAP Insight: Higher monthly income generally correlates with negative SHAP values (reducing attrition likelihood). The interaction with 'JobLevel' suggests that this effect might vary depending on the employee's job level. Figure saved to {output_figures_dir}/09_shap_dependence_monthly_income.png")


# Dependence Plot for JobSatisfaction (interacting with EnvironmentSatisfaction)
plt.figure(figsize=(10, 6)) # Create a new figure for SHAP plot
shap.dependence_plot("JobSatisfaction", shap_values_for_plot, X_test, interaction_index="EnvironmentSatisfaction", show=False)
plt.title('SHAP Dependence Plot: Job Satisfaction vs. Attrition', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, '10_shap_dependence_job_satisfaction.png'))
plt.close()
print(f"SHAP Insight: Lower 'JobSatisfaction' (values 1-2) strongly correlates with positive SHAP values (increasing attrition likelihood). The interaction with 'EnvironmentSatisfaction' indicates combined effects of workplace contentment. Figure saved to {output_figures_dir}/10_shap_dependence_job_satisfaction.png")

# Note: shap.initjs() is typically for interactive plots in Jupyter notebooks.
# For a standalone script, the static plots will be generated.
print("--- SHAP Analysis Completed ---")
print(f"\nAll generated figures have been saved to the '{output_figures_dir}' directory. 🎉")


# --- Python Code to Get Average Absolute SHAP Value ---

# Assuming you have already loaded and preprocessed your data
# and trained your model. Here is the code for those steps:

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# import numpy as np

# # Example data (replace with your preprocessed data)
# X_preprocessed = ...
# y = ...

# X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
# model = LogisticRegression(solver='liblinear', random_state=42)
# model.fit(X_train, y_train)


# --- SHAP Value Calculation ---

# Use the shap.Explainer to get the SHAP values for the positive class (Attrition=1)
explainer = shap.Explainer(model.predict_proba, X_train)
shap_values_obj = explainer(X_test)

# Extract SHAP values for the positive class
# shap_values_for_plot is a matrix of (n_samples, n_features)
shap_values_for_plot = shap_values_obj.values[:, :, 1]

# --- Calculate the Average Absolute SHAP Value for each feature ---
feature_importance = np.mean(np.abs(shap_values_for_plot), axis=0)

# --- Create a DataFrame for the Power BI plot ---
# This DataFrame will contain the two parameters you need:
# Feature names (for the Y-axis) and their importance (for the X-axis).
shap_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Average Absolute SHAP Value': feature_importance
})

# Sort the DataFrame to rank features from most to least important
shap_df = shap_df.sort_values(by='Average Absolute SHAP Value', ascending=False)

# Display the top 10 most important features
print("Top 10 Most Important Features for Attrition Prediction:")
print(shap_df.head(10))

# --- Save the DataFrame to a CSV File ---
# This is the file you will import into Power BI
output_csv_file = 'feature_importance_data.csv'
shap_df.to_csv(output_csv_file, index=False)

print(f"\nSuccessfully generated and saved the feature importance data to '{output_csv_file}'.")
print("You can now import this file into Power BI to create your SHAP plot.")