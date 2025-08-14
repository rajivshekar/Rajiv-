import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# 1. Data Import and Basic Overview
print("1. Data Import and Basic Overview")
file_name = 'C:\\Users\\sheka\\Downloads\\CloudWatch_Traffic_Web_Attack.csv'
df = pd.read_csv(file_name)

print("\nInitial DataFrame Info:")
df.info()
print("\nInitial DataFrame Head:")
print(df.head())

# 2. Data Preprocessing
print("\n\n2. Data Preprocessing")
# Remove duplicate rows
df_unique = df.drop_duplicates().copy()

# Convert time-related columns to datetime format
df_unique['creation_time'] = pd.to_datetime(df_unique['creation_time'])
df_unique['end_time'] = pd.to_datetime(df_unique['end_time'])
df_unique['time'] = pd.to_datetime(df_unique['time'])

# Standardize text data (ensure country codes are uppercase)
df_unique['src_ip_country_code'] = df_unique['src_ip_country_code'].str.upper()

print("\nCleaned DataFrame Info:")
df_unique.info()

# 3. Exploratory Data Analysis (EDA)
print("\n\n3. Exploratory Data Analysis (EDA)")

# Analyze Traffic Patterns Based on bytes_in and bytes_out
plt.figure(figsize=(12, 6))
sns.histplot(df_unique['bytes_in'], bins=50, color='blue', kde=True, label='Bytes In')
sns.histplot(df_unique['bytes_out'], bins=50, color='red', kde=True, label='Bytes Out')
plt.title('Distribution of Bytes In and Bytes Out')
plt.xlabel('Bytes')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('bytes_distribution.png')
plt.show()

# Count of Protocols Used
plt.figure(figsize=(10, 5))
sns.countplot(x='protocol', data=df_unique, palette='viridis')
plt.title('Protocol Count')
plt.xlabel('Protocol')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('protocol_count.png')
plt.show()

# Country-based Interaction Analysis
plt.figure(figsize=(15, 8))
sns.countplot(y='src_ip_country_code', data=df_unique,
              order=df_unique['src_ip_country_code'].value_counts().index)
plt.title('Interaction Count by Source IP Country Code')
plt.xlabel('Count')
plt.ylabel('Country Code')
plt.tight_layout()
plt.savefig('country_interaction_count.png')
plt.show()

# Time-Series Analysis of Bytes In and Bytes Out
plt.figure(figsize=(12, 6))
plt.plot(df_unique['creation_time'], df_unique['bytes_in'], label='Bytes In', marker='o')
plt.plot(df_unique['creation_time'], df_unique['bytes_out'], label='Bytes Out', marker='o')
plt.title('Web Traffic Analysis Over Time')
plt.xlabel('Time')
plt.ylabel('Bytes')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('web_traffic_analysis.png')
plt.show()

# 4. Feature Engineering
print("\n\n4. Feature Engineering")
# Duration of the session in seconds
df_unique['session_duration'] = (df_unique['end_time'] - df_unique['creation_time']).dt.total_seconds()
print("\nDataFrame with 'session_duration' added:")
print(df_unique.head())

# 5. Data Transformation
print("\n\n5. Data Transformation")
# Features for scaling and encoding
numerical_features = ['bytes_in', 'bytes_out', 'session_duration']
categorical_features = ['src_ip_country_code']

# Create preprocessor with StandardScaler and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform the data
X_transformed = preprocessor.fit_transform(df_unique)
feature_names = preprocessor.get_feature_names_out()
transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

print("\nTransformed DataFrame with scaled and encoded features:")
print(transformed_df.head())

# 6. Modeling
print("\n\n6. Modeling: Random Forest Classifier")
df_unique['is_suspicious'] = (df_unique['detection_types'] == 'waf_rule').astype(int)

# Using only numerical features for the model, as per the example in the PDF
X = df_unique[['bytes_in', 'bytes_out', 'session_duration']]
y = df_unique['is_suspicious']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# 7. Evaluation
print("\n\n7. Evaluation")
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {accuracy_rf}")

classification_rf = classification_report(y_test, y_pred_rf)
print("\nRandom Forest Classification Report:\n", classification_rf)

# Isolation Forest Model
print("\n\nModeling: Isolation Forest Anomaly Detection")
# Selecting features for anomaly detection
features_iso = df_unique[['bytes_in', 'bytes_out', 'session_duration']]

# Initialize the Isolation Forest model
model_iso = IsolationForest(contamination=0.05, random_state=42)

# Fit and predict anomalies
df_unique['anomaly'] = model_iso.fit_predict(features_iso)
df_unique['anomaly'] = df_unique['anomaly'].apply(lambda x: 'Suspicious' if x == -1 else 'Normal')

# Display anomalies
suspicious_activities = df_unique[df_unique['anomaly'] == 'Suspicious']
print("\nTop 5 Suspicious Activities detected by Isolation Forest:")
print(suspicious_activities.head())
print("\nDistribution of detected anomalies:")
print(df_unique['anomaly'].value_counts())