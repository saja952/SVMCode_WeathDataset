# SVMCode_WeathDataset
process the data by cleaning ,remove outliers fill missing values and save the clean data in anther file  then train a machine learning model (svm )

the purpose of the code :
process the data by cleaning ,remove outliers fill missing values and save the clean data in anther file 
then train a machine learning model (svm )

the  steps :
1) Data Collect
2) preprocess the data
3) Split the data into two equal parts
4)Train the model
5) Evaluate the model

libraries required :
import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, 
    mean_squared_error
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

and this is the code with some explain on it :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load the dataset

df = pd.read_csv('/content/north.csv')

# Display basic info about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Fill missing numerical values with the mean
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing categorical values with the mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify no missing values remain
print("\nMissing Values after Cleaning:")
print(df.isnull().sum())

# Define a function to remove outliers using the IQR method
def remove_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Apply the outlier removal to all numerical columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df = remove_outliers(df, col)

# Normalize numerical data using Min-Max Scaling
scaler = MinMaxScaler()
df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

# Encode categorical data
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution plots for numerical columns
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


print("\nFinal Processed Dataset Preview:")
print(df.head())

# Save the cleaned and processed dataset
df.to_csv('processed_dataset.csv', index=False)
print("Processed dataset saved as 'processed_dataset.csv'.")





import pandas as pd

# Load the original .csv file
file_path = '/content/processed_dataset.csv' 
data = pd.read_csv(file_path)

# Split the data into two equal parts
file_1_data = data.iloc[:len(data)//2]  # First half
file_2_data = data.iloc[len(data)//2:]  # Second half

# Save to new .csv files for parallel processing
file_1_path = 'file_1.csv' 
file_2_path = 'file_2.csv'  

file_1_data.to_csv(file_1_path, index=False)
file_2_data.to_csv(file_2_path, index=False)

print(f"Data split into:\n1. {file_1_path}\n2. {file_2_path}")




import pandas as pd
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    classification_report, accuracy_score, 
    precision_score, recall_score, f1_score, 
    mean_squared_error
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv('/content/file_2.csv')

# Define target and features
target_column = 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'  
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset.")

X = df.drop(columns=[target_column])
y = df[target_column]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine Model Type
if len(np.unique(y)) > 10:  # Regression
    model_type = "Regression"
    base_model = SVR(kernel='rbf')

    # Hyperparameter Tuning for Regression
    param_dist = {
        'C': uniform(0.1, 2),
        'epsilon': uniform(0.01, 0.1),
        'gamma': uniform(0.01, 0.1)
    }
else:  # Classification
    model_type = "Classification"
    base_model = SVC(kernel='rbf')

    # Hyperparameter Tuning for Classification
    param_dist = {
        'C': uniform(0.1, 2),
        'gamma': uniform(0.01, 0.1)
    }

# Perform Hyperparameter Tuning
print("\nPerforming Hyperparameter Tuning...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy' if model_type == "Classification" else 'neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)
print("\nBest Parameters:", random_search.best_params_)

# Train the final model with optimal parameters
optimized_model = random_search.best_estimator_
optimized_model.fit(X_train, y_train)

# Evaluate the model
y_pred = optimized_model.predict(X_test)

if model_type == "Classification":
    # Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()
else:
    # Regression Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Predictions vs Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.title("Predictions vs Actual (Regression)")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()
