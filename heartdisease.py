#Step 1: Descriptive Analysis
# Importing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
# Importing dataset
dataset = pd.read_csv('heart_disease.csv')
# Preview data
dataset.head()
# Dataset dimensions - (rows, columns)
dataset.shape
#output: (253681, 22)
# Features data-type
dataset.info()
# Statistical summary
dataset.describe().T
# Count of null values
dataset.isnull().sum()

#Step 2: Data Visualizations
# Outcome countplot
sns.countplot(x = 'HeartDiseaseorAttack',data = dataset)

# Histogram of each feature
import itertools
col = dataset.columns[:22]
plt.subplots(figsize = (20, 15))
length = len(col)
print(length)
for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot(int(length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    dataset[i].hist(bins = 20)
    plt.title(i)
plt.show()

# Pairplot 
sns.pairplot(data = dataset, hue = 'HeartDiseaseorAttack')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot = True)
plt.show()

#Step 3: Data Preprocessing

dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
# Count of NaN
dataset.isnull().sum()
# Replacing NaN with mean values
dataset["HighBP"].fillna(dataset["HighBP"].mean(), inplace = True)
dataset["HighChol"].fillna(dataset["HighChol"].mean(), inplace = True)
dataset["BMI"].fillna(dataset["BMI"].mean(), inplace = True)
dataset["Stroke"].fillna(dataset["Stroke"].mean(), inplace = True)
dataset["Diabetes"].fillna(dataset["Diabetes"].mean(), inplace = True)

# Feature scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset_new)
dataset_scaled = pd.DataFrame(dataset_scaled)

# Selecting features - [HighBP, HighChol, BMI, Sroke, Diabetes, AGE]
X = dataset_scaled.iloc[:, [1, 2, 4, 6, 7,19]].values
Y = dataset_scaled.iloc[:, 0].values
# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )
# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

#Step 4: Data Modelling
# Support Vector Classifier Algorithm
from sklearn.svm import SVC
svc = SVC(kernel = 'linear', random_state = 42)
svc.fit(X_train, Y_train)
# Making predictions on test dataset
Y_pred = svc.predict(X_test)

#Step 5: Model Evaluation
# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy: " + str(accuracy * 100))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm
# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


