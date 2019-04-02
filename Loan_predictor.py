# Loan Predictor

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train_loan_prediction.csv')
predicting_data = pd.read_csv('test_loan_prediction.csv')

def datapreprocessing(dataset,variables):
    # Taking care of missing data
    dataset.apply(lambda x: sum(x.isnull()),axis=0)
    dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(), inplace=True)

    dataset['Self_Employed'].fillna('No',inplace=True)
    dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
    dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
    dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
    dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace=True)
    dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace=True)

    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    for i in variables:
        dataset[i] = labelencoder.fit_transform(dataset[i])
    dataset.dtypes
    return dataset
    
variables = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
dataset = datapreprocessing(dataset,variables)
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

#Splitting training and testing data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Fitting the Regression Model to the dataset
# Create your Logistic Regression here
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fit the model:
regressor.fit(X_train,y_train)

#Make predictions on test set:
predictions = regressor.predict(X_test)

#Printing accuracy
from sklearn import metrics
accuracy = metrics.accuracy_score(predictions,y_test)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

variables_new = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
predicting = datapreprocessing(predicting_data,variables_new)
predicting = predicting.iloc[:, 1:12].values
predictions = []
for i in regressor.predict(predicting):
    if(i==1):
        predictions.append('Y')
    else:
        predictions.append('N')