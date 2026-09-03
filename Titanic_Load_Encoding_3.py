import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix

#-------------------------------------------------------------
#   Function name : DisplayInfo
#   Description   : it displays the Formatted title
#   Parameters    : title(str)
#   Return        : None
#   Date          : 14/03/26
#   Author        : Aashutosh Pravin Nangare
#-------------------------------------------------------------

def Displayinfo(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

#-------------------------------------------------------------
#   Function name : ShowData
#   Description   : It shows the basic information of the Dataset
#   Parameters    : df
#                   df -> Pandas dataframe object
#                   message 
#                   message -> Heading text to display              
#   Return        : None
#   Date          : 14/03/26
#   Author        : Aashutosh Pravin Nangare
#-------------------------------------------------------------

def Showdata(df,message):
    Displayinfo(message)

    print("First 5 Rows of Datset ")
    print(df.head())

    print("\n Shape of dataset")
    print(df.shape)

    print("\n Columns names : ")
    print(df.columns.tolist())

    print("Missing values in each coulmn")
    print(df.isnull().sum())

#-------------------------------------------------------------
#   Function name : CleanTitanicData
#   Description   : It does preprocessing 
#                   It removes unnecessary columns
#                   It handles misssing data
#                   It converts text data to numeric format
#                   It does encoding to categorical columns
#   Parameters    : df
#                   df -> Pandas dataframe
#   Return        : df -> Cleans Panda dataframe
#   Date          : 14/03/26
#   Author        : Aashutosh Pravin Nangare
#-------------------------------------------------------------

def CleanTitanicData(df):
    Displayinfo("Step 2 : Original Data")
    print(df.head())

    #Remove unnecessary columns
    drop_columns = ["Passengerid","zero","Name","Cabin"]
    existing_columns = [col for col in drop_columns if col in df.columns]

    print("Columns to be dropped : ")
    print(existing_columns)

    #drop the unwanted coulmns
    df = df.drop(columns = existing_columns)
    Displayinfo("Step 2 : Data after column removal")
    print(df.head())

    #Handle age column
    if "Age" in df.columns:
        print("Age column before filling missing values")
        print(df["Age"].head(10))

        # coerce -> invalid value gets converted to NaN 
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

        age_median = df["Age"].median()

        # replace missing value  with median
        df["Age"] = df["Age"].fillna(age_median)

        print("\nAge column after preprocessing : ")
        print(df["Age"].head(10))

    #Handles the Fare column
    if "Fare" in df.columns:
        print("\n Fare column before preprocessing")
        print(df["Fare"].head(10))

        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")

        fare_median = df["Fare"].median()
        print("\nMedian of Fare column : ",fare_median)

        # replace missing value  with median
        df["Fare"] = df["Fare"].fillna(fare_median)

        print("\nFare column after preprocessing : ")
        print(df["Fare"].head(10))

    #Handles the Embarked column
    if "Embarked" in df.columns:
        print("\n Embarked column before preprocessing")
        print(df["Embarked"].head(10))

        #convert the data in string
        df["Embarked"] = df["Embarked"].astype(str).str.strip()

        #removes missing values
        df["Embarked"] = df["Embarked"].replace(['nan','None',''],np.nan)

        # Get most frequent value
        embarked_mode = df["Embarked"].mode()[0]        
        print("\nMode of Embarked column : ",embarked_mode)

        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
        print("\n Embarked column after preprocessing")
        print(df["Embarked"].head(10))

     #Handles the Sex column
    if "Sex" in df.columns:
        print("\n Sex column before preprocessing")
        print(df["Sex"].head(10))

        df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

        print("\n Sex column after preprocessing")
        print(df["Sex"].head(10))

    Displayinfo("Data after preprocessing")
    print(df.head())

    print("Missing values after preprocessing : ")
    print(df.isnull().sum())

    #Encode Embark column
    df = pd.get_dummies(df,columns=["Embarked"],drop_first=True)
    print("\nData after encoding")

    print(df.head())
    print("Shape of Dataset : ",df.shape)

    #Convert boolean columns into integer
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    print("\nData after encoding")

    print(df.head())

    return df

#-------------------------------------------------------------
#   Function name : MarvellousTitaniclogistic
#   Description   : This is main pipeline controller
#                   It loads the datset  , shows raw data
#                   It preprocesses the dataset & train the model 
#   Parameters    : Datapath of Dataset of file
#   Return        : None
#   Date          : 14/03/26
#   Author        : Aashutosh Pravin Nangare
#-------------------------------------------------------------

def MarvellousTitaniclogistic(Datapath):
    Displayinfo("Step 1 : Loading the dataset")
    df = pd.read_csv(Datapath)

    Showdata(df,"Initial Dataset")

    df = CleanTitanicData(df)


#-------------------------------------------------------------
#   Function name : main
#   Description   : Starting point of the application
#   Parameters    : None
#   Return        : None
#   Date          : 14/03/26
#   Author        : Aashutosh Pravin Nangare
#-------------------------------------------------------------

def main():
    MarvellousTitaniclogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
    main()