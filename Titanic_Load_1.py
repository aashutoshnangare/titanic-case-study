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