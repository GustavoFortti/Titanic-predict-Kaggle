from operator import index
from matplotlib.pyplot import axes, axis
import pandas as pd
import numpy as np

from scipy.sparse.construct import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# self.test = self.clean_cabin(test, ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin'], True)
# self.train = self.clean_cabin(train, ['PassengerId', 'Age', 'Name', 'Cabin'], True)

class build_cabin():
    def __init__(self, test, train, has_clf=False):
        
        pd.options.mode.chained_assignment = None 

        if has_clf == False:
            self.clf = GradientBoostingClassifier()
        
        self.test = self.classifier(test, ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin'], False)
        self.train = self.classifier(train, ['PassengerId', 'Age', 'Name', 'Cabin'], True)


    def classifier(self, df, cols, is_train):
        df_half = df[~df.Cabin.isna()]
        
        df_half['Cabin'] = [ 0 if i in [0, 1, 2, 3] else 1 for i in [ord(i[0]) - 65 if type(i) != float else int(i) - 28 for i in df_half.Cabin]]
        
        if (is_train == False):
            x = self.predict(df_half, cols, True) 
        
        df_outher_half = df[df.Cabin.isna()]
        predict = self.predict(df_outher_half, cols, False)
        
        
    def predict(self, df, cols, is_test):
        y = df.Cabin
        x = df.drop(columns=cols)
        
        if (is_test == True):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=337)
            self.clf.fit(x_train, y_train)
        else:
            x_test = x            
            
        y_pred = self.clf.predict(x_test)
            
        return y_pred
        
        
    # def clean_cabin(self, df, cols, is_test):
    #     # df = pd.DataFrame(df)
        
    #     if (is_test == False):
    #         df_cabin = df[~df.Cabin.isna()]
    #         df_cabin.Cabin = [ord(i[0]) - 65 if type(i) != float else int(i) - 28 for i in df_cabin.Cabin]
    #         df_cabin.Cabin = [ 0 if i in [0, 1, 2, 3] else 1 for i in df_cabin.Cabin]
    #     else:
    #         df_cabin = df[df.Cabin.isna()]

    #     y = df_cabin.Cabin
    #     x = df_cabin.drop(columns=cols)
        
    #     return [x, y]
    
        
    # def predict_cabin(df, test=0):
    #     df = pd.DataFrame(df)
        
    #     # Trein DataSet -> pd.read_csv('./train.csv')
    #     # Cleaning DataSet
    #     drop_train = ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin']
    #     x_train, y_train = clean_cabin(df, drop_train, False)  
        
    #     # Predict train DataSet
    #     x_train_aux, x_test_aux, y_train_aux, y_test_aux = train_test_split(x_train, y_train, test_size=0.30, random_state=337)
    #     clf = GradientBoostingClassifier()
    #     clf.fit(x_train_aux, y_train_aux)
    #     y_pred = clf.predict(x_test_aux)

    #     # Predict test DataSet
    #     # x_df_train, y_df_train = clean_cabin(df, drop_train, True)  
    #     # df_train = clf.predict(x_df_train) 
        
    #     # Predict final DataSet -> pd.read_csv('./test.csv')
    #     drop_test = ['PassengerId', 'Age', 'Name', 'Cabin']
    #     x_df_test, y_df_test = clean_cabin(test, drop_test, True)
    #     df_test = clf.predict(x_df_test)
        
    #     # join DataSet
    #     df.loc[~df['Cabin'].isna(), 'Cabin'] = y_train
    #     df.loc[df['Cabin'].isna(), 'Cabin'] = df_train
        
    #     useless, y_notna = clean_cabin(test, drop_test, 1)
    #     test.loc[~test['Cabin'].isna(), 'Cabin'] = y_notna
    #     test.loc[test['Cabin'].isna(), 'Cabin'] = df_test

    #     print(accuracy_score(y_pred,y_test_aux ))
        
    #     return [df, test]
        
    # df_3, test_2 = predict_cabin(df_2, test_1)