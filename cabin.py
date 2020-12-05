import pandas as pd
import numpy as np
import re
import matplotlib.pyploy as plt
import seaborn as sns

class cabin():
    def __init__(self, df):
        

    def clean_cabin(df, drop, na):
        df = pd.DataFrame(df)
        
        if (na == 1):
            df_cabin = df[~df.Cabin.isna()]
            df_cabin.Cabin = [ord(i[0]) - 65 if type(i) != float else int(i) - 28 for i in df_cabin.Cabin]
            df_cabin.Cabin = [ 0 if i in [0, 1, 2, 3] else 1 for i in df_cabin.Cabin]
        else:
            df_cabin = df[df.Cabin.isna()]

        y = df_cabin.Cabin
        x = df_cabin.drop(columns=drop)
        
        return [x, y]
        
    def predict_cabin(df, test=0):
        df = pd.DataFrame(df)
        
        # Trein DataSet -> pd.read_csv('./train.csv')
        # Cleaning DataSet
        drop_train = ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin']
        x_train, y_train = clean_cabin(df, drop_train, 1)  
        
        # Predict train DataSet
        x_train_aux, x_test_aux, y_train_aux, y_test_aux = train_test_split(x_train, y_train, test_size=0.30, random_state=337)
        clf = GradientBoostingClassifier()
        clf.fit(x_train_aux, y_train_aux)
        y_pred = clf.predict(x_test_aux)

        # Predict test DataSet
        x_df_train, y_df_train = clean_cabin(df, drop_train, 0)  
        df_train = clf.predict(x_df_train) 
        
        # Predict final DataSet -> pd.read_csv('./test.csv')
        drop_test = ['PassengerId', 'Age', 'Name', 'Cabin']
        x_df_test, y_df_test = clean_cabin(test, drop_test, 0)
        df_test = clf.predict(x_df_test)
        
        # join DataSet
        df.loc[~df['Cabin'].isna(), 'Cabin'] = y_train
        df.loc[df['Cabin'].isna(), 'Cabin'] = df_train
        
        useless, y_notna = clean_cabin(test, drop_test, 1)
        test.loc[~test['Cabin'].isna(), 'Cabin'] = y_notna
        test.loc[test['Cabin'].isna(), 'Cabin'] = df_test

        print(accuracy_score(y_pred,y_test_aux ))
        
        return [df, test]
        
    df_3, test_2 = predict_cabin(df_2, test_1)