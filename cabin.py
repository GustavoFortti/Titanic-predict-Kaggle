from operator import index
from matplotlib.pyplot import axes, axis
import pandas as pd
import numpy as np

from scipy.sparse.construct import random
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
class build_cabin():
    def __init__(self):
        pd.options.mode.chained_assignment = None 
        self.clf = GradientBoostingClassifier()

    def train(self, df, cols):
        self.classifier(df, cols, True)
        self.df_train = self.classifier(df, cols, False)
        return self.df_train
    
    def test(self, df, cols): 
        self.df_test = self.classifier(df, cols, False)
        return self.df_test
        
    def classifier(self, df, cols, is_train):
        df_half = df[~df.Cabin.isna()]
        df_half['Cabin'] = [ 0 if i in [0, 1, 2, 3] else 1 for i in [ord(i[0]) - 65 if type(i) != float else int(i) - 28 for i in df_half.Cabin]]
        
        if (is_train == True):
            self.predict(df_half, cols, True)
        else:
            df.loc[~df.Cabin.isna(), 'Cabin'] = df_half.Cabin
            
            df_outher_half = df[df.Cabin.isna()]
            df.loc[df.Cabin.isna(), 'Cabin']  = self.predict(df_outher_half, cols, False)

            return df
        
    def predict(self, df, cols, is_train):
        y = df.Cabin
        x = df.drop(columns=cols)
        
        if (is_train == True):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=337)
            self.clf.fit(x_train, y_train)
        else:
            x_test = x            
            
        y_pred = self.clf.predict(x_test)
        
        # print(y_pred)
        return y_pred