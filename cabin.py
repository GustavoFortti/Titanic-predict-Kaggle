import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class build_cabin():
    def __init__(self):
        pd.options.mode.chained_assignment = None
        self.clf = GradientBoostingClassifier()
        self.accuracy = "Not found"

    def train(self, df, cols):
        self.modify(df, cols, True)
        df_train = self.modify(df, cols, False)
        return df_train
    
    def test(self, df, cols): 
        df_test = self.modify(df, cols, False)
        return df_test
        
    def modify(self, df, cols, is_train):
        df_notna = df[~df.Cabin.isna()]
        df_notna['Cabin'] = [ 0 if i in [0, 1, 2, 3] else 1 for i in [ord(i[0]) - 65 if type(i) != float else int(i) - 28 for i in df_notna.Cabin]]
        
        if (is_train == True):
            self.predict(df_notna, cols, True)
        else:
            df.loc[~df.Cabin.isna(), 'Cabin'] = df_notna.Cabin
            
            df_na = df[df.Cabin.isna()]
            df.loc[df.Cabin.isna(), 'Cabin']  = self.predict(df_na, cols, False)

            return df
        
    def predict(self, df, cols, is_train):
        x = df.drop(columns=cols)
        
        if (is_train == True):
            y = df.Cabin
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=337)
            self.clf.fit(x_train, y_train)
        else:
            x_test = x            
            
        y_pred = self.clf.predict(x_test)
        
        if (is_train == True):
            self.accuracy = accuracy_score(y_test, y_pred)
        
        return y_pred
    
    def get_accuracy(self):
        return (self.accuracy)
