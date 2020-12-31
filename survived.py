import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class build_survived():
    def __init__(self):
        self.clf = GradientBoostingClassifier()
        self.accuracy = "Not found"
        
    def train(self, df, cols):
        self.modify(df, cols, True)
        
    def test(self, df, cols):
        resp = pd.DataFrame(self.modify(df, cols, False), index=df.PassengerId, columns=['Survived'])
        resp['Survived'] = resp['Survived'].astype('int32')
        resp.to_csv('./data/Survived.csv')
        return resp
    
    def modify(self, df, cols, is_train):
        df['Alone'] = [0 if i == 0 else 1 for i in df.SibSp + df.Parch]
        
        if (is_train):
            df_notna = df[~df.Survived.isna()]
            self.predict(df_notna, cols, True)
        else:
            return self.predict(df, cols, False)
    
    def predict(self, df, cols, is_train):
        x = df.drop(columns=cols)
        
        if (is_train):
            y = df['Survived']
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=337)
            self.clf.fit(x_train, y_train)
        else:
            x_test = x
            
        y_pred = self.clf.predict(x_test)
        
        if (is_train):
            self.accuracy = accuracy_score(y_test, y_pred)
            
        return y_pred
    
    def get_accuracy(self):
        return self.accuracy