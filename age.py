import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class build_age():
    
    def __init__(self):
        self.clf = GradientBoostingRegressor()
    
    def train(self, df, cols):
        self.modify(df, cols, True)
        return self.modify(df, cols, False)
        
    def test(self, df, cols):
        return self.modify(df, cols, False)
        
    def modify(self, df, cols, is_train):
        df_to_age = self.clean(df)
        df = df.join(df_to_age)
        
        df_notna = df[df.Age.notna()]

        if (is_train):
            self.predict(df_notna, cols, True)
        else:
            df_na = df[df.Age.isna()]
            df.loc[df.Age.isna(), 'Age'] = self.predict(df_na, cols, False)
            df['Age'] = df['Age'].astype('int32')
            
            return df
        
    def predict(self, df, cols, is_train):
        x = df.drop(columns=cols) 

        if (is_train):
            y = df['Age']
            y = ([int(value) for value in y])

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=337)
            self.clf.fit(x_train, y_train)
        
        else:
            x_test = x
            
        y_pred = self.clf.predict(x_test)
        
        return y_pred
    
    def clean(self, df):
        Miss = [int(y) for y in df[(df.Name.str.contains('Miss', na=False))].Age.describe()[2:8]]
        Mrs = [int(y) for y in df[(df.Name.str.contains('Mrs', na=False))].Age.describe()[2:8]]
        Mr = [int(y) for y in df[(df.Name.str.contains('Mr', na=False))].Age.describe()[2:8]]
        Master = [int(y) for y in df[(df.Name.str.contains('Master', na=False))].Age.describe()[2:8]]
        Outer = [int(y) for y in df[~(df.Name.str.contains('Miss|Mrs|Master|Mr', na=False))].Age.describe()[2:8]]
        
        describe = [Miss, Mrs, Mr, Master, Outer]
        
        age = pd.DataFrame([self.filter_age(i, describe) for i in df.Name])

        age.columns = ['x', 'y']
        
        age.y = [ int(str(i)[1: -1])for i in age.y]
        age = age.join(pd.DataFrame([str(i)[1: -1].split(',') for i in age.x]))
        age = age.drop(columns=['x'])
        age.columns = ['Title', 'Min', 'Per_25', 'Per_50', 'Pre_75', 'Max', 'Std']


        for i in age.columns:
            age[i] = age[i].astype('int32')
        
        return age

    def filter_age(self, title, values):

        describe = []
        if('Master' in str(title)):
            describe = values[3]
            li_title = [1]
        elif('Mrs' in str(title)):
            describe = values[1]
            li_title = [5]
        elif('Mr' in str(title)):
            describe = values[2]
            li_title = [2]
        elif('Miss' in str(title)):
            describe = values[0]
            li_title = [3]
        else :
            describe = values[4]
            li_title = [4]
            
        return [describe, li_title]
        
        