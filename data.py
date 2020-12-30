import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class build_data():
    def __init__(self, test_set, train_set=None, only_test=False):
        self.test = test_set

        if only_test == False:
            self.train = train_set
            self.df = [self.train, self.test]
            self.df = pd.concat(self.df, sort=False)
        
            self.df = self.df.loc[:, ['PassengerId', 'Name', 'Pclass', 'Sex', 'Parch', 'SibSp', 'Survived', 'Ticket', 'Age', 'Cabin', 'Fare', 'Embarked']].reset_index().drop(columns='index')
        else:
            self.df = self.test

    def clean_data(self):
        
        # Simplificando valores
        self.df.Embarked = [ 0 if i == 'S' else 1 if i == 'C' else 2 for i in self.df.Embarked]
        self.df.Sex = [1 if i == "male" else 0 for i in self.df.Sex]
        
        # Corrigindo NA
        self.df[self.df.Embarked.isna()].Embarked = int(self.df.Embarked.mean())
        self.df.loc[self.df['Fare'].isna(), 'Fare'] = self.df.Fare.mean()
        # self.df.Fare = [ int(i) for i in self.df.Fare]
        
    def clean_ticket(self):
        Ticket = pd.DataFrame([i.split(' ') for i in self.df.Ticket])
        Ticket.columns = ['Type', 'Sub', 'Num']

        a = []
        b = []
        c = []
        for i, k, j in zip(Ticket.Type, Ticket.Sub, Ticket.Num):
            try:
                c.append(int(i))
                b.append('ZZZ')
                a.append('ZZZ')
            except:
                try:
                    c.append(int(k))
                    b.append('ZZZ')
                    a.append(i)
                except:
                    c.append(j)
                    b.append(k)
                    a.append(i)

        dic = {
            'Tck1': a,
            'Tck2': b,
            'Tck3': c
        }

        Ticket = (pd.DataFrame(dic))
        Ticket.loc[Ticket['Tck3'].isna(), 'Tck3'] = 0
        Ticket.loc[Ticket['Tck2'].isna(), 'Tck2'] = 'NF'
        Ticket.Tck3 = Ticket.Tck3.astype(int)
        
        a_rang = np.arange(0, Ticket.groupby('Tck1').count().shape[0])
        
        a_dic = {
            'inx': Ticket.groupby('Tck1').count().index,
            'val': a_rang
        }
        
        a_dic = pd.DataFrame(a_dic)
        Ticket_a = []

        for i in Ticket.Tck1:
            for j, k in zip(a_dic.inx, a_dic.val):
                if i == j:
                    Ticket_a.append(k)

        Ticket.Tck1 = Ticket_a
        
        b_rang = np.arange(0, Ticket.groupby('Tck2').count().shape[0])
        
        b_dic = {
            'inx': Ticket.groupby('Tck2').count().index,
            'val': b_rang
        }
        
        b_dic = pd.DataFrame(b_dic)
        Ticket_b = []

        for i in Ticket.Tck2:
            for j, k in zip(b_dic.inx, b_dic.val):
                if i == j:
                    Ticket_b.append(k)

        Ticket.Tck2 = Ticket_b
        
        self.df = self.df.join(Ticket)
        self.df = self.df.drop(columns='Ticket')
        
    def export_data(self, name):
        self.name = name
        self.df.to_csv('./data/' + self.name, index=False)
    
    def get_data(self, name=False):
        if name != False:
            self.name = name

        df = []

        try:
            df = pd.read_csv('./data/' + self.name)
        except:
            print('File not found')
            
        return df