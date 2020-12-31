import pandas as pd

from data import build_data
from cabin import build_cabin
from age import build_age
from survived import build_survived

def main():

    test = pd.read_csv('./data/test.csv')
    train = pd.read_csv('./data/train.csv')
    
    ### Build data
    
    # TRAIN
    model = build_data(test, train)
    # CREATE NEW DATAFRAME
    model.clean_data() 
    model.clean_ticket()
    model.export_data('model_train.csv')
    df_train = model.get_data('model_train.csv')
    
    # TEST
    model_test = build_data(test_set=test, only_test=True)
    model_test.clean_data()
    model_test.clean_ticket()
    model_test.export_data('model_test.csv')
    df_test = model_test.get_data('model_test.csv') 

    ### Build predict to cabin
    
    # cabin = build_cabin(df_test, df_train)
    cabin = build_cabin()
    df_train_2 = cabin.train(df_train, ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin'])
    df_test_2 = cabin.test(df_test, ['PassengerId', 'Age', 'Name', 'Cabin'])
    # print(cabin.get_accuracy())

    # print(df_test_2.head(50))
    # print(df_train_2.head(50))
    
    ### Build predict to age
    
    age = build_age()
    df_train_3 = age.train(df_train_2, ['PassengerId', 'Name', 'Age', 'Survived'])
    df_test_3 = age.test(df_test_2, ['PassengerId', 'Name', 'Age'])
        
    # print(df_test_3.Embarked.describe())
    # print(df_train_3.Embarked.describe())
    
    ### Build predict to survived
    
    cols_train = ['Std', 'Min' ,'Per_25', 'Per_50', 'Pre_75', 'Max', 'PassengerId', 'Name', 'Title', 'SibSp', 'Parch', 'Survived']
    cols_test = ['Std', 'Min' ,'Per_25', 'Per_50', 'Pre_75', 'Max', 'PassengerId', 'Name', 'Title', 'SibSp', 'Parch']
    
    survived = build_survived()
    survived.train(df_train_3, cols_train)
    print(survived.get_accuracy())
    resp = survived.test(df_test_3, cols_test)
    # print(resp)
    
if __name__ == "__main__":
    main()