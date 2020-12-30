import pandas as pd

from data import build_data
from cabin import build_cabin

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

    # ### Build predict to cabin
    
    # cabin = build_cabin(df_test, df_train)
    cabin = build_cabin()
    df_train_2 = cabin.train(df_train, ['PassengerId', 'Survived', 'Age', 'Name', 'Cabin'])
    df_test_2 = cabin.test(df_test, ['PassengerId', 'Age', 'Name', 'Cabin'])

    print(df_test_2.head(50))
    print(df_train_2.head(50))
    
if __name__ == "__main__":
    main()