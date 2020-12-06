import pandas as pd

from data import build_data
from cabin import build_cabin

def main():

    test = pd.read_csv('./data/test.csv')
    train = pd.read_csv('./data/train.csv')
    
    
    ### Build data
    
    # TREIN
    model = build_data(test, train)
    # CREATE NEW DATAFRAME
    # model.clean_data() 
    # model.clean_ticket()
    # model.export_data('model_test.csv')
    df_test = model.get_data('model_test.csv')
    
    # TEST
    model_test = build_data(test_set=test, only_test=True)
    # model_test.clean_data()
    # model_test.clean_ticket()
    # model_test.export_data('model_test.csv')
    df_train = model_test.get_data('model_train.csv') 

    ### Build predict to cabin
    
    cabin = build_cabin(df_test, df_train)
    
    

    # print(df_test)
    # print(df_train)
    
if __name__ == "__main__":
    main()