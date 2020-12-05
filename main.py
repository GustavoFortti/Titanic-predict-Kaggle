import pandas as pd

from data import build_data

def main():

    test = pd.read_csv('./data/test.csv')
    train = pd.read_csv('./data/train.csv')

    # TREIN
    model = build_data(test, train)
    # CREATE NEW DATAFRAME
    # model.clean_data()
    # model.clean_ticket()
    # model.export_data('model.csv')
    df = model.get_data('model.csv')
    
    # TEST
    model_test = build_data(test_set=test, only_test=True)
    # model_test.clean_data()
    # model_test.clean_ticket()
    # model_test.export_data('model_test.csv')
    df_train = model_test.get_data('model_test.csv') 

    print(df)
    print(df_train)
    
if __name__ == "__main__":
    main()