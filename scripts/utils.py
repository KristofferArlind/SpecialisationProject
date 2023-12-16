import pandas as pd
import numpy as np

def process_data(data, target_variable, horizon, weekday = None):

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    fwd_variable = f'{target_variable}_fwd'

    final_features_df = pd.read_csv('final_features.csv', delimiter=';')
    final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

    data = data[final_features]

    dtype_dict = dict(zip(final_features_df["final_feature"], final_features_df["feature_type"]))
    del dtype_dict['date']
    del dtype_dict[np.nan]


    for index, feature in final_features_df.iterrows():
        if pd.isnull(feature["final_feature"]) or feature["final_feature"] == "date":
            continue
        if feature["feature_type"] == "string":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(str)
        elif feature["feature_type"] == "int" and feature["categorical"] == True:
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(str)
        elif feature["feature_type"] == "int":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(int)
        elif feature["feature_type"] == "float":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(float)
        if feature["categorical"] == True:
            data[feature["final_feature"]] = data[feature["final_feature"]].astype("category")
            

    data[fwd_variable] = data.groupby(['gvkey'])[target_variable].shift(-horizon)

    max_date = data['date'].unique().max()
    min_date = data['date'].unique().min()
    data = data[(data["date"] > min_date) & (data["date"] < max_date)]

    data.dropna(subset=[fwd_variable], inplace=True)
    
    data[fwd_variable] = data[fwd_variable].astype(float)

    #REMOVES NEXT TO ZERO STOCKS, LEFTOVER FROM BEFORE BUT STILL USED IN THE PROJECT, SO KEPT HERE:
    data = data[data[fwd_variable] <= 2]
    data = data[data[fwd_variable] >= -0.7]
    
    assert(data[fwd_variable].isnull().sum() == 0)
    assert((data[fwd_variable] < -0.7).sum() == 0)
    assert((data[fwd_variable] > 2).sum() == 0)

    if weekday != None:
        data = data[data["date"].dt.weekday == weekday]

    return data

def create_train_validation_test(data, test_date, validation=0.2, horizon_days=5, random_validation_dates=False):
    data_test = None
    data_validation = None

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    all_dates = data["date"].unique()

    if (test_date != None):
        test_date = pd.to_datetime(test_date, format="%Y-%m-%d")

        all_dates = (all_dates[all_dates < test_date])

        data_test = data[data["date"] == test_date]
        print("Test size: ", len(data_test))
        if len(data_test) == 0:
            print("Test date not in dataset")
            return None, None, None
        print("Test days: ", data_test["date"].unique().min() , "to", data_test["date"].unique().max())

        print("Dropping dates: ")
        print(all_dates[-horizon_days:])
        all_dates = all_dates[:-horizon_days]

    if (random_validation_dates):
        np.random.shuffle(all_dates)
        
    training_days, validation_days = np.split(all_dates, [int(len(all_dates)* (1-validation))])
    
    if validation > 0:
        training_days = training_days[:-horizon_days]
        data_validation = data[data["date"].isin(validation_days)]
        print("Validation size: ", len(data_validation))
        print("Validation days: ", data_validation["date"].unique().min(), "to", data_validation["date"].unique().max())
    
    data_train = data[data["date"].isin(training_days)]
    
    print("Train size: ", len(data_train))
    print("Train days: ", data_train["date"].unique().min() , "to", data_train["date"].unique().max())

    return data_train, data_validation, data_test


