# %%
import pandas as pd
import numpy as np
import catboost
from sklearn.preprocessing import StandardScaler
import utils
import matplotlib.ticker as mtick
from datetime import datetime
import os
from sklearn import metrics
import xgboost as xgb
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--detached", help="Detached test dates", action="store_true")
parser.add_argument("--bear", help="Train on bear weeks", action="store_true")
parser.add_argument("--bull", help="Train on bull weeks", action="store_true")
parser.add_argument("--friday", help="Train only on fridays", action="store_true")
parser.add_argument("-region", help="Train on region (NA, ROW, Global)", default="NA")
parser.add_argument("-val_pc", help="Validation percentage of train data", type=float, default=0.0)
parser.add_argument("--rand_val", help="Randomize validation date selection", action="store_true")

args = parser.parse_args()


run_time_and_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

os.chdir("../../")
print(os.listdir())


hp_iterations = 1000
hp_learning_rate = 0.02
hp_depth = 14
hp_random_seed = 42


validation_percentage = args.val_pc
detached_test_dates = args.detached
train_on_bear = args.bear
test_on_bear = False

train_on_bull = args.bull
test_on_bull = False

train_only_on_fridays = args.friday

sys.stdout.write("Detached test dates: " + str(detached_test_dates) + "\n")
sys.stdout.write("Train on bear weeks: " + str(train_on_bear) + "\n")
sys.stdout.write("Train on bull weeks: " + str(train_on_bull) + "\n")
sys.stdout.write("Train only on fridays: " + str(train_only_on_fridays) + "\n")
sys.stdout.write("Region to train on: " + str(args.region) + "\n")
sys.stdout.write("Validation percentage: " + str(validation_percentage) + "\n")
sys.stdout.write("Randomize validation dates: " + str(args.rand_val) + "\n")

assert(train_on_bear == False or train_on_bull == False)
assert(args.region == "NA" or args.region == "ROW" or args.region == "Global")

volume_usd_5_min = 1000000
market_cap_usd_min = 400000000

min_market_cap_percentile_na = 0.6
min_market_cap_percentile_global = 0.65

random_validation_dates = args.rand_val

target_horizon = 5
target_variable = f'trr_{target_horizon}'

# %%
final_features_df = pd.read_csv('final_features.csv', delimiter=';')
final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

# %%
if args.region == "NA":
    na_data_list = []
    na_data_list.append(pd.read_parquet('data/na_data_2001only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2002only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2003only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2004only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2005only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2006only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2007only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2008only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2009only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2010only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2011only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2012only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2013only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2014only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2015only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2016only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2017only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2018only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2019only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2020only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2021only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2022only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2023only_processed.parquet', engine='pyarrow'))


    na_data = pd.concat(na_data_list)

    na_data = utils.process_data(na_data, target_variable, target_horizon)

    na_data = na_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_na)]).reset_index(drop=True)
    na_data = na_data[na_data["volume_usd_5"] > volume_usd_5_min]

    data = na_data

elif args.region == "ROW":
    global_data_list = []
    global_data_list.append(pd.read_parquet('data/global_data_2001only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2002only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2003only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2004only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2005only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2006only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2007only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2008only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2009only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2010only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2011only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2012only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2013only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2014only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2015only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2016only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2017only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2018only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2019only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2020only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2021only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2022only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2023only_processed.parquet', engine='pyarrow'))


    global_data = pd.concat(global_data_list)

    global_data = utils.process_data(global_data, target_variable, target_horizon)

    global_data = global_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_global)]).reset_index(drop=True)
    global_data = global_data[global_data["volume_usd_5"] > volume_usd_5_min]

    data = global_data

elif args.region == "Global":
    global_data_list = []
    na_data_list = []

    # NA data
    na_data_list.append(pd.read_parquet('data/na_data_2001only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2002only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2003only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2004only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2005only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2006only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2007only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2008only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2009only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2010only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2011only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2012only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2013only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2014only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2015only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2016only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2017only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2018only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2019only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2020only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2021only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2022only_processed.parquet', engine='pyarrow'))
    na_data_list.append(pd.read_parquet('data/na_data_2023only_processed.parquet', engine='pyarrow'))

    # Global data
    global_data_list.append(pd.read_parquet('data/global_data_2001only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2002only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2003only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2004only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2005only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2006only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2007only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2008only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2009only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2010only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2011only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2012only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2013only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2014only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2015only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2016only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2017only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2018only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2019only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2020only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2021only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2022only_processed.parquet', engine='pyarrow'))
    global_data_list.append(pd.read_parquet('data/global_data_2023only_processed.parquet', engine='pyarrow'))
    global_data = pd.concat(global_data_list)
    na_data = pd.concat(na_data_list)

    global_data.sort_values(by=['date', 'gvkey'], inplace=True)
    na_data.sort_values(by=['date', 'gvkey'], inplace=True)

    global_data = utils.process_data(global_data, target_variable, target_horizon)
    na_data = utils.process_data(na_data, target_variable, target_horizon)

    global_data = global_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_global)]).reset_index(drop=True)
    na_data = na_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_na)]).reset_index(drop=True)

    global_data = global_data[global_data["volume_usd_5"] > volume_usd_5_min]
    na_data = na_data[na_data["volume_usd_5"] > volume_usd_5_min]

    # REMOVE CROSS-LISTED COMPANIES, KEEP ROW
    na_data["date"] = pd.to_datetime(na_data["date"], format='%Y-%m-%d')
    global_data["date"] = pd.to_datetime(global_data["date"], format='%Y-%m-%d')
    merged_data = pd.merge(global_data, na_data[["date", "gvkey", "volume_usd_1"]], on=['date', 'gvkey'], how='left', suffixes=('', '_na'))
    merged_data["volume_usd_1_na"] = merged_data["volume_usd_1_na"].fillna(0)

    merged_data['volume_usd_1'] = merged_data['volume_usd_1_na'] + merged_data['volume_usd_1']
    merged_data.drop(labels="volume_usd_1_na", axis="columns", inplace=True)

    na_global_data = pd.concat([merged_data, na_data], ignore_index=True)
    na_global_data.drop_duplicates(["date", "gvkey"], keep="first", inplace=True)

    na_global_data.sort_values(["date", "gvkey"], inplace=True)

    data = na_global_data

data = data[data["date"] < pd.Timestamp("2023-08-30")]

data[f'{target_variable}_fwd_class'] = 0
data.loc[data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x <= x.quantile(0.3333)), f'{target_variable}_fwd_class'] = -1
data.loc[data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x >= x.quantile(0.6666)), f'{target_variable}_fwd_class'] = 1

print (data[f'{target_variable}_fwd_class'].value_counts())

# %%
if train_on_bear:
    bear_weeks = pd.read_csv('bear_weeks.csv', delimiter=';')

    date_intervals = []
    for index, row in bear_weeks.iterrows():
        start_date = row['monday_start']
        end_date = row['monday_end']
        date_intervals.append((pd.to_datetime(start_date), pd.to_datetime(end_date)))

    filtered_data = pd.concat([data[(data['date'] >= start_date) & (data['date'] <= end_date)] 
                            for start_date, end_date in date_intervals])
    
    data = filtered_data

    
# %%
if train_on_bull:
    bull_weeks = pd.read_csv('bull_weeks.csv', delimiter=';')

    date_intervals = []
    for index, row in bull_weeks.iterrows():
        start_date = row['monday_start']
        end_date = row['monday_end']
        date_intervals.append((pd.to_datetime(start_date), pd.to_datetime(end_date)))

    filtered_data = pd.concat([data[(data['date'] >= start_date) & (data['date'] <= end_date)] 
                            for start_date, end_date in date_intervals])
    data = filtered_data


# %%
if detached_test_dates:
    test_data_list_na = []
    test_data_list_global = []

    test_data_list_na.append(pd.read_parquet('data/na_data_2022only_processed.parquet', engine='pyarrow'))
    test_data_list_na.append(pd.read_parquet('data/na_data_2023only_processed.parquet', engine='pyarrow'))
    
    test_data_list_global.append(pd.read_parquet('data/global_data_2022only_processed.parquet', engine='pyarrow'))
    test_data_list_global.append(pd.read_parquet('data/global_data_2023only_processed.parquet', engine='pyarrow'))
    
    test_data_na = pd.concat(test_data_list_na)
    test_data_global = pd.concat(test_data_list_global)
    
    # REMOVE CROSS-LISTED COMPANIES, KEEP ROW
    test_data_na["date"] = pd.to_datetime(test_data_na["date"], format='%Y-%m-%d')
    test_data_global["date"] = pd.to_datetime(test_data_global["date"], format='%Y-%m-%d')
    
    merged_data = pd.merge(test_data_global, test_data_na[["date", "gvkey", "volume_usd_1"]], on=['date', 'gvkey'], how='left', suffixes=('', '_na'))
    merged_data["volume_usd_1_na"] = merged_data["volume_usd_1_na"].fillna(0)

    merged_data['volume_usd_1'] = merged_data['volume_usd_1_na'] + merged_data['volume_usd_1']
    merged_data.drop(labels="volume_usd_1_na", axis="columns", inplace=True)

    test_data = pd.concat([merged_data, test_data_na], ignore_index=True)
    test_data.drop_duplicates(["date", "gvkey"], keep="first", inplace=True)

    test_data.sort_values(by=['date', 'gvkey'], inplace=True)
    
    test_data = utils.process_data(test_data, target_variable, target_horizon)

    test_data[f'{target_variable}_fwd_class'] = 0
    test_data.loc[test_data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x <= x.quantile(0.3333)), f'{target_variable}_fwd_class'] = -1
    test_data.loc[test_data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x >= x.quantile(0.6666)), f'{target_variable}_fwd_class'] = 1

    print(test_data[f'{target_variable}_fwd_class'].value_counts())

    detached_test_data = test_data.copy()


# %%
if train_only_on_fridays:
    data = data[data["date"].dt.weekday == 4]


# %%
if (detached_test_dates):
    assert(detached_test_dates == True)
    assert(detached_test_data.columns.tolist() == data.columns.tolist())
    
    

    drop_cols = ["gvkey", "company_name", "business_description", "city", "state", target_variable + "_fwd", target_variable + "_fwd_class", "date"]
    df_results = pd.DataFrame(columns=drop_cols + ["pred_0", "pred_1", "pred_2", "conviction", "conviction_class", "pred_class"])
    df_importances = pd.DataFrame(columns=["date", "feature", "importance"])


    test_start_date = "2023-10-13"
    test_start_date = pd.to_datetime(test_start_date, format="%Y-%m-%d")
    test_interval = 7
    test_end_date = "2021-12-24"
    test_end_date = pd.to_datetime(test_end_date, format="%Y-%m-%d")
    test_dates = pd.date_range(start= test_start_date, end=test_end_date, freq=f"{-test_interval}D")

    data = data[data["date"] < test_end_date]

    data_train, data_validation, data_test = utils.create_train_validation_test(data, test_date=None, validation=validation_percentage, horizon_days=target_horizon)
    
    train_dates = pd.DataFrame(columns = ["train_dates", "first_test_date"], data = [[data_train["date"].unique(), test_dates.min()]])
        

    X_train = data_train.drop(columns=drop_cols)
    y_train = data_train[target_variable + "_fwd_class"]



    categorical_features = final_features_df[final_features_df["categorical"] == True]["final_feature"].to_list()
    categorical_features = [x for x in categorical_features if x not in drop_cols]

    for col in categorical_features:
        if X_train[col].isnull().any():
            X_train[col] = X_train[col].cat.add_categories("Unknown").fillna("Unknown")
    for col in X_train.columns:
        if col not in categorical_features:
            X_train[col] = X_train[col].fillna(0)
            X_train[col] = X_train[col].replace([np.inf], 1000000000000000)
            X_train[col] = X_train[col].replace([-np.inf], -1000000000000000)

    if validation_percentage > 0:
        X_val = data_validation.drop(columns=drop_cols)
        y_val = data_validation[target_variable + "_fwd_class"]

        for col in categorical_features:
            if X_val[col].isnull().any():
                X_val[col] = X_val[col].cat.add_categories("Unknown").fillna("Unknown")
        for col in X_val.columns:
            if col not in categorical_features:
                X_val[col] = X_val[col].fillna(0)
                X_val[col] = X_val[col].replace([np.inf], 1000000000000000)
                X_val[col] = X_val[col].replace([-np.inf], -1000000000000000)

        model = catboost.CatBoostClassifier(iterations=hp_iterations, 
                                            learning_rate=hp_learning_rate, 
                                            depth=hp_depth,
                                            loss_function='MultiClass', 
                                            eval_metric="Accuracy", 
                                            random_seed=hp_random_seed,
                                            task_type="GPU",
                                            devices='0',
                                            verbose=True)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, cat_features=categorical_features)

    else:
        model = catboost.CatBoostClassifier(iterations=hp_iterations, 
                                            learning_rate=hp_learning_rate, 
                                            depth=hp_depth,
                                            loss_function='MultiClass', 
                                            eval_metric="Accuracy", 
                                            random_seed=hp_random_seed,
                                            task_type="GPU",
                                            devices='0',
                                            verbose=True)
        model.fit(X_train, y_train, plot=True, cat_features=categorical_features)

    importance = model.get_feature_importance(prettified=True)
    df_importances = pd.DataFrame({"date": "detached", "feature": importance["Feature Id"], "importance": importance["Importances"]})

    print(test_dates)
    for test_date in test_dates:
        print("Test date: ")
        print(test_date)
        
        data_test = detached_test_data[detached_test_data["date"] == test_date]
        
        if data_test.empty:
            continue

        X_test = data_test.drop(columns=drop_cols)
        y_test = data_test[target_variable + "_fwd_class"]
        test_actual = data_test.copy()


        for col in categorical_features:
            if X_test[col].isnull().any():
                X_test[col] = X_test[col].cat.add_categories("Unknown").fillna("Unknown")
        for col in X_train.columns:
            if col not in categorical_features:
                X_test[col] = X_test[col].fillna(0)
                X_test[col] = X_test[col].replace([np.inf], 1000000000000000)
                X_test[col] = X_test[col].replace([-np.inf], -1000000000000000)

        preds_proba = model.predict_proba(X_test)
        preds_class = model.predict(X_test)

        test_actual["pred_0"] = preds_proba[:, 0]
        test_actual["pred_1"] = preds_proba[:, 1]
        test_actual["pred_2"] = preds_proba[:, 2]
        test_actual["conviction"] = preds_proba[:, 2] - preds_proba[:, 0]
        test_actual["conviction_class"] = 0
        test_actual.loc[test_actual['conviction'] <= test_actual['conviction'].quantile(0.3333), 'conviction_class'] = -1
        test_actual.loc[test_actual['conviction'] >= test_actual['conviction'].quantile(0.6666), 'conviction_class'] = 1
        test_actual["pred_class"] = preds_class

        
        df_results = pd.concat([df_results, test_actual])

# %%
else:
    assert(detached_test_dates == False)

    drop_cols = ["gvkey", "company_name", "business_description", "city", "state", target_variable + "_fwd", target_variable + "_fwd_class", "date"]
    df_results = pd.DataFrame(columns=drop_cols + ["pred_0", "pred_1", "pred_2", "conviction", "conviction_class", "pred_class"])
    df_importances = pd.DataFrame(columns=["date", "feature", "importance"])

    train_dates = pd.DataFrame(columns=["train_dates", "first_test_date"])

    test_start_date = "2023-08-18"
    test_start_date = pd.to_datetime(test_start_date, format="%Y-%m-%d")
    test_interval = 7
    test_end_date = "2022-01-01"
    test_end_date = pd.to_datetime(test_end_date, format="%Y-%m-%d")
    test_dates = pd.date_range(start= test_start_date, end=test_end_date, freq=f"{-test_interval}D")
    print(test_dates)
    for test_date in test_dates:
        print("Test date: ")
        print(test_date)
        data_train, data_validation, data_test = utils.create_train_validation_test(data, test_date=test_date, validation=validation_percentage, horizon_days=5)

        if data_test is None:
            continue
        
        train_dates.loc[len(train_dates)] = [data_train["date"].unique(), data_test["date"].min()]

        X_train = data_train.drop(columns=drop_cols)
        y_train = data_train[target_variable + "_fwd_class"]
        
        X_test = data_test.drop(columns=drop_cols)
        y_test = data_test[target_variable + "_fwd_class"]
        test_actual = data_test.copy()

        categorical_features = final_features_df[final_features_df["categorical"] == True]["final_feature"].to_list()
        categorical_features = [x for x in categorical_features if x not in drop_cols]

        for col in categorical_features:
            if X_train[col].isnull().any() or X_test[col].isnull().any():
                X_train[col] = X_train[col].cat.add_categories("Unknown").fillna("Unknown")
                X_test[col] = X_test[col].cat.add_categories("Unknown").fillna("Unknown")
        for col in X_train.columns:
            if col not in categorical_features:
                X_train[col] = X_train[col].fillna(0)
                X_test[col] = X_test[col].fillna(0)

                X_train[col] = X_train[col].replace([np.inf], 1000000000000000)
                X_test[col] = X_test[col].replace([np.inf], 1000000000000000)
                
                X_train[col] = X_train[col].replace([-np.inf], -1000000000000000)
                X_test[col] = X_test[col].replace([-np.inf], -1000000000000000)


        if validation_percentage > 0:
            X_val = data_validation.drop(columns=drop_cols)
            y_val = data_validation[target_variable + "_fwd_class"]
            for col in categorical_features:
                if X_val[col].isnull().any():
                    X_val[col] = X_val[col].cat.add_categories("Unknown").fillna("Unknown")
            for col in X_val.columns:
                if col not in categorical_features:
                    X_val[col] = X_val[col].fillna(0)
                    X_val[col] = X_val[col].replace([np.inf], 1000000000000000)
                    X_val[col] = X_val[col].replace([-np.inf], -1000000000000000)
                
            model = catboost.CatBoostClassifier(iterations=hp_iterations, 
                                                learning_rate=hp_learning_rate, 
                                                depth=hp_depth,
                                                loss_function='MultiClass', 
                                                eval_metric="Accuracy", 
                                                random_seed=hp_random_seed,
                                                task_type="GPU",
                                                devices='0',
                                                verbose=True)    
            model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=True, cat_features=categorical_features)

        else:
            model = catboost.CatBoostClassifier(iterations=hp_iterations, 
                                                learning_rate=hp_learning_rate, 
                                                depth=hp_depth,
                                                loss_function='MultiClass', 
                                                eval_metric="Accuracy", 
                                                random_seed=hp_random_seed,
                                                task_type="GPU",
                                                devices='0',
                                                verbose=True)
            model.fit(X_train, y_train, plot=True, cat_features=categorical_features)

        preds_proba = model.predict_proba(X_test)
        preds_class = model.predict(X_test)

        test_actual["pred_0"] = preds_proba[:, 0]
        test_actual["pred_1"] = preds_proba[:, 1]
        test_actual["pred_2"] = preds_proba[:, 2]
        test_actual["conviction"] = preds_proba[:, 2] - preds_proba[:, 0]
        test_actual["conviction_class"] = 0
        test_actual.loc[test_actual['conviction'] <= test_actual['conviction'].quantile(0.3333), 'conviction_class'] = -1
        test_actual.loc[test_actual['conviction'] >= test_actual['conviction'].quantile(0.6666), 'conviction_class'] = 1
        test_actual["pred_class"] = preds_class
        importance = model.get_feature_importance(prettified=True)
        df_importance = pd.DataFrame({"date": test_date, "feature": importance["Feature Id"], "importance": importance["Importances"]})
        
        df_results = pd.concat([df_results, test_actual])
        df_importances = pd.concat([df_importances, df_importance])
    



# %%
df_results["pred_0"] = pd.to_numeric(df_results["pred_0"])
df_results["pred_1"] = pd.to_numeric(df_results["pred_1"])
df_results["pred_2"] = pd.to_numeric(df_results["pred_2"])
df_results["conviction"] = pd.to_numeric(df_results["conviction"])

# %%
model_name = "catboost_" + str(args.region) + "_" + run_time_and_date
if detached_test_dates:
    model_name = model_name + "_detached"
if train_on_bear:
    model_name = model_name + "_bear_trained"
if train_on_bull:
    model_name = model_name + "_bull_trained"
if train_only_on_fridays:
    model_name = model_name + "_friday_trained"

if validation_percentage > 0:
    if random_validation_dates:
        model_name = model_name + "_random_validation"
    else:
        model_name = model_name + "_sequential_validation"
else:
    model_name = model_name + "_no_validation"

model_name = model_name + "_min_vol_5_" + str(volume_usd_5_min) + "_min_mcap_percentile_na_" + str(min_market_cap_percentile_na) + "_min_mcap_percentile_global_" + str(min_market_cap_percentile_global)



# %% SAVE RESULTS
df_results['model'] = model_name
df_results['test_interval'] = test_interval

if not os.path.exists(f"results/{model_name}"):
    os.makedirs(f"results/{model_name}")

hyperparameters = model.get_all_params()
with open(f"results/{model_name}/hyperparameters.txt", 'w') as f:
    for key, value in hyperparameters.items():
        f.write('%s:%s\n' % (key, value))

df_results.to_parquet(f"results/{model_name}/results.parquet", index=False)

train_dates.to_csv(f"results/{model_name}/train_dates.csv", index=False)
df_importances.to_csv(f"results/{model_name}/all_importances.csv")

mean_importances = df_importances.groupby("feature")["importance"].mean().sort_values(ascending=False)
mean_importances.to_csv(f"results/{model_name}/importances.csv")



