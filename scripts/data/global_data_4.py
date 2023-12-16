
import pandas as pd
import numpy as np
import os
os.chdir("../")
pd.set_option('display.max_columns', 100)


global_data = pd.read_csv('global_data_1990_2002.zip', dtype={'priusa': str, 'prican' : str, 'prirow' : str, 'iid' : str})

print("Loaded data")

# In[] RENAME COLUMNS:
column_name_dict = {'gvkey': 'gvkey',
                    'iid': 'iid',
                    'datadate': 'date',
                    'conm': 'company_name',
                    'curcddv': 'currency_dividend',
                    'cheqv': 'cash_equivalent',
                    'div': 'dividend_loc',
                    'divd': 'dividend_cash',
                    'divsp': 'dividend_special',
                    'anncdate': 'dividend_declare_date',
                    'cheqvpaydate': 'cash_equivalent_pay_date',
                    'divsppaydate': 'dividend_special_pay_date',
                    'recorddate': 'dividend_record_date',
                    'curcdd': 'currency',
                    'ajexdi': 'adjustment_factor',
                    'cshoc': 'shares_outstanding',
                    'cshtrd': 'volume_shares',
                    'prccd': 'price_close',
                    'prchd': 'price_high',
                    'prcld': 'price_low',
                    'prcstd': 'price_status_code',
                    'trfd': 'total_return_factor',
                    'exchg': 'exchange_code',
                    'secstat': 'security_status',
                    'tpci': 'issue_type_code',
                    'cik': 'cik',
                    'fic': 'foreign_incorporation_code',
                    'add1': 'address_line_1',
                    'add2': 'address_line_2',
                    'add3': 'address_line_3',
                    'add4': 'address_line_4',
                    'addzip': 'address_zip',
                    'busdesc': 'business_description',
                    'city': 'city',
                    'conml': 'company_legal_name',
                    'costat': 'company_status',
                    'county': 'county',
                    'dlrsn': 'deletion_reason',
                    'ein': 'ein',
                    'fax': 'fax',
                    'fyrc': 'fiscal_year_end_month',
                    'ggroup': 'ggroup',
                    'gind': 'gind',
                    'gsector': 'gsector',
                    'gsubind': 'gsubind',
                    'idbflag': 'idb_flag',
                    'incorp': 'incorporation_code',
                    'loc': 'country_hq',
                    'naics': 'naics',
                    'phone': 'phone',
                    'prican': 'primary_canada',
                    'prirow': 'primary_row',
                    'priusa': 'primary_usa',
                    'sic': 'sic',
                    'spcindcd': 'sp_industry_code',
                    'spcseccd': 'sp_sector_code',
                    'spcsrc': 'sp_quality',
                    'state': 'state',
                    'stko': 'stko',
                    'weburl': 'web_url',
                    'dldte': 'download_date',
                    'ipodate': 'ipo_date'}


global_data = global_data[column_name_dict.keys()]
global_data.rename(columns=column_name_dict, inplace=True)
# In[]:
global_data = global_data.sort_values(by=['date', 'gvkey', 'iid'])

# In[]:
global_data = global_data[global_data["volume_shares"].notnull()]
global_data = global_data[global_data["volume_shares"] != 0]
global_data = global_data[global_data["price_close"].notnull()]
global_data = global_data[global_data["price_close"] != 0]




# In[] CURRENCY DATA:
forex_data = pd.read_csv('forex_data.csv')
global_data = global_data[~global_data['currency'].isnull()]

global_data['date'] = pd.to_datetime(global_data['date'])
forex_data['date'] = pd.to_datetime(forex_data['date'])
global_data.reset_index(drop=True, inplace=True)
forex_data.reset_index(drop=True, inplace=True)

global_data = global_data[global_data['currency'].isin(forex_data.columns[1:].tolist() + ["USD"])]
global_data = global_data[global_data['currency_dividend'].isin(forex_data.columns[1:].tolist()) | global_data['currency_dividend'].isnull()]

# In[] REMOVE OTC EXCHANGES AND NON NORMAL SHARES:
disallowed_exchange_codes = [13, 19, 229, 290]
global_data = global_data[~global_data["exchange_code"].isin(disallowed_exchange_codes)]

global_data["issue_type_code"] = global_data["issue_type_code"].astype(str)
global_data = global_data[global_data["issue_type_code"].isin(["0"])]


# In[] ADJUST PRICE_CLOSE:
global_data["price_close_adj"] = global_data["price_close"] / global_data["adjustment_factor"]

# In[] CONVERT TO USD:
merged_data = pd.merge(global_data, forex_data, left_on="date", right_on="date", how="left")

merged_data["market_cap_loc"] = merged_data["price_close"] * merged_data["shares_outstanding"]
merged_data["market_cap_usd"] = merged_data.apply(lambda row: row["market_cap_loc"] if row["currency"] == "USD" else row["market_cap_loc"] / row[row["currency"]], axis=1)
merged_data["price_close_usd"] = merged_data.apply(lambda row: row["price_close"] if row["currency"] == "USD" else row["price_close"] / row[row["currency"]], axis=1)

merged_data["volume_loc"] = merged_data["volume_shares"] * merged_data["price_close"]
merged_data["volume_usd_1"] = merged_data.apply(lambda row: row["volume_loc"] if row["currency"] == "USD" else row["volume_loc"] / row[row["currency"]], axis=1)
merged_data["dividend_usd"] = merged_data.apply(
    lambda row: row["dividend_loc"] if row["currency_dividend"] == "USD" 
               else np.nan if pd.isna(row["currency_dividend"]) 
               else row["dividend_loc"] / row[row["currency_dividend"]] if not pd.isna(row[row["currency_dividend"]]) and row[row["currency_dividend"]] != 0 
               else np.nan,
    axis=1
)

currency_columns = forex_data.columns[1:].tolist() 
global_data = merged_data.drop(columns=currency_columns)

# In[] REMOVE DUPLICATE LISTINGS. KEEP HIGHEST AVG DAILY VOLUME, SUM VOLUMES:
global_data["total_volume_usd_1"] = global_data.groupby(["gvkey", "date"])["volume_usd_1"].transform("sum")

global_data = global_data[(global_data["iid"] == global_data["primary_row"]) | (global_data["primary_row"].isna())]

avg_volumes = global_data.groupby(['gvkey', 'iid']).apply(lambda x: x['volume_usd_1'].mean()).reset_index()
avg_volumes.columns = ['gvkey', 'iid', 'avg_volume']
max_volumes = avg_volumes.groupby('gvkey').apply(lambda x: x[x['avg_volume'] == x['avg_volume'].max()]).reset_index(drop=True)
global_data = pd.merge(global_data, max_volumes[['gvkey', 'iid']], on=['gvkey', 'iid'], how='inner')
global_data.drop_duplicates(subset=["date", "gvkey"], inplace=True)

global_data["volume_usd_1"] = global_data["total_volume_usd_1"]
global_data.drop(labels=["total_volume_usd_1"], axis="columns", inplace=True)

print((global_data.groupby('gvkey')['iid'].nunique() > 1).sum())

# In[] TECHNICAL HORIZONS:
global_data["volume_usd_2"]   = global_data.groupby("gvkey")["volume_usd_1"].rolling(2  ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_3"]   = global_data.groupby("gvkey")["volume_usd_1"].rolling(3  ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_5"]   = global_data.groupby("gvkey")["volume_usd_1"].rolling(5  ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_10"]  = global_data.groupby("gvkey")["volume_usd_1"].rolling(10 ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_20"]  = global_data.groupby("gvkey")["volume_usd_1"].rolling(20 ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_30"]  = global_data.groupby("gvkey")["volume_usd_1"].rolling(30 ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_60"]  = global_data.groupby("gvkey")["volume_usd_1"].rolling(60 ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_90"]  = global_data.groupby("gvkey")["volume_usd_1"].rolling(90 ).mean().reset_index(level=0, drop=True)
global_data["volume_usd_120"] = global_data.groupby("gvkey")["volume_usd_1"].rolling(120).mean().reset_index(level=0, drop=True)
global_data["volume_usd_240"] = global_data.groupby("gvkey")["volume_usd_1"].rolling(240).mean().reset_index(level=0, drop=True)

# ADJUSTED RETURNS
global_data["total_return_factor"] = global_data["total_return_factor"].fillna(1)
global_data["price_adj_return_factor"] = global_data["price_close_adj"] * global_data["total_return_factor"]

global_data["trr_1"] = global_data.groupby(["gvkey"])["price_adj_return_factor"].pct_change(1).reset_index(level=0, drop=True)
global_data.drop(columns=["price_adj_return_factor"], inplace=True)

global_data["trr_1_geom"] = global_data["trr_1"] + 1
global_data["trr_1_geom"].fillna(1, inplace=True)

#ROLLING PRODUCT
global_data["trr_2_geom"]   = global_data.groupby("gvkey")["trr_1_geom"].rolling(2  ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_3_geom"]   = global_data.groupby("gvkey")["trr_1_geom"].rolling(3  ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_5_geom"]   = global_data.groupby("gvkey")["trr_1_geom"].rolling(5  ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_10_geom"]  = global_data.groupby("gvkey")["trr_1_geom"].rolling(10 ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_20_geom"]  = global_data.groupby("gvkey")["trr_1_geom"].rolling(20 ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_30_geom"]  = global_data.groupby("gvkey")["trr_1_geom"].rolling(30 ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_60_geom"]  = global_data.groupby("gvkey")["trr_1_geom"].rolling(60 ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_90_geom"]  = global_data.groupby("gvkey")["trr_1_geom"].rolling(90 ).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_120_geom"] = global_data.groupby("gvkey")["trr_1_geom"].rolling(120).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1
global_data["trr_240_geom"] = global_data.groupby("gvkey")["trr_1_geom"].rolling(240).apply(np.prod, raw=True).reset_index(level=0, drop=True) - 1

#GEOMETRIC MEAN
global_data["trr_2_geom_m"]   = (global_data["trr_2_geom"] + 1).pow(1/2) -1
global_data["trr_3_geom_m"]   = (global_data["trr_3_geom"] + 1).pow(1/3) -1
global_data["trr_5_geom_m"]   = (global_data["trr_5_geom"] + 1).pow(1/5) -1
global_data["trr_10_geom_m"]  = (global_data["trr_10_geom"] + 1).pow(1/10) -1
global_data["trr_20_geom_m"]  = (global_data["trr_20_geom"] + 1).pow(1/20) -1
global_data["trr_30_geom_m"]  = (global_data["trr_30_geom"] + 1).pow(1/30) -1
global_data["trr_60_geom_m"]  = (global_data["trr_60_geom"] + 1).pow(1/60) -1
global_data["trr_90_geom_m"]  = (global_data["trr_90_geom"] + 1).pow(1/90) -1
global_data["trr_120_geom_m"] = (global_data["trr_120_geom"] + 1).pow(1/120) -1
global_data["trr_240_geom_m"] = (global_data["trr_240_geom"] + 1).pow(1/240) -1

global_data["trr_2"]   = global_data["trr_2_geom_m"]
global_data["trr_3"]   = global_data["trr_3_geom_m"]
global_data["trr_5"]   = global_data["trr_5_geom_m"]
global_data["trr_10"]  = global_data["trr_10_geom_m"]
global_data["trr_20"]  = global_data["trr_20_geom_m"]
global_data["trr_30"]  = global_data["trr_30_geom_m"]
global_data["trr_60"]  = global_data["trr_60_geom_m"]
global_data["trr_90"]  = global_data["trr_90_geom_m"]
global_data["trr_120"] = global_data["trr_120_geom_m"]
global_data["trr_240"] = global_data["trr_240_geom_m"]

global_data.drop(columns=["trr_2_geom_m", "trr_3_geom_m", "trr_5_geom_m", "trr_10_geom_m", "trr_20_geom_m", 
                      "trr_30_geom_m", "trr_60_geom_m", "trr_90_geom_m", "trr_120_geom_m", "trr_240_geom_m", 
                      "trr_2_geom", "trr_3_geom", "trr_5_geom", "trr_10_geom", "trr_20_geom", 
                      "trr_30_geom", "trr_60_geom","trr_90_geom","trr_120_geom", "trr_240_geom",
                      "trr_1_geom"
                      ], inplace=True)

global_data["volatility_5"]   = global_data.groupby("gvkey")["trr_1"].rolling(5  ).std().reset_index(level=0, drop=True)
global_data["volatility_10"]  = global_data.groupby("gvkey")["trr_1"].rolling(10 ).std().reset_index(level=0, drop=True)
global_data["volatility_20"]  = global_data.groupby("gvkey")["trr_1"].rolling(20 ).std().reset_index(level=0, drop=True)
global_data["volatility_30"]  = global_data.groupby("gvkey")["trr_1"].rolling(30 ).std().reset_index(level=0, drop=True)
global_data["volatility_60"]  = global_data.groupby("gvkey")["trr_1"].rolling(60 ).std().reset_index(level=0, drop=True)
global_data["volatility_90"]  = global_data.groupby("gvkey")["trr_1"].rolling(90 ).std().reset_index(level=0, drop=True)
global_data["volatility_120"] = global_data.groupby("gvkey")["trr_1"].rolling(120).std().reset_index(level=0, drop=True)
global_data["volatility_240"] = global_data.groupby("gvkey")["trr_1"].rolling(240).std().reset_index(level=0, drop=True)

print("Starting fundamentals")

fundamentals_feature_dict = {
    'gvkey': 'gvkey',
    'datadate': 'date',
    'fyr': 'fiscal_year_end_month',
    'acctstdq': 'accounting_standard_code',
    'bsprq': 'best_practice_code',
    'compstq': 'company_status_code',
    'curcdq': 'currency_code',
    'datacqtr': 'data_change_quarter',
    'datafqtr': 'data_final_change_quarter',
    'fqtr': 'fiscal_quarter',
    'fyearq': 'fiscal_year',
    'rp': 'reporting_period',
    'scfq': 'source_code',
    'srcq': 'source_code_change',
    'staltq': 'statement_type_code',
    'updq': 'update_code',
    'datadate': 'date',
    'fdateq': 'fiscal_date',
    'pdateq': 'period_date'}

# In[ ] LOAD FUNDAMENTALS DATA:
global_fundamentals_data = pd.read_csv('global_fundamentals_1990.zip')


# In[ ] LOAD CHOSEN FUNDAMENTAL FEATURES:
global_fundamentals_features = pd.read_csv('global_fundamental_features.csv', delimiter=';')
global_fundamentals_features = global_fundamentals_features[global_fundamentals_features['wrds_feature'].notnull()]


# In[ ] RENAME:
all_global_fundamental_features = list(fundamentals_feature_dict.keys()) + (global_fundamentals_features['wrds_name'].str.lower().tolist())
all_global_fundamental_feature_new_names = list(fundamentals_feature_dict.values()) + (global_fundamentals_features['wrds_feature'].tolist())
global_fundamentals_data = global_fundamentals_data[all_global_fundamental_features]
global_fundamentals_data.columns = all_global_fundamental_feature_new_names


# In[ ] CALCULATE FUNDAMENTAL FEATURES AND RATIOS:
fundamentals_data = global_fundamentals_data

fundamentals_data['net_income'] = fundamentals_data['income_bex'] + fundamentals_data['extra']
fundamentals_data['net_income'] = fundamentals_data['net_income'].fillna(fundamentals_data['income_bex'])

global_data['date'] = pd.to_datetime(global_data['date'])
fundamentals_data['date'] = pd.to_datetime(fundamentals_data['date'])
fundamentals_data = fundamentals_data.reset_index(drop=True)

# ROLLING 12M
fundamentals_data['net_income_12m'] = fundamentals_data.sort_values(by="date", ascending=True).groupby(['gvkey'])[['net_income', 'date']].rolling(f"{30*11}D", on='date').sum().reset_index().set_index('level_1')['net_income']
fundamentals_data['income_bex_12m'] = fundamentals_data.sort_values(by="date", ascending=True).groupby(['gvkey'])[['income_bex', 'date']].rolling(f"{30*11}D", on='date').sum().reset_index().set_index('level_1')['income_bex']

fundamentals_data['net_debt'] = fundamentals_data['short_debt'] + fundamentals_data['long_debt'] - fundamentals_data['cash_and_eq']
fundamentals_data['current_ratio'] = fundamentals_data['current_assets'] / fundamentals_data['current_liabilites']
fundamentals_data['quick_ratio'] = (fundamentals_data['current_assets'] - fundamentals_data['inventories']) / fundamentals_data['current_liabilites']
fundamentals_data['cash_ratio'] = fundamentals_data['cash_and_eq'] / fundamentals_data['current_liabilites']
fundamentals_data['total_assets_to_liabilites'] = fundamentals_data['total_assets'] / fundamentals_data['total_liabilites']
fundamentals_data['equity_to_debt_ratio'] = fundamentals_data['stockholders_equity'] / fundamentals_data['total_liabilites']
fundamentals_data['interest_coverage_ratio'] = fundamentals_data['op_income_ad'] / fundamentals_data['interest_expense']
fundamentals_data['debt_service_coverage_ratio'] = fundamentals_data['op_income_ad'] / fundamentals_data['short_debt']
fundamentals_data['asset_turnover_ratio'] = fundamentals_data['net_sales'] / fundamentals_data['total_assets']
fundamentals_data['inventory_turnover_ratio'] = fundamentals_data['cost_goods_sold'] / fundamentals_data['inventories']
fundamentals_data['operating_margin_ratio'] = fundamentals_data['op_income_ad'] / fundamentals_data['net_sales']
fundamentals_data['return_on_assets'] = fundamentals_data['net_income_12m'] / fundamentals_data['total_assets']
fundamentals_data['return_on_equity'] = fundamentals_data['net_income_12m'] / fundamentals_data['stockholders_equity']
fundamentals_data['EBITDA'] = fundamentals_data['op_income_ad'] + fundamentals_data['dep_am']
fundamentals_data['EBITDA_to_net_debt'] = fundamentals_data['net_debt'] / fundamentals_data['EBITDA'] #wrong way around, should be inverted
fundamentals_data['EBITDA_to_interest_expense'] = fundamentals_data['EBITDA'] / fundamentals_data['interest_expense']
fundamentals_data['total_assets_to_debt'] = fundamentals_data['total_assets'] / (fundamentals_data['short_debt'] + fundamentals_data['long_debt'])
fundamentals_data['gross_margin'] = (fundamentals_data['net_sales'] - fundamentals_data['cost_goods_sold'])

# In[ ] MERGE FUNDAMENTALS AND GLOBAL DATA:
global_data.reset_index(drop=True, inplace=True)
fundamentals_data.reset_index(drop=True, inplace=True)
fundamentals_data.sort_values(by=['date', 'gvkey'], inplace=True)
global_data.sort_values(by=['date', 'gvkey'], inplace=True)

joined_data = pd.merge_asof(global_data, fundamentals_data, on='date', by='gvkey', allow_exact_matches=False)



# In[ ] TO PRICE RATIOS:
joined_data["earnings_per_share"] = joined_data["net_income"]*1000000 / joined_data["shares_outstanding"]
joined_data["earnings_bex_per_share"] = joined_data["income_bex"]*1000000 / joined_data["shares_outstanding"]
joined_data["earnings_12m_per_share"] = joined_data["net_income_12m"]*1000000 / joined_data["shares_outstanding"]
joined_data["earnings_bex_12m_per_share"] = joined_data["income_bex_12m"]*1000000 / joined_data["shares_outstanding"]
joined_data["book_per_share"] = joined_data["stockholders_equity"] / joined_data["shares_outstanding"] 
joined_data["sales_per_share"] = joined_data["net_sales"] / joined_data["shares_outstanding"]

joined_data["earnings_to_price"] = joined_data["earnings_per_share"] / joined_data["price_close_usd"]
joined_data["earnings_bex_to_price"] = joined_data["earnings_bex_per_share"] / joined_data["price_close_usd"]
joined_data["earnings_12m_to_price"] = joined_data["earnings_12m_per_share"] / joined_data["price_close_usd"]
joined_data["earnings_bex_12m_to_price"] = joined_data["earnings_bex_12m_per_share"] / joined_data["price_close_usd"] 
joined_data["sales_to_price"] = joined_data["sales_per_share"]  / joined_data["price_close_usd"]
joined_data["book_to_price"] =  joined_data["book_per_share"] / joined_data["price_close_usd"]
joined_data["dividend_yield"] = joined_data["dividend_usd"] / joined_data["price_close_usd"]
# ROLLING 12M
joined_data["dividend_yield_12m"] = joined_data.sort_values(by="date", ascending=True).groupby(['gvkey'])[['dividend_yield', 'date']].rolling(f"{30*11}D", on='date').sum().reset_index().set_index('level_1')['dividend_yield']


# In[ ] KEEP ONLY FINAL FEATURES:
final_features_df = pd.read_csv('final_features.csv', delimiter=';')
final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

print("Not in joined_data:")
print(set(final_features) - set(joined_data.columns.tolist()))
final_features = [feature for feature in final_features if feature in joined_data.columns.tolist()]
final_features_df = final_features_df[final_features_df['final_feature'].isin(final_features)]

joined_data = joined_data[final_features]

# In[ ] CONVERT DATA TYPES, ALSO DONE ON IMPORT:
dtype_dict = dict(zip(final_features_df["final_feature"], final_features_df["feature_type"]))
del dtype_dict['date']

for index, feature in final_features_df.iterrows():
    if pd.isnull(feature["final_feature"]) or feature["final_feature"] == "date":
        continue
    if feature["feature_type"] == "string":
        joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(str)
    elif feature["feature_type"] == "int":
        joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(pd.Int64Dtype())
    elif feature["feature_type"] == "float":
        joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(float)
    if feature["categorical"] == "True":
        joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype("category")



print("Writing data")

# In[ ]:
joined_data.to_csv('global_data_1993_to_2004_processed.zip', index=False)

# In[ ]:
joined_data[joined_data["date"].dt.year == 2004].to_parquet('global_data_2004only_processed.parquet', index=False)
# In[ ]:
joined_data[joined_data["date"].dt.year == 2003].to_parquet('global_data_2003only_processed.parquet', index=False)
# In[ ]:
joined_data[joined_data["date"].dt.year == 2002].to_parquet('global_data_2002only_processed.parquet', index=False)
# In[ ]:
joined_data[joined_data["date"].dt.year == 2001].to_parquet('global_data_2001only_processed.parquet', index=False)
# In[ ]:
joined_data[joined_data["date"].dt.year == 2000].to_parquet('global_data_2000only_processed.parquet', index=False)
