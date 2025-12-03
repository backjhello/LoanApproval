import pandas as pd

def compute_age(df):
    df['dob'] = pd.to_datetime(df['dob'])
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    return df

def make_age_groups(df):
    bins = [0, 25, 35, 50, 65, df['age'].max()]
    labels = ['<25', '25-34', '35-49', '50-64', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df

def make_city_groups(df):
    df['city_group'] = pd.qcut(df['city_pop'], q=4, labels=['Small','Medium','Large','Very Large'])
    return df

def feature_engineering(df):
    customer = df.groupby('customer_id').agg({
        'amt':'sum',
        'spending_std':'median',
        'avg_transaction':'mean'
    }).reset_index()
    return df, customer
