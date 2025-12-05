import pandas as pd
from src.preprocessing import compute_age, make_age_groups, make_city_groups, feature_engineering, prepare_model_dataset


def test_feature_engineering_and_prepare():
    df = pd.DataFrame({
        'customer_id': [1, 1, 2, 2],
        'amt': [10, 20, 30, 40],
        'spending_std': [1, 2, 3, 4],
        'avg_transaction': [5, 6, 7, 8],
        'dob': ['1990-01-01', '1990-01-01', '1980-05-01', '1980-05-01'],
        'trans_date_trans_time': ['2022-01-01', '2022-02-01', '2022-01-01', '2022-02-01'],
        'city_pop': [1000, 1000, 5000, 5000]
    })

    out, customer = feature_engineering(df)
    assert 'total_spent' in customer.columns

    cdf = compute_age(df)
    assert 'age' in cdf.columns

    ag = make_age_groups(cdf)
    assert 'age_group' in ag.columns

    cg = make_city_groups(cdf)
    assert 'city_group' in cg.columns

    # add simple binary target for testing prepare_model_dataset
    customer['high_spender'] = (customer['total_spent'] > customer['total_spent'].median()).astype(int)
    X, y = prepare_model_dataset(customer, feature_columns=['avg_transaction', 'spending_std', 'total_spent'], target_col='high_spender')
    assert X.shape[0] == customer.shape[0]
    assert y.shape[0] == customer.shape[0]
