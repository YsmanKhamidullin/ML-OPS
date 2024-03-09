import pandas as pd
from sklearn.preprocessing import StandardScaler

train_data_path = 'train/temperature_train.csv'
train_data = pd.read_csv(train_data_path)

test_data_path = 'test/temperature_test.csv'
test_data = pd.read_csv(test_data_path)


def preprocess_date_columns(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data = data.drop(columns=['Date'])

    return data


train_data = preprocess_date_columns(train_data)
test_data = preprocess_date_columns(test_data)


def preprocess_data(data):
    numeric_features = data.select_dtypes(include=['float64']).columns
    scaler = StandardScaler()
    data[numeric_features] = scaler.fit_transform(data[numeric_features])

    return data


train_data_preprocessed = preprocess_data(train_data)
train_data_preprocessed.to_csv('train/temperature_train_preprocessed.csv', index=False)

test_data_preprocessed = preprocess_data(test_data)
test_data_preprocessed.to_csv('test/temperature_test_preprocessed.csv', index=False)
