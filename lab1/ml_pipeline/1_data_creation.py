import os
import numpy as np
import pandas as pd

if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')


def generate_data(size, anomaly_prob=0.05, noise_std=2):
    dates = pd.date_range('2022-01-01', periods=size, freq='D')
    temperatures = np.random.normal(25, 5, size)

    anomaly_indices = np.random.choice(size, int(anomaly_prob * size), replace=False)
    temperatures[anomaly_indices] += np.random.uniform(-10, 10, len(anomaly_indices))

    temperatures += np.random.normal(0, noise_std, size)

    return pd.DataFrame({'Date': dates, 'Temperature': temperatures})


train_data = generate_data(1000)
train_data.to_csv('train/temperature_train.csv', index=False)

test_data = generate_data(300)
test_data.to_csv('test/temperature_test.csv', index=False)
