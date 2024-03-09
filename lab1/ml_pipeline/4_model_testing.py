import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error

test_data_path = 'test/temperature_test_preprocessed.csv'
test_data = pd.read_csv(test_data_path)

model_filename = 'trained_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

X_test = test_data.drop('Temperature', axis=1)
y_test = test_data['Temperature']

test_predictions = model.predict(X_test)

mse_test = mean_squared_error(y_test, test_predictions)
print(f'Mean Squared Error on Test Data: {mse_test}')
