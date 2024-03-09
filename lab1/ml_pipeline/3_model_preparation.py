import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_data_path = 'train/temperature_train_preprocessed.csv'
train_data = pd.read_csv(train_data_path)

X = train_data.drop(columns=['Temperature'])
y = train_data['Temperature']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

mse = mean_squared_error(y_valid, y_pred)
print(f'Mean Squared Error on Validation Set: {mse}')

model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

print(f'Model saved to {model_filename}')
