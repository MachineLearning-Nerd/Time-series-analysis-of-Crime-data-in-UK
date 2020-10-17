import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

windows = 10
all_data = pd.read_csv('./processed_data.csv')
new_data = []

for i in range(all_data.shape[0]-2*windows):
    temp_data = []
    column_name = []
    for j in range(windows):

        if j != 0 and j != windows-1:
            temp = all_data.iloc[i+j, :]
            temp.pop('Date')
            column_name.extend(temp.index)
            temp = temp.to_numpy()

        elif j == windows - 1:
            temp = all_data.iloc[i+j+windows, :]
            temp = [temp['Date'], temp['Total(Crime)']]
            column_name.append('Total(Date)Target')
            column_name.append('Total(Crime)Target')

        else:
            temp = all_data.iloc[i+j, :]
            column_name.extend(temp.index)
            temp = temp.to_numpy()

        temp_data.extend(temp)
    new_data.append(temp_data)

new_data = pd.DataFrame(new_data)
new_data.columns = column_name

X = new_data.iloc[:, 1:-2].to_numpy()
y = new_data.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

if os.path.isfile('model.sav')==False:
    regr = RandomForestRegressor(n_estimators=1000, random_state=42)
    regr.fit(X_train, y_train)
else:
    regr = pickle.load(open('model.sav', 'rb'))

pickle.dump(regr, open('model.sav', 'wb'))
y_pred = regr.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(sum(np.abs(y_pred - y_test))/len(y_pred))

y_pred = regr.predict(X)
new_data['Prediction Value'] = y_pred
new_data.to_csv('temp.csv', index=False)
labels = new_data.Date.to_numpy()
plt.plot(labels, y)
plt.plot(labels, y_pred)
locs, labels = plt.xticks()
labels = new_data.Date.to_numpy()
plt.xticks(locs[::100], labels[::100])  # Set label locations.
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Crimes')
plt.legend(['Original', 'Prediction'])
plt.tight_layout()
os.makedirs('Images', exist_ok=True)
plt.savefig('Images/Prediction_vs_original.png', dpi=300)
plt.show()
print(mean_squared_error(y, y_pred))
print(sum(np.abs(y - y_pred))/len(y_pred))
