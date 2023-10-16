import matplotlib.pyplot as plt
import pandas as pd
import sklearn

load_file = pd.read_csv("data_ml.csv", index_col=[0]) # will probably look like this on your computer
# if you have trouble running then you can delete this path and change the path above using the file on your computer
# and the path that goes with it

for columns in load_file.columns:
    print(columns)  # print all column names

load_headers = load_file.iloc[:5]
print(load_headers)  # print first 5 headers

stats = ["Asset_Turnover", "Interest_Expense", "Eps", "Net_Margin", "Roa", "Roe", "Ta", "Total_Debt", "R1M_Usd"]
statsd = load_file[stats].describe()
print(statsd)  # print descript stats

col1 = ["Ta", "R1M_Usd"]  # Column for scatterplot
col2 = ["Total_Capital", "Total_Debt"]  # Column for scatterplot
plt.figure(figsize=(15, 10))  # scatterplot 1
plt.scatter(load_file[col1[0]], load_file[col1[1]], alpha=0.5)
plt.title(" Ta Vs. R1M_Usd ")
plt.xlabel("Ta")
plt.ylabel("R1M_Usd")
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 10))  # scatterplot 2
plt.scatter(load_file[col1[0]], load_file[col1[1]], alpha=0.5)
plt.title(" Total_Capital Vs. Total_Debt ")
plt.xlabel("Total_Capital")
plt.ylabel("Total_Debt")
plt.grid(True)
plt.show()

sorted_data = load_file.sort_values(by="date", ascending=True)  # sorted the data

load_file["date"] = pd.to_datetime(load_file["date"])

train_date = "01/01/1000"  # set min
train_end = "01/01/2015"
test_date = "01/01/2015"
test_end = "03/31/2019"  # set max

train_data = load_file[(load_file["date"] >= train_date) & (load_file["date"] < train_end)]
test_data = load_file[(load_file["date"] >= test_date) & (load_file["date"] <= test_end)]

train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)



print("Train Data:")
print(train_data)
print("Test Data:")
print(test_data)

stats_train = train_data[stats].describe()
stats_test = test_data[stats].describe()

print(stats_train)
print(stats_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

stats_train_std = scaler.fit_transform(stats_train)

stats_test_std = scaler.transform(stats_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(stats_train_std,stats_test_std)
from sklearn.metrics import mean_squared_error, r2_score

y_train_pred = model.predict(stats_train_std)
y_test_pred = model.predict(stats_test_std)


mse_train = mean_squared_error (y_test_pred,y_train_pred)
r2_train = r2_score(y_test_pred,y_train_pred)
print(f"Training Data: MSE = {mse_train}, R-squared = {r2_train}")

mse_test = mean_squared_error(y_train_pred, y_test_pred)
r2_test = r2_score(y_train_pred, y_test_pred)
print(f"Test Data: MSE = {mse_test}, R-squared = {r2_test}")

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(stats_train_std,stats_train_std)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(tree, filled=True,rounded=True, fontsize=10)
plt.show()

