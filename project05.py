import numpy as np
import pandas

from pandas           import read_csv
from matplotlib       import pyplot
from matplotlib.dates import DateFormatter, WeekdayLocator, drange, RRuleLocator, YEARLY, rrulewrapper, MonthLocator
from sklearn.pipeline import make_pipeline
from sklearn          import preprocessing
from sklearn.metrics  import explained_variance_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model   import LinearRegression
from sklearn.ensemble       import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Load dataset
names = ['Product_Code','Warehouse','Product_Category','Date','Order_Demand']
dataset_HDC = read_csv('./data/HistoricalProductDemand.csv', names=names)

dataset_HDC  = dataset_HDC.dropna()
print(dataset_HDC.head(10))

#Analyze data through visualization.
fig, ax = pyplot.subplots(1)
fig.autofmt_xdate()
pyplot.plot(dataset_HDC['Date'], dataset_HDC['Order_Demand'],'x')
xfmt = DateFormatter('%y')
ax.xaxis.set_major_formatter(xfmt)
rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
loc = RRuleLocator(rule)
ax.xaxis.set_major_locator(MonthLocator())
pyplot.show()

# Analyze data through statistics
print(dataset_HDC.Product_Code.value_counts())
print(dataset_HDC.Warehouse.value_counts())
print(dataset_HDC.Product_Category.value_counts())
print(dataset_HDC.Date.value_counts())

#sort data by using date
dataset_HDC.sort_values(by = 'Date', inplace = True)


#After comparing the differences in the data, we found that the types of Warehouses are less different.
#There is almost no effect on the overall data. So we delete this part.

dataset_HDC.drop('Warehouse', axis = 1)


# Since we found that the data set contains a large number of products, and each product is different, we must analyze different products so that the data obtained will be more effective.
# Taking our statistical data as an example, product 1359 has the largest order demand.
# From this perspective, we can focus our productivity on these products with high demand.
# But for these high-demand products, we don't know if there is a big demand all the time.

product = dataset_HDC[dataset_HDC.Product_Code == 'Product_1359'].copy()
product['DateTime'] = pandas.to_datetime(product.Date)
product.drop('Date', axis = 1)
product.plot(x = 'DateTime', y = 'Order_Demand')
pyplot.show()

# Combine all order in the same day
product = product.groupby(product.DateTime.dt.date).sum()
product.reset_index(inplace = True)
print(product.head(10))

product.plot(x = 'DateTime', y = 'Order_Demand')
pyplot.show()

# Initial Models
linear_regression = LinearRegression()
gradient_boost    = GradientBoostingRegressor(random_state=0)
neural_network    = MLPRegressor()



# create training set and testing set
product['Year'] = product['DateTime'].apply(lambda x: x)
product['Day'] = product['DateTime'].apply(lambda x: x)
product['Month'] = product['DateTime'].apply(lambda x: x)
product['Year'] = product['Year'].apply(lambda x: x.year)
product['Day'] = product['Day'].apply(lambda x: x.day)
product['Month'] = product['Month'].apply(lambda x: x.month)
print(product.head(10))

x = product.drop(['Order_Demand', 'DateTime'], axis = 1)
y = product.Order_Demand.astype('float')

print(x.head(5))
print(y.head(5))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.04)

# Training 3 different models
linear_regression.fit(x_train,y_train)
gradient_boost.fit(x_train,y_train)
neural_network.fit(x_train,y_train)

linear_regression.fit(x_test,y_test)
gradient_boost.fit(x_test,y_test)
neural_network.fit(x_test,y_test)

#Testing 3 different models
prediction1 = linear_regression.predict(x_test)
prediction2 = gradient_boost.predict(x_test)
prediction3 = neural_network.predict(x_test)

# predict
print('LinearRegression',explained_variance_score(y_test, prediction1))
print('GradientBoostingRegressor',explained_variance_score(y_test, prediction2))
print('NeuralNetwork',explained_variance_score(y_test, prediction3))
