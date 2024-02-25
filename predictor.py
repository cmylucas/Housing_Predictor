import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pickle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
      
#Read and plot
df = pd.read_csv("/kaggle/input/housedata/housingdata2.csv", header = 0)
df = df.drop("neighbourhood", axis=1)
df = df.dropna()
df = df.drop("pool",axis = 1)
independents = df.columns[1:]
for column in independents:
    
    if column == "coordinate": continue
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column], df['price'], alpha=0.5)  
    plt.title(f'{column} vs price')  
    plt.xlabel(column) 
    plt.ylabel('Price') 
    plt.grid(True) 
    plt.show()

    print(f'r for {column} vs price = {df["price"].corr(df[column])}')
    
    correlation_matrix = df[[df.columns[0],column]].corr()

#Removing outliers
k = 1
columns_to_check = ['price','bedroom', 'bathroom', 'sqft'] 
df_no_outliers = pd.DataFrame()
for column in columns_to_check:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    df = df[~outliers]
print(df)

#K-means clustering
df['coordinate'] = df['coordinate'].apply(lambda x: tuple(map(float, x.strip('()').split(','))))
coordinates = pd.DataFrame(df['coordinate'].tolist(), columns=['x', 'y'])
print(coordinates)
scaler = StandardScaler()
coordinates_scaled = scaler.fit_transform(coordinates)

k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster_label'] = kmeans.fit_predict(coordinates_scaled)

plt.scatter(coordinates_scaled[:, 0], coordinates_scaled[:, 1], c=df['cluster_label'], cmap='viridis')
plt.xlabel('Standardized Coordinate X')
plt.ylabel('Standardized Coordinate Y')
plt.title('K-Means Clustering of Houses')
plt.show()

avg=[]
for x in range(0,10):
    avg.append([0,x])
cnt=[0]*10
for x in df.index:
    code = df["cluster_label"][x]
    p = df["price"][x]
    avg[code][0] += p
    cnt[code] += 1
for x in range(0,9):
    avg[x][0] = avg[x][0]/cnt[x]
avg.sort()
print(avg)
for x in range(10):
    df['cluster_label'].replace(avg[x][1], x+10, inplace=True)

plt.figure(figsize=(8, 6))
plt.scatter(df['cluster_label'], df['price'], alpha=0.5)  
plt.title(f'region vs price')  
plt.xlabel('Region') 
plt.ylabel('Price') 
plt.grid(True) 
plt.show()

#Polynomial regression
X = df[['sqft', 'bedroom', 'bathroom','cluster_label']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

poly_transformer = PolynomialFeatures(degree=2)

X_train_poly = poly_transformer.fit_transform(X_train)
X_test_poly = poly_transformer.transform(X_test)

model1 = LinearRegression()

model1.fit(X_train_poly, y_train)

modlist = [poly_transformer, model1]
with open('poly.pkl', 'wb') as f:
    pickle.dump(modlist,f)

y_pred = model1.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r_squared}")

def predict(bedroom, bathroom, sqft, age, label):
    mlist = []
    with open('poly.pkl', 'rb') as f:
        mlist = pickle.load(f)
    polytransformer = mlist[0]
    model = mlist[1]
    data = np.array([[int(sqft), int(bedroom),bathroom,int(label)]])
    dataframe = pd.DataFrame(data,columns = ['sqft','bedroom','bathroom','cluster_label'])
    dataframe = polytransformer.transform(dataframe)
    y_pred = model.predict(dataframe)
    return y_pred

testnum = 3
test = X.iloc[testnum]
print(f"predicion: {predict(test.loc['bedroom'],test.loc['bathroom'],test.loc['sqft'],None,test.loc['cluster_label'])}")
print(y.iloc[testnum])
