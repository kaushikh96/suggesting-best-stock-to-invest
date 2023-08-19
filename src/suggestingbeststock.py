#imports
import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
import yfinance as yf
from datetime import datetime
from plotly.subplots import make_subplots
import matplotlib.pyplot as mlplt
import plotly.graph_objects as plt
from itertools import cycle
import plotly.express as pltX
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
import matplotlib.pyplot as pyplot
import seaborn as sns
from numpy import array

mlplt.style.use("fivethirtyeight")
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
sns.set_style('whitegrid')

# fetching data from yahoo finance
def fetch_data():
  companies_list = ['TSLA', 'GOOG', 'META', 'AAPL']
  end_time = datetime.now()
  start_time = datetime(end_time.year - 1, end_time.month, end_time.day)
  for stock in companies_list:
    globals()[stock] = yf.download(stock, start_time, end_time)
  
  companies_list = [TSLA, GOOG, META, AAPL]
  TSLA.describe()
  companies_name = ["TESLA", "GOOGLE", "META", "APPLE"]

  for company, com_name in zip(companies_list, companies_name):
      company["company_name"] = com_name
  input_df = pd.concat(companies_list, axis=0)
  input_df = input_df.reset_index();
  input_df = input_df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                  'Closing Price':'adj_close','Volume':'volume'})
  input_df['date'] = pd.to_datetime(input_df.date)
  return input_df, company, companies_list, companies_name

def fetch_data_single_company(company_name, start_time):
  input_df = DataReader(company_name , data_source='yahoo', start=start_time, end=datetime.now())
  input_df = input_df.reset_index();
  input_df = input_df.rename(columns={'Date': 'date','Open':'open','High':'high','Low':'low','Close':'close',
                                  'Closing Price':'adj_close','Volume':'volume'})
  input_df['date'] = pd.to_datetime(input_df.date)
  return input_df

def analyze_data(input_df):
  analyzed_data = dict()
  analyzed_data['data_set_size'] = input_df.shape[0]
  analyzed_data['null_values'] = input_df.isnull().values.sum()
  analyzed_data['na_values'] = input_df.isna().values.any()
  analyzed_data['start_date'] = input_df.iloc[0][0]
  analyzed_data['end_date'] = input_df.iloc[-1][0]
  analyzed_data['Total_duration_for_given_dataset'] = input_df.iloc[-1][0] - input_df.iloc[0][0]
  result_df = pd.DataFrame([analyzed_data])
  return result_df

# evaluating the dataset
def evaluate_dataset(input_df):
  evaluate = dict()
  month_wise= input_df.groupby(input_df['date'].dt.strftime('%B'))[['open','close']].mean()
  new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
              'September', 'October', 'November', 'December']
  month_wise = month_wise.reindex(new_order, axis=0)
  evaluate['month_wise'] = month_wise
  month_wise_h = input_df.groupby(input_df['date'].dt.strftime('%B'))['high'].max()
  monthvise_high = month_wise_h.reindex(new_order, axis=0)
  month_wise_l = input_df.groupby(input_df['date'].dt.strftime('%B'))['low'].min()
  month_wise_l = month_wise_l.reindex(new_order, axis=0)
  evaluate['month_wise_high'] = month_wise_h
  evaluate['month_wise_low'] = month_wise_l
  #result_df = pd.DataFrame(evaluate)
  return evaluate

def plot_OC(evaluate_dict):
  plot = plt.Figure()
  plot.add_trace(plt.Bar(
      x=evaluate_dict['month_wise'].index,
      y=evaluate_dict['month_wise']['open'],
      name='Opening price of stock',
      marker_color='rgb(26, 43, 238)'
  ))
  plot.add_trace(plt.Bar(
      x=evaluate_dict['month_wise'].index,
      y=evaluate_dict['month_wise']['close'],
      name='Closing price of stock',
      marker_color='rgb(26, 238, 99)'
  ))

  plot.update_layout(barmode='group', 
                    title='stock opening and closing price with monthly intrevals')
  plot.show()

def plot_HL(evaluate_dict):
  plot = plt.Figure()
  plot.add_trace(plt.Bar(
      x=evaluate_dict['month_wise_high'].index,
      y=evaluate_dict['month_wise_high'],
      name='Stock high Price',
      marker_color='rgb(26, 43, 238)'
  ))
  plot.add_trace(plt.Bar(
      x=evaluate_dict['month_wise_low'].index,
      y=evaluate_dict['month_wise_low'],
      name='Stock low Price',
      marker_color='rgb(26, 238, 99)'
  ))

  plot.update_layout(barmode='group', 
                    title=' Monthly High and Low stock price')
  plot.show()

def plot_stock_chart(input_df):
  variables = cycle(['Stock Open Price','Stock Close Price','Stock High Price','Stock Low Price'])

  plot = pltX.line(input_df, x=input_df.date, y=[input_df['open'], input_df['close'], 
                                            input_df['high'], input_df['low']],
              labels={'date': 'Date','value':'Stock value'})
  plot.update_layout(title_text='Stock chart',legend_title_text='Stock Params')
  plot.for_each_trace(lambda t:  t.update(name = next(variables)))
  plot.update_xaxes(showgrid=False)
  plot.update_yaxes(showgrid=False)
  plot.show()

def check_correlation_between_features(input_df):
  df = input_df.copy()
  del df['']

data, company_name, companies_list, companies_name = fetch_data()

data.tail(10)

print(company_name)

def plot_cl_price():
  mlplt.figure(figsize=(30, 30))
  mlplt.subplots_adjust(top=1.25, bottom=1.2)

  company__name_list = ['TSLA', 'GOOG', 'META', 'AAPL']

  mlplt.subplot(4, 1, 1)
  TSLA['Adj Close'].plot(color='orange')
  mlplt.ylabel('Closing Price')
  mlplt.xlabel(None)
  mlplt.title(f"Closing Price of TSLA")

  mlplt.subplot(4, 1, 2)
  GOOG['Adj Close'].plot()
  mlplt.ylabel('Closing Price')
  mlplt.xlabel(None)
  mlplt.title(f"Closing Price of GOOG")

  mlplt.subplot(4, 1, 3)
  META['Adj Close'].plot(color='indigo')
  mlplt.ylabel('Closing Price')
  mlplt.xlabel(None)
  mlplt.title(f"Closing Price of META")

  mlplt.subplot(4, 1, 4)
  AAPL['Adj Close'].plot(color='green')
  mlplt.ylabel('Closing Price')
  mlplt.xlabel(None)
  mlplt.title(f"Closing Price of AAPL")
      
  mlplt.tight_layout()
plot_cl_price()

def check_correlation_between_features(input_df):
  df = input_df.copy()
  colors = ['#000000','#ADD8E6']
  del df['date']
  fig=mlplt.figure(figsize=(15,8))
  sns.heatmap(df.corr(), annot=True, cmap=[colors[0],colors[1]], linecolor='white', linewidth=2 )

stock_data = fetch_data_single_company(company_name='TSLA', start_time='2021-01-01')
stock_data_apple = fetch_data_single_company(company_name='AAPL', start_time='2021-01-01')
stock_data_meta = fetch_data_single_company(company_name='META', start_time='2021-01-01')
stock_data_google = fetch_data_single_company(company_name='GOOG', start_time='2021-01-01')
analyzed_df = analyze_data(stock_data)
data_insights = evaluate_dataset(stock_data)
plot_OC(data_insights)
plot_HL(data_insights)
plot_stock_chart(stock_data)
check_correlation_between_features(stock_data)

def show_ma_chart(input_frame, company_name):
    moving_average_A= 75
    moving_average_B= 100
    input_frame.head()
    moving_average_1 = pd.Series.rolling(input_frame['close'], window=moving_average_A).mean()
    moving_average_2 = pd.Series.rolling(input_frame['close'], window=moving_average_B).mean()
    mlplt.figure(figsize=(15,5))
    mlplt.style.use('ggplot')
    date_range = [x for x in input_frame.date]
    mlplt.plot(date_range, input_frame['close'], lw=1, color="black", label="Price")
    mlplt.plot(date_range, moving_average_1,lw=3,linestyle="dotted", label="Moving Average {} days".format(moving_average_A))
    mlplt.plot(date_range, moving_average_2,lw=3,linestyle="dotted", label=" Moving Average {} days".format(moving_average_B))
    mlplt.legend(loc='best')
    mlplt.title("Moving average of " + company_name)
    mlplt.xlabel('Date')
    mlplt.ylabel('Price')
    xmin = input_frame.index[0]
    xmax = input_frame.index[-1]
    mlplt.show()

show_ma_chart(stock_data_apple, "Apple")
show_ma_chart(stock_data, "Tesla")
show_ma_chart(stock_data_meta, "Meta")
show_ma_chart(stock_data_google, "Google")

def polt_v(companies_list_iterator):
  for company_name in companies_list_iterator:
      company_name['Daily Return'] = company_name['Adj Close'].pct_change()

  fig, axes = mlplt.subplots(nrows=4, ncols=1)
  fig.set_figheight(30)
  fig.set_figwidth(30)

  TSLA['Daily Return'].plot(ax=axes[0], legend=True, linestyle='--', marker='o', color='orange')
  axes[0].set_title('TESLA')

  GOOG['Daily Return'].plot(ax=axes[1], legend=True, linestyle='--', marker='o')
  axes[1].set_title('GOOGLE')

  META['Daily Return'].plot(ax=axes[2], legend=True, linestyle='--', marker='o', color='indigo')
  axes[2].set_title('META')

  AAPL['Daily Return'].plot(ax=axes[3], legend=True, linestyle='--', marker='o', color='green')
  axes[3].set_title('APPLE')

  fig.tight_layout()

polt_v(companies_list)

def plot_dl(companies_list_enumerator):
  mlplt.figure(figsize=(12, 7))
  print(companies_name)

  for i, company_name in enumerate(companies_list, 1):
      mlplt.subplot(2, 2, i)
      company_name['Daily Return'].hist(bins=50)
      mlplt.ylabel('Daily Return')
      mlplt.title(f'{companies_name[i - 1]}')
      
  mlplt.tight_layout()

plot_dl(companies_list)

comapnies_name_list = ['TSLA', 'GOOG', 'META', 'AAPL']
closing_df = DataReader(comapnies_name_list, 'yahoo', start, end)['Adj Close']
closing_df.head()

companies_info = closing_df.pct_change()
companies_info.head()

sns.jointplot(x='GOOG', y='AAPL', data=companies_info, kind='reg', color='red')

sns.jointplot(x='GOOG', y='TSLA', data=companies_info, kind='reg', color='red')

sns.jointplot(x='AAPL', y='TSLA', data=companies_info, kind='reg', color='red')

sns.jointplot(x='GOOG', y='META', data=companies_info, kind='reg', color='red')

companies_info.head()

risk_a = companies_info.dropna()

area = np.pi * 20

mlplt.figure(figsize=(10, 7))
mlplt.scatter(risk_a.mean(), risk_a.std(), s=area)
mlplt.xlabel('Expected return')
mlplt.ylabel('Risk')

for label, x, y in zip(risk_a.columns, risk_a.mean(), risk_a.std()):
    mlplt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

"""Looking at the graph we can identify that Microsoft has the lowest risk involved when we compare this with the other stocks. We will feed this as another parameter to our model. """

def transform_data(input,time_step):
  data_time_series = []
  data_time_series_Y = []
  for i in range(time_step, len(input)):
    data_time_series.append(input[i-time_step:i, 0])
    data_time_series_Y.append(input[i, 0])
  trasformed_X, transformed_Y = np.array(data_time_series), np.array(data_time_series_Y)
  trasformed_X = np.reshape(trasformed_X, (trasformed_X.shape[0], trasformed_X.shape[1], 1))
  return (trasformed_X, transformed_Y)

def generate_dataset(stock_data,scaler,training_split,time_step):

  stock_data_copy = stock_data.copy()
  del stock_data_copy['date']
  X = stock_data_copy[['open','high']]
  Y = stock_data_copy.filter(['close'])
  training_data_len = int(np.ceil( len(stock_data_copy) * training_split))
  
  X_training_set = X.iloc[:int(training_data_len)].values
  X_testing_set = X.iloc[int(training_data_len):].values
  X_train_scaled = scaler.fit_transform(X_training_set)
  X_test_scaled = scaler.transform(X_testing_set)
  
  X_train, y_train = transform_data(X_train_scaled,time_step)
  X_test, y_test = transform_data(X_test_scaled,time_step)
  return (X_train, y_train,X_test, y_test)

scaler=MinMaxScaler(feature_range=(0,1))
training_split = 0.7
time_step = 30
X_train, Y_train, X_test, Y_test = generate_dataset(stock_data,scaler,training_split,time_step)
X_train_google, Y_train_google, X_test_google, Y_test_google = generate_dataset(stock_data_google,scaler,training_split,time_step)
X_train_apple, Y_train_apple, X_test_apple, Y_test_apple = generate_dataset(stock_data_apple,scaler,training_split,time_step)
X_train_meta, Y_train_meta, X_test_meta, Y_test_meta = generate_dataset(stock_data_meta,scaler,training_split,time_step)

mod = dict()

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, Y_train, validation_data=(X_test,Y_test),epochs = 100, batch_size = 20)

model_google = Sequential()
model_google.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model_google.add(Dropout(0.2))
model_google.add(LSTM(units = 50, return_sequences = True))
model_google.add(Dropout(0.2))
model_google.add(LSTM(units = 50, return_sequences = True))
model_google.add(Dropout(0.2))
model_google.add(LSTM(units = 50))
model_google.add(Dropout(0.2))
model_google.add(Dense(units = 1))
model_google.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_google.fit(X_train_google, Y_train_google, validation_data=(X_test_google,Y_test_google),epochs = 100, batch_size = 20)

model_apple = Sequential()
model_apple.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model_apple.add(Dropout(0.2))
model_apple.add(LSTM(units = 50, return_sequences = True))
model_apple.add(Dropout(0.2))
model_apple.add(LSTM(units = 50, return_sequences = True))
model_apple.add(Dropout(0.2))
model_apple.add(LSTM(units = 50))
model_apple.add(Dropout(0.2))
model_apple.add(Dense(units = 1))
model_apple.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_apple.fit(X_train_apple, Y_train_apple, validation_data=(X_test_apple,Y_test_apple),epochs = 100, batch_size = 20)

model_meta = Sequential()
model_meta.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model_meta.add(Dropout(0.2))
model_meta.add(LSTM(units = 50, return_sequences = True))
model_meta.add(Dropout(0.2))
model_meta.add(LSTM(units = 50, return_sequences = True))
model_meta.add(Dropout(0.2))
model_meta.add(LSTM(units = 50))
model_meta.add(Dropout(0.2))
model_meta.add(Dense(units = 1))
model_meta.compile(optimizer = 'adam', loss = 'mean_squared_error')
model_meta.fit(X_train_meta, Y_train_meta, validation_data=(X_test_meta,Y_test_meta),epochs = 100, batch_size = 20)

# Model validation
x_test, y_test = transform_data(X_test,time_step)
predictions = model.predict(x_test)
print(x_test.shape)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

print(rmse)

x_test_google, y_test_google = transform_data(X_test_google,time_step)
predictions_google = model.predict(x_test_google)
print(x_test_google.shape)
rmse = np.sqrt(np.mean(((predictions_google - y_test_google) ** 2)))

print(rmse)

x_test_apple, y_test_apple = transform_data(X_test_apple,time_step)
predictions_apple = model.predict(x_test_apple)
print(x_test_apple.shape)
rmse = np.sqrt(np.mean(((predictions_apple - y_test_apple) ** 2)))

print(rmse)

x_test_meta, y_test_meta = transform_data(X_test_meta,time_step)
predictions_meta = model.predict(x_test_meta)
print(x_test_meta.shape)
rmse = np.sqrt(np.mean(((predictions_meta - y_test_meta) ** 2)))

print(rmse)

# Performanca validation
print("Train data R2 score:", r2_score(y_test.reshape(-1), predictions.reshape(-1)))

# Performanca validation
print("Train data R2 score:", r2_score(y_test_apple.reshape(-1), predictions_apple.reshape(-1)))

print("Train data R2 score:", r2_score(y_test_google.reshape(-1), predictions_google.reshape(-1)))

print("Train data R2 score:", r2_score(y_test_meta.reshape(-1), predictions_meta.reshape(-1)))

"""**Plot between test and predicted values**"""

def plot_test_prediction_chart(y_test, predictions, company_name):
  validations=pd.DataFrame(columns=['test','predictions'])
  validations.reset_index()
  validations['test'] = y_test.reshape(-1)
  validations['predictions'] = predictions.reshape(-1)
  plot=mlplt.figure()
  mlplt.title("Test vs Predicted value of " + company_name , size=20, weight='bold')
  mlplt.plot(validations)
  mlplt.legend(['test','predictions'])
  r2=np.round(r2_score(y_test,predictions),2)
  mse=np.round(mean_squared_error(y_test,predictions),2)
  mae=np.round(mean_squared_error(y_test,predictions),2)

plot_test_prediction_chart(y_test, predictions, "Tesla")
plot_test_prediction_chart(y_test_apple, predictions_apple, "Apple")
plot_test_prediction_chart(y_test_google, predictions_google, "Google")
plot_test_prediction_chart(y_test_meta, predictions_meta, "Meta")

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

def generate_future_data(model, x_test, y_test, future_days, model_type):
  x_input=x_test[len(x_test)-future_days:]
  future_output=[]
  n_steps=time_step
  i=0
  pred_days = future_days
  if model_type == 'svr':
    x_input = x_input.reshape(x_input.shape[0],-1)
  while(i<pred_days): 
    print(x_input.shape)
    pred = model.predict(x_input[i:i+1])  
    future_output.extend(pred.tolist())
    i=i+1
  return future_output

def plot_comparision_for_future(model, x_test, y_test, time_step, predict_future, scaler, company_name, model_type):
  future_output = generate_future_data(model, x_test,y_test,predict_future, model_type)
  last_days=np.arange(1,time_step+1)
  day_pred=np.arange(time_step+1,time_step+predict_future+1)

  temp_mat1 = np.empty((len(last_days)+predict_future+1,1))
  temp_mat1[:] = np.nan
  temp_mat1 = temp_mat1.reshape(1,-1).tolist()[0]
  temp_mat2 = np.empty((len(last_days)+predict_future+1,1))
  temp_mat2[:] = np.nan
  temp_mat2 = temp_mat2.reshape(1,-1).tolist()[0]
  last_original_days_value = temp_mat1
  next_predicted_days_value = temp_mat2

  last_original_days_value[0:time_step+1] = y_test[y_test.shape[0] - time_step - predict_future: y_test.shape[0] - predict_future].reshape(1,-1).tolist()[0]
  next_predicted_days_value[time_step:] = np.array(future_output).reshape(-1,1).reshape(1,-1).tolist()[0]

  new_pred_plot = pd.DataFrame({
      'last_original_days_value':last_original_days_value,
      'next_predicted_days_value':next_predicted_days_value
  })

  names = cycle(['closing price of previous 30 days','Predicted next 6 days close price'])

  fig = pltX.line(new_pred_plot,x=new_pred_plot.index, y=[new_pred_plot['last_original_days_value'],
                                                        new_pred_plot['next_predicted_days_value']],
                labels={'value': 'Stock price','index': 'Timestamp'})
  fig.update_layout(title_text='Compare last 30 days vs next 6 days of ' + company_name,
                    plot_bgcolor='white', font_size=15, font_color='black',legend_title_text='Close Price')
  fig.for_each_trace(lambda t:  t.update(name = next(names)))
  fig.update_xaxes(showgrid=False)
  fig.update_yaxes(showgrid=False)
  fig.show()

predict_future = 6
plot_comparision_for_future(model,x_test=x_test,y_test=y_test,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="Tesla",model_type="lstm")
plot_comparision_for_future(model_google,x_test=x_test_apple,y_test=y_test_apple,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="Google",model_type="lstm")
plot_comparision_for_future(model_apple,x_test=x_test_apple,y_test=y_test_apple,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="apple",model_type="lstm")
plot_comparision_for_future(model_meta,x_test=x_test_meta,y_test=y_test_meta,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="meta",model_type="lstm")

"""**SVR**"""

from sklearn.svm import SVR
X_svr_train = X_train.reshape(X_train.shape[0],-1)
svr_model = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
svr_model.fit(X_svr_train, Y_train)

X_svr_train_google = X_train_google.reshape(X_train.shape[0],-1)
svr_model_google = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
svr_model_google.fit(X_svr_train_google, Y_train_google)

X_svr_train_apple = X_train_apple.reshape(X_train.shape[0],-1)
svr_model_apple = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
svr_model_apple.fit(X_svr_train_apple, Y_train_apple)

X_svr_train_meta = X_train_meta.reshape(X_train.shape[0],-1)
svr_model_meta = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
svr_model_meta.fit(X_svr_train_meta, Y_train_meta)

# Model validation

x_test, y_test = transform_data(X_test,time_step)
X_svr_test = x_test.reshape(x_test.shape[0],-1)
predictions = svr_model.predict(X_svr_test)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print('rmse for tesla',rmse)

x_test_google, y_test_google = transform_data(X_test_google,time_step)
X_svr_test_google = x_test_google.reshape(x_test.shape[0],-1)
predictions_google = svr_model_google.predict(X_svr_test_google)
rmse_google = np.sqrt(np.mean(((predictions_google - y_test_google) ** 2)))
print('rmse for google',rmse_google)

x_test_apple, y_test_apple = transform_data(X_test_apple,time_step)
X_svr_test_apple = x_test_apple.reshape(x_test.shape[0],-1)
predictions_apple = svr_model_apple.predict(X_svr_test_apple)
rmse_apple = np.sqrt(np.mean(((predictions_apple - y_test_apple) ** 2)))
print('rmse for apple',rmse_apple)

x_test_meta, y_test_meta = transform_data(X_test_meta,time_step)
X_svr_test_meta = x_test_meta.reshape(x_test_meta.shape[0],-1)
predictions_meta = svr_model_meta.predict(X_svr_test_meta)
rmse_meta = np.sqrt(np.mean(((predictions_meta - y_test_meta) ** 2)))
print('rmse for meta',rmse_meta)

# Performanca validation
print("Train data R2 score TESLA:", r2_score(y_test.reshape(-1), predictions.reshape(-1)))
print("Train data R2 score GOOGLE:", r2_score(y_test_google.reshape(-1), predictions_google.reshape(-1)))
print("Train data R2 score APPLE:", r2_score(y_test_apple.reshape(-1), predictions_apple.reshape(-1)))
print("Train data R2 score META:", r2_score(y_test_meta.reshape(-1), predictions_meta.reshape(-1)))

plot_test_prediction_chart(y_test, predictions, "Tesla")
plot_test_prediction_chart(y_test_apple, predictions_apple, "Apple")
plot_test_prediction_chart(y_test_google, predictions_google, "Google")
plot_test_prediction_chart(y_test_meta, predictions_meta, "Meta")

predict_future = 6
plot_comparision_for_future(svr_model,x_test=x_test,y_test=y_test,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="Tesla",model_type="svr")
plot_comparision_for_future(svr_model_google,x_test=x_test_apple,y_test=y_test_apple,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="Google",model_type="svr")
plot_comparision_for_future(svr_model_apple,x_test=x_test_apple,y_test=y_test_apple,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="apple",model_type="svr")
plot_comparision_for_future(svr_model_meta,x_test=x_test_meta,y_test=y_test_meta,time_step=time_step,predict_future=predict_future,scaler = scaler, company_name="meta",model_type="svr")