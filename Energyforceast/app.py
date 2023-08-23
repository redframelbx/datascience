#deploy model_with_holidays into streamlit with user input number of days to be forecast
from prophet.serialize import model_to_json, model_from_json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
import datashader as ds

with open('/mount/src/datascience/Energyforceast/serialized_model.json', 'r') as fin:
     model_with_holidays = model_from_json(fin.read())  # Load model

st.set_page_config(page_title='Prophet Forecasting Network',
                   page_icon=':battery::battery::battery:',
                   layout='wide')

st.title('Forecasting Energy consumption using Prophet Forecasting Network')
st.write('This app forecasts power use in megawatts for the next 365 days or by user input number of days to be forecast')

#user input for number of days to be forecast
# days = st.slider(int, 'How many days would you like to forecast?',1,365,30)
days = st.slider('How many days would you like to forecast?', 1, 365, 365)
future = model_with_holidays.make_future_dataframe(periods=days, freq='d', include_history=False)
forecast = model_with_holidays.predict(future)
# st.write(forecast)

#plot the forecast

d1 = st.date_input("view Energy consumption FROM when?", datetime.date(2013, 1, 1))
d2 = st.date_input("view Energy consumption UNTILL when?", datetime.date(2016,1, 1))
fig, ax = plt.subplots(figsize=(14, 8))
fig1 = model_with_holidays.plot(forecast, ax=ax)
ax.set_title('Forecasted power use mw')
ax.set_xlabel('Date')
# ax.set_xlim([datetime.date(2013, 1, 1),
#                 datetime.date(2015, 12, 20)],
#             )
ax.set_xlim([d1, d2])
ax.set_ylabel('power use mw')
# plt.show()

fig, ax = plt.subplots(figsize=(14, 8))
fig2 = model_with_holidays.plot(forecast, ax=ax)
ax.set_title('Forecasted power use mw')
ax.set_xlabel('Date')
ax.set_ylabel('power use mw')
# ax.set_xbound([datetime.date(201, 1, 1),
#                 datetime.date(2015, 2, 1)],
#               )
ax.set_xlim([datetime.date(2014, 12, 1),
                datetime.date(2015, 1, 31)],
            )
ax.set_ylim(0, 60000)
plot = plt.suptitle('Forecast for Jan 2015')

#plot the components
fig3 = model_with_holidays.plot_components(forecast)
# plt.show()
fig4 = go.Figure(go.Scattergl(x=forecast['ds'][::1],
                             y=forecast['yhat'][::1],
                             mode='markers')
)
fig4.update_layout(title_text='Forecasted power use mw')
fig.show()

st.plotly_chart(fig4,use_container_width=True)
st.title('Annual Energy Consumption with Predicted Consumption')
st.pyplot(fig1,use_container_width=True)
st.title('Previous month with Predicted Consumption')
st.pyplot(fig2,use_container_width=True)
st.title('Predicted Consumption BY Weekly, Daily & Hour')
st.pyplot(fig3,use_container_width=True)

#display the forecast data group by date and display only the yhat column only
# st.write(forecast.groupby(forecast['ds'].dt.date)['yhat'].sum())
# byhour = forecast.groupby(forecast['ds'].dt.date)['yhat'].sum()
# fig5 = go.Figure(go.Scattergl(x=byhour.index[::1],
#                                 y=byhour.values[::1],mode='markers'))
# fig5.update_layout(title_text='Forecasted power use mw by Hour')
# st.plotly_chart(fig5)
