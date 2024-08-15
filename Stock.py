import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from io import BytesIO

# Custom CSS for background image
css = """
<style>
body {
    background-image: url('data:download (3).jfif');
    background-size: cover;
    background-attachment: fixed;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Avg Price'] = data[['Open', 'Low', 'High', 'Adj Close']].mean(axis=1)
    data['Differenced'] = data['Avg Price'].diff()
    data.dropna(subset=['Differenced'], inplace=True)
    return data

def sarimax_model(train, test, steps):
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit(disp=False)
    predictions = fitted_model.forecast(steps=len(test))
    future_predictions = fitted_model.forecast(steps=steps)

    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)

    return predictions, future_predictions, mae, mse, rmse

st.title('Stock Price Prediction with SARIMAX')

symbol = st.text_input('Stock Symbol', 'RELIANCE.NS')
start_date = st.date_input('Start Date', pd.to_datetime('2014-07-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-06-30'))
future_steps = st.number_input('Future Steps', min_value=1, max_value=100, value=30)

data = get_stock_data(symbol, start_date, end_date)
if data.empty:
    st.error('No data fetched. Please check the stock symbol and date range.')
else:
    st.sidebar.title('Navigation')
    section = st.sidebar.radio('Go to', ['Data Details', 'Visualizations', 'Predictions'])

    if section == 'Data Details':
        st.header('Data Details')
        st.write(data)
        st.write('## Summary Statistics')
        st.write(data.describe())

    elif section == 'Visualizations':
        st.header('Visualizations')

        st.subheader('Historical Average Price')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Avg Price'], label='Avg Price')
        ax.set_title('Historical Average Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig)

    elif section == 'Predictions':
        st.header('Predictions')

        train_size = int(len(data) * 0.8)
        train = data['Avg Price'][:train_size]
        test = data['Avg Price'][train_size:]

        if st.button('Predict'):
            sarimax_predictions, sarimax_future, sarimax_mae, sarimax_mse, sarimax_rmse = sarimax_model(train, test, future_steps)
            st.write(f"SARIMAX Model - MAE: {sarimax_mae}, MSE: {sarimax_mse}, RMSE: {sarimax_rmse}")

            future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_steps, freq='B')

            st.subheader('Prediction Results')
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(data.index, data['Avg Price'], label='Historical Data')
            ax.plot(test.index, test, label='Test Data', color='blue')
            ax.plot(test.index, sarimax_predictions, label='SARIMAX Predictions', color='red')
            ax.plot(future_dates, sarimax_future, label='SARIMAX Future Predictions', color='orange')
            ax.legend()
            st.pyplot(fig)

            st.subheader('Future Predictions')
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Avg Price': sarimax_future})
            st.write(future_df)
