import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import os
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Chhattisgarh Reservoir Dashboard",
    page_icon="🌊",
    layout="wide"
)

# --- Helper function for LSTM sequences ---
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x = data[i:i + n_steps, :]
        seq_y = data[i + n_steps, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data():
    """
    Loads all raw data and processes it into a clean master DataFrame.
    """
    try:
        reservoir_files = glob.glob('data/Reservoir*.csv')
        if not reservoir_files:
            st.error("Reservoir CSV files not found in 'data/' folder.")
            return None
        df_list = [pd.read_csv(file) for file in reservoir_files]
        df_reservoir = pd.concat(df_list, ignore_index=True)
        df_reservoir.columns = ['Date', 'Storage_BCM']
        df_reservoir['Date'] = pd.to_datetime(df_reservoir['Date'])
        df_reservoir = df_reservoir.sort_values(by='Date').reset_index(drop=True)

        TARGET_LAT, TARGET_LON = 22.6, 82.6
        def get_weather_data_for_location(date, data_type_prefix, data_folder='data/'):
            if data_type_prefix == 'Rainfall':
                filename = os.path.join(data_folder, f'Rainfall_ind{date.year}_rfp25.grd')
                grid_params = {'lat_start': 6.5, 'lon_start': 66.5, 'lat_step': 0.25, 'lon_step': 0.25, 'num_lats': 129, 'num_lons': 135}
            elif data_type_prefix == 'Maxtemp':
                filename = os.path.join(data_folder, f'Maxtemp_MaxT_{date.year}.GRD')
                grid_params = {'lat_start': 7.5, 'lon_start': 67.5, 'lat_step': 1.0, 'lon_step': 1.0, 'num_lats': 31, 'num_lons': 31}
            else:
                return np.nan
            
            if not os.path.exists(filename): return np.nan
            
            lat_index = int((TARGET_LAT - grid_params['lat_start']) / grid_params['lat_step'])
            lon_index = int((TARGET_LON - grid_params['lon_start']) / grid_params['lon_step'])
            
            try:
                data = np.fromfile(filename, dtype=np.float32)
                data = data.reshape(-1, grid_params['num_lats'], grid_params['num_lons'])
                day_of_year = date.dayofyear - 1
                value = data[day_of_year, lat_index, lon_index]
                return value if value != -999.0 else np.nan
            except Exception:
                return np.nan

        df_reservoir['Rainfall_mm'] = df_reservoir['Date'].apply(lambda date: get_weather_data_for_location(date, 'Rainfall'))
        df_reservoir['Temp_C'] = df_reservoir['Date'].apply(lambda date: get_weather_data_for_location(date, 'Maxtemp'))
        
        df_master = df_reservoir.set_index('Date')
        df_master.dropna(subset=['Rainfall_mm', 'Temp_C'], how='all', inplace=True)
        df_master.ffill(inplace=True)
        
        df_weekly = df_master.resample('W').mean()
        df_weekly.dropna(inplace=True)
        
        df_weekly['Month'] = df_weekly.index.month
        df_weekly['DayOfYear'] = df_weekly.index.dayofyear
        
        return df_weekly
    except Exception as e:
        st.error(f"An error occurred during data processing: {e}")
        return None

@st.cache_resource
def load_models():
    """
    Loads the trained models from the 'models/' folder.
    """
    try:
        sarima_model = joblib.load('models/sarima_model.pkl')
    except FileNotFoundError:
        sarima_model = None
    try:
        lstm_model = load_model('models/lstm_model.keras')
    except (FileNotFoundError, IOError):
        lstm_model = None
    return sarima_model, lstm_model

# --- Main App ---
st.title("🌊 Hasdeo Bango Dam Water Level Dashboard")
st.markdown("An interactive dashboard to visualize historical data and compare forecasting models for reservoir water storage in Chhattisgarh.")

df_weekly = load_and_process_data()
sarima_model, lstm_model = load_models()

if df_weekly is not None:
    st.sidebar.header("Dashboard Options")
    
    train_df = df_weekly.loc['2010-01-01':'2023-12-31']
    test_df = df_weekly.loc['2024-01-01':]
    
    sarima_forecast_series = None
    lstm_forecast_series = None

    if sarima_model:
        sarima_forecast = sarima_model.predict(n_periods=len(test_df))
        sarima_forecast_series = pd.Series(sarima_forecast, index=test_df.index)

    if lstm_model:
        feature_columns = ['Storage_BCM', 'Rainfall_mm', 'Temp_C', 'Month', 'DayOfYear']
        train_df_for_scaling = train_df[feature_columns]
        test_df_for_scaling = test_df[feature_columns]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df_for_scaling)
        test_scaled = scaler.transform(test_df_for_scaling)
        
        n_steps = 4
        X_test, _ = create_sequences(test_scaled, n_steps)
        
        if X_test.size > 0:
            lstm_predictions_scaled = lstm_model.predict(X_test)
            
            dummy_predictions = np.zeros((len(lstm_predictions_scaled), len(feature_columns)))
            dummy_predictions[:, 0] = lstm_predictions_scaled.ravel()
            lstm_predictions_unscaled = scaler.inverse_transform(dummy_predictions)[:, 0]
            
            lstm_forecast_index = test_df.index[n_steps:]
            lstm_forecast_series = pd.Series(lstm_predictions_unscaled, index=lstm_forecast_index)
    
    st.header("Latest Recorded Data")
    latest_data = df_weekly.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Storage", f"{latest_data['Storage_BCM']:.2f} BCM")
    col2.metric("Latest Rainfall", f"{latest_data['Rainfall_mm']:.2f} mm")
    col3.metric("Latest Temperature", f"{latest_data['Temp_C']:.1f} °C")
    
    st.header("2024 Forecast vs. Actual Storage")
    
    plot_df = pd.DataFrame({'Actual': test_df['Storage_BCM']})
    if sarima_forecast_series is not None:
        plot_df['SARIMA Forecast'] = sarima_forecast_series
    if lstm_forecast_series is not None:
        plot_df['LSTM Forecast'] = lstm_forecast_series
    
    fig_forecast = px.line(plot_df, title="Model Forecasts vs. Actual Weekly Storage", labels={'value': 'Storage (BCM)', 'Date': 'Week'})
    fig_forecast.update_traces(selector=dict(name='Actual'), line=dict(width=3))
    if 'SARIMA Forecast' in plot_df:
        fig_forecast.update_traces(selector=dict(name='SARIMA Forecast'), line=dict(dash='dash'))
    if 'LSTM Forecast' in plot_df:
        fig_forecast.update_traces(selector=dict(name='LSTM Forecast'), line=dict(dash='dot'))
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.header("Historical Data Explorer")
    
    # --- FIX: ADDED YEAR RANGE SLIDER ---
    min_year = df_weekly.index.year.min()
    max_year = df_weekly.index.year.max()
    
    selected_years = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(2012, 2024) # Default range as requested
    )
    
    # Filter the dataframe based on the selected years
    filtered_df = df_weekly[
        (df_weekly.index.year >= selected_years[0]) & 
        (df_weekly.index.year <= selected_years[1])
    ]
    
    selected_metric = st.sidebar.selectbox(
        "Select metric to view:",
        options=['Storage_BCM', 'Rainfall_mm', 'Temp_C']
    )
    
    # Plot the filtered data
    fig_historical = px.line(filtered_df, y=selected_metric, title=f"Historical Weekly {selected_metric.replace('_', ' ')} ({selected_years[0]}-{selected_years[1]})")
    st.plotly_chart(fig_historical, use_container_width=True)
    
    if st.sidebar.checkbox("Show Raw Data for Selected Period"):
        st.subheader(f"Raw Weekly Data ({selected_years[0]}-{selected_years[1]})")
        st.dataframe(filtered_df)
else:
    st.warning("Data could not be loaded. Please ensure your data files are correctly placed in the 'data' directory.")