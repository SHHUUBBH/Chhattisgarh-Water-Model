"""
Script to retrain SARIMA model with compatible dependency versions.
This script uses numpy 1.26.4 which is compatible with pmdarima.
"""

import pandas as pd
import numpy as np
import glob
import os
import joblib
import pmdarima as pm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("RETRAINING SARIMA MODEL")
print("=" * 70)
print(f"NumPy version: {np.__version__}")
print(f"pmdarima version: {pm.__version__}")

# --- 1. Load and combine the reservoir CSVs ---
print("\n[1/5] Loading reservoir data...")
reservoir_files = glob.glob('data/Reservoir*.csv') 
df_list = [pd.read_csv(file) for file in reservoir_files]
df_reservoir = pd.concat(df_list, ignore_index=True)

df_reservoir.columns = ['Date', 'Storage_BCM']
df_reservoir['Date'] = pd.to_datetime(df_reservoir['Date'])
df_reservoir = df_reservoir.sort_values(by='Date').reset_index(drop=True)

# --- 2. Function to read weather data ---
TARGET_LAT = 22.6
TARGET_LON = 82.6

def get_weather_data_for_location(date, data_type_prefix, data_folder='data/'):
    """Reads IMD binary .grd files"""
    if data_type_prefix == 'Rainfall':
        filename = os.path.join(data_folder, f'Rainfall_ind{date.year}_rfp25.grd')
        lat_start, lon_start = 6.5, 66.5
        lat_step, lon_step = 0.25, 0.25
        num_lats, num_lons = 129, 135
    elif data_type_prefix == 'Maxtemp':
        filename = os.path.join(data_folder, f'Maxtemp_MaxT_{date.year}.GRD')
        lat_start, lon_start = 7.5, 67.5
        lat_step, lon_step = 1.0, 1.0
        num_lats, num_lons = 31, 31
    else:
        return np.nan
    
    if not os.path.exists(filename):
        return np.nan
    
    lat_index = int((TARGET_LAT - lat_start) / lat_step)
    lon_index = int((TARGET_LON - lon_start) / lon_step)
    
    try:
        data = np.fromfile(filename, dtype=np.float32)
        data = data.reshape(-1, num_lats, num_lons)
        day_of_year = date.dayofyear - 1
        value = data[day_of_year, lat_index, lon_index]
        return value if value != -999.0 else np.nan
    except Exception:
        return np.nan

# --- 3. Build the master dataset ---
print("[2/5] Extracting weather data (this may take a moment)...")
df_reservoir['Rainfall_mm'] = df_reservoir['Date'].apply(
    lambda date: get_weather_data_for_location(date, 'Rainfall')
)
df_reservoir['Temp_C'] = df_reservoir['Date'].apply(
    lambda date: get_weather_data_for_location(date, 'Maxtemp')
)

df_master = df_reservoir.set_index('Date')
df_master.dropna(subset=['Rainfall_mm', 'Temp_C'], inplace=True, how='all')
df_master.ffill(inplace=True)

# --- 4. Resample to weekly ---
print("[3/5] Resampling to weekly frequency...")
df_weekly = df_master.resample('W').mean()
df_weekly.dropna(inplace=True)

# Split into train and test
train_weekly = df_weekly.loc['2010-01-01':'2023-12-31']
test_weekly = df_weekly.loc['2024-01-01':]

print(f"Training samples: {len(train_weekly)}, Test samples: {len(test_weekly)}")

# --- 5. Train SARIMA Model ---
print("\n[4/5] Training SARIMA model (this will take a few minutes)...")
print("Using auto_arima to find best parameters...")

sarima_model_weekly = pm.auto_arima(
    train_weekly['Storage_BCM'],
    seasonal=True,
    m=52,  # Weekly seasonality
    trace=True,
    error_action='ignore',  
    suppress_warnings=True, 
    stepwise=True,
    max_p=3,
    max_q=3,
    max_P=2,
    max_Q=2,
    max_d=2,
    max_D=1,
    n_jobs=-1
)

print(f"\nBest SARIMA Model:")
print(f"  Order: {sarima_model_weekly.order}")
print(f"  Seasonal Order: {sarima_model_weekly.seasonal_order}")
print(f"  AIC: {sarima_model_weekly.aic():.2f}")

# --- 6. Save the model ---
print("\n[5/5] Saving SARIMA model...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save SARIMA model
joblib.dump(sarima_model_weekly, 'models/sarima_model.pkl')
print("✓ SARIMA model saved to models/sarima_model.pkl")

# --- 7. Quick validation ---
print("\n" + "=" * 70)
print("VALIDATION - Testing model loading")
print("=" * 70)

try:
    sarima_loaded = joblib.load('models/sarima_model.pkl')
    print("✓ SARIMA model loads successfully")
    print(f"  Model order: {sarima_loaded.order}")
    print(f"  Seasonal order: {sarima_loaded.seasonal_order}")
    
    # Test prediction
    test_forecast = sarima_loaded.predict(n_periods=5)
    print(f"  Test forecast (first 5 weeks): {test_forecast[:5]}")
except Exception as e:
    print(f"✗ SARIMA model failed to load: {e}")

print("\n" + "=" * 70)
print("SARIMA MODEL RETRAINING COMPLETE!")
print("=" * 70)
