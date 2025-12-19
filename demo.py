import pandas as pd
import streamlit as st
import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from SALib.sample import saltelli
from SALib.analyze import sobol

st.title('Energy Market Forecast')
def forecast_models(df_model, target_col='Total final consumption (PJ)', split_year=2015, models_to_run=None, plot=True, future_years=None): 
    df_model = df_model.sort_values('Year').reset_index(drop=True)
    y = df_model[target_col]
    X = df_model.drop(columns=[target_col, 'Year'])   
    mask_train = df_model['Year'] < split_year
    mask_test  = df_model['Year'] >= split_year
    X_train, y_train = X.loc[mask_train], y.loc[mask_train]
    X_test, y_test   = X.loc[mask_test], y.loc[mask_test]
    years_train = df_model.loc[mask_train, 'Year']
    years_test  = df_model.loc[mask_test, 'Year'] 
    n_neighbors = min(4, len(X_train))
    all_models = {'Linear Regression': LinearRegression(), 'Lasso': Lasso(alpha=0.1), 'Ridge': Ridge(alpha=1), 
                  'Decision Tree': DecisionTreeRegressor(random_state=42),
                  'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42), 
                  'kNN': KNeighborsRegressor(n_neighbors=n_neighbors), 'SVR': SVR()} 
    if models_to_run is None:
        models = all_models
    else:
        if isinstance(models_to_run, str):
            models_to_run = [models_to_run]
        models = {name: all_models[name] for name in models_to_run if name in all_models}
        missing = set(models_to_run) - set(models.keys())
        if missing:
            raise ValueError(f'Unknown model(s): {missing}')
    predictions_dict = {}
    trained_models = {}  # <-- store trained models
    if future_years is not None:
        X_future = pd.DataFrame()
        for col in X.columns:
            slope = X[col].iloc[-1] - X[col].iloc[-2]
            X_future[col] = [X[col].iloc[-1] + slope*(i+1) for i in range(len(future_years))]
        X_future['Year'] = future_years
    else:
        X_future = None    
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model 
        y_train_pred = model.predict(X_train)
        y_test_pred  = model.predict(X_test)
        if X_future is not None:
            X_future_aligned = X_future[X.columns]
            y_future_pred = model.predict(X_future_aligned)
        else:
            y_future_pred = []
        df_pred = pd.DataFrame({
            'Year': list(years_train) + list(years_test) + list(future_years if X_future is not None else []),
            'Train Prediction': list(y_train_pred) + [np.nan]*len(y_test_pred) + [np.nan]*len(y_future_pred),
            'Test Prediction':  [np.nan]*len(y_train_pred) + list(y_test_pred) + [np.nan]*len(y_future_pred),
            'Future Prediction': [np.nan]*(len(y_train_pred)+len(y_test_pred)) + list(y_future_pred)})
        predictions_dict[name] = df_pred
        if plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_pred['Year'], df_pred['Train Prediction'], marker='o', label='Train Prediction')
            ax.plot(df_pred['Year'], df_pred['Test Prediction'], marker='o', label='Test Prediction')
            if len(y_future_pred) > 0:
                ax.plot(df_pred['Year'], df_pred['Future Prediction'], marker='o', label='Future Prediction')
            ax.plot(df_model['Year'], y, linestyle='--', color='k', alpha=0.6, label='Actual')
            ax.set_title(f'{name} Predictions vs Actual')
            ax.set_xlabel('Year')
            ax.set_ylabel(target_col)
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
    return predictions_dict, trained_models

def run_sensitivity_analysis(model, df_model, country, gdp_2024=None, crude_2024=None, crude_2025=None, oil_2024=None, oil_2025=None, natgas_2024=None, 
                             natgas_2025=None, snp_2024=None, snp_2025=None, n_samples=1000):
    features = df_model.drop(columns=['Year', 'Country', 'Total final consumption (PJ)'], errors='ignore').columns.tolist()
    problem_features = []
    bounds = []
    base_values = []
    if 'GDP' in features:
        gdp_value = gdp_2024
        if np.isnan(gdp_value):
            raise ValueError(f'GDP for {country} not found in gdp_dict')
        problem_features.append('GDP')
        bounds.append([0.9*gdp_value, 1.1*gdp_value])
        base_values.append(gdp_value)
    if 'G_crude' in features:
        problem_features.append('G_crude')
        lower = min(0.9*crude_2024, 1.1*crude_2025)
        upper = max(0.9*crude_2024, 1.1*crude_2025)
        bounds.append([lower, upper])
        base_values.append(crude_2024)    
    if 'G_oil_products' in features and oil_2024 is not None:
        problem_features.append('G_oil_products')
        lower = min(0.9*oil_2024, 1.1*oil_2025)
        upper = max(0.9*oil_2024, 1.1*oil_2025)
        bounds.append([lower, upper])
        base_values.append(oil_2024)
    if 'G_natgas' in features and natgas_2024 is not None:
        problem_features.append('G_natgas')
        lower = min(0.9*natgas_2024, 1.1*natgas_2025)
        upper = max(0.9*natgas_2024, 1.1*natgas_2025)
        bounds.append([lower, upper])
        base_values.append(natgas_2024)   
    if 'G_S&P' in features and snp_2024 is not None:
        problem_features.append('G_S&P')
        lower = min(0.9*snp_2024, 1.1*snp_2025)
        upper = max(0.9*snp_2024, 1.1*snp_2025)
        bounds.append([lower, upper])
        base_values.append(snp_2024)
    if len(problem_features) == 0:
        raise ValueError('No matching features found for sensitivity analysis')
    problem = {'num_vars': len(problem_features), 'names': problem_features, 'bounds': bounds}
    param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
    Y = []
    X_features = model.feature_names_in_ 
    for row in param_values:
        x = base_values.copy()
        for i, val in enumerate(row):
            x[i] = val
        x_dict = dict(zip(problem_features, x))
        full_input = df_model[X_features].iloc[-1].copy()
        for k, v in x_dict.items():
            if k in full_input.index:  
                full_input[k] = v
        y_pred = model.predict(full_input.values.reshape(1, -1))
        Y.append(y_pred[0])
    Y = np.array(Y)
    Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=True) 
    return Si

def si_to_dataframe(Si_dict, feature_names):
    return pd.DataFrame({
        'S1': Si_dict['S1'],
        'S1_conf': Si_dict['S1_conf'],
        'ST': Si_dict['ST'],
        'ST_conf': Si_dict['ST_conf'],
    }, index=feature_names)

def interpret_sensitivity_df(Si_df, threshold_high=0.7, threshold_medium=0.3):
    table = []
    for feature in ['GDP', 'G_oil_products', 'G_crude', 'G_natgas', 'G_S&P']:
        if feature not in Si_df.index or Si_df.loc[feature, 'S1'] == 0:
            interpretation = 'No impact on prices.'
            s1_val = 0.0
        else:
            s1_val = Si_df.loc[feature, 'S1']
            if s1_val >= threshold_high:
                interpretation = f'Highly sensitive. Monitor {feature} as it strongly affects energy consumption.'
            elif s1_val >= threshold_medium:
                interpretation = f'Moderately sensitive. {feature} movements may influence consumption trends.'
            else:
                interpretation = f'Slight sensitivity; minor effect on energy consumption.'
        table.append({'Feature': feature, 'S1': s1_val, 'Interpretation': interpretation})
    return pd.DataFrame(table)


def get_last_price(ticker, fallback):
    t = yf.Ticker(ticker)
    hist = t.history(period='5d')
    if not hist.empty:
        return float(hist['Close'].iloc[-1])
    return fallback



url_dict = {
    'Ireland': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSBmkEyStd7YrwpIqUnswch2g_1bU7H0OpQb_Q0JlKAPsfmI0UdGYXIfNMDkIwjq1xH9MvOAhwvRy4h/pub?output=csv',
    'Belgium': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRlhSegddg8JHn48gZbKv7BBrusSHfE9bKv7VayHpt9x4_E3g16iuRonWGXskymjZDsedcjmeKBMhfJ/pub?output=csv',
    'Norway': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSXwT5yUdmIiGmKwSp33Ks-ohIBGmcFe2A6qC60qr9xR6jwxpvpJ-cwihYDK-CNb97S6HDnyKxrt-bJ/pub?output=csv',
    'Poland': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTYl0Bb4gWasfG3WcTrNyKbjdd8Eb-n199Akj15Vx5PNPfHrublOH96CFkhTfJxZKzc-7QUUjJxWHpA/pub?output=csv'}

crude_2024  = 74.64
natgas_2024 = 3.633
snp_2024    = 5935.7
oil_2024 = 268.45
coal_2024 = 184.20
crude_2025  = get_last_price('BZ=F', 60.06)
natgas_2025 = get_last_price('NG=F', 3.91)
snp_2025    = get_last_price('ES=F', 6838.75)
crude_pct  = (crude_2025  - crude_2024)  / crude_2024
natgas_pct = (natgas_2025 - natgas_2024) / natgas_2024
snp_pct    = (snp_2025    - snp_2024)    / snp_2024
gdp_dict = {'Ireland': 577389000000.00, 'Austria': 521640000000.00, 'Belgium': 644783000000.00, 'Norway': 517102000000.00, 'Poland': 1030000000000.00,    
            'Sweden': 584960000000.00}


market_df = pd.DataFrame({'Asset': ['Brent Oil', 'Natural Gas', 'S&P Index'], 'Price (Today)': [crude_2025, natgas_2025, snp_2025], '% Change (vs 2024)': [crude_pct, natgas_pct, snp_pct]})


timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
st.subheader('Market Snapshot')
st.caption(f'Updated at {timestamp} - Refresh app to get lastest quotes')
st.dataframe(market_df, use_container_width=True, hide_index=True)

country = st.selectbox('Select a country', list(url_dict.keys()))

DATA_URL = url_dict[country]
df = pd.read_csv(DATA_URL)
gdp_2024 = gdp_dict[country]
if country == 'Ireland':
    predictions, trained_models = forecast_models(df, models_to_run='Random Forest', future_years=list(range(2024,2031)))
    model = trained_models['Random Forest']
    st.caption(f'Loading sensibility analysis might take up to 4 minutes')
    Si = run_sensitivity_analysis(model,  df_model=df,  country='Ireland', gdp_2024=gdp_2024, crude_2024=crude_2024, crude_2025=crude_2025, oil_2024=oil_2024,
                                  oil_2025= 253.59, natgas_2024=natgas_2024, natgas_2025=natgas_2025, snp_2024=snp_2024, snp_2025=snp_2025, n_samples=500)
    s1_map = dict(zip(Si['names'], Si['S1']))  # Si['names'] should have feature names
    features_list = ['GDP', 'G_oil_products', 'G_crude', 'G_natgas', 'G_S&P']
    table_rows = []
    for feature in features_list:
        s1_val = s1_map.get(feature, 0)
        if s1_val >= 0.7:
            interpretation = f"Highly sensitive. Monitor {feature} as it strongly affects energy consumption."
        elif s1_val >= 0.3:
            interpretation = f"Moderately sensitive. {feature} movements may influence consumption trends."
        elif s1_val > 0:
            interpretation = f"Slight sensitivity; minor effect on energy consumption."
        else:
            interpretation = "No impact on prices."
        table_rows.append({'Feature': feature, 'S1': s1_val, 'Interpretation': interpretation})
    sensitivity_df = pd.DataFrame(table_rows)
    st.subheader(f"Sensitivity Analysis for {country}")
    st.dataframe(sensitivity_df, use_container_width=True)
elif country == 'Belgium':
    predictions, trained_models = forecast_models(df, models_to_run='kNN', future_years=list(range(2024,2031)))
    model = trained_models['kNN']
    st.caption(f'Loading sensibility analysis might take up to 4 minutes')
    Si = run_sensitivity_analysis(model,  df_model=df,  country='Belgium', gdp_2024=gdp_2024, crude_2024=crude_2024, crude_2025=crude_2025, oil_2024=oil_2024, oil_2025= 253.59, natgas_2024=natgas_2024, natgas_2025=natgas_2025, snp_2024=snp_2024, snp_2025=snp_2025, n_samples=500)
    s1_map = dict(zip(Si['names'], Si['S1']))  # Si['names'] should have feature names
    features_list = ['GDP', 'G_oil_products', 'G_crude', 'G_natgas', 'G_S&P']
    table_rows = []
    for feature in features_list:
        s1_val = s1_map.get(feature, 0)
        if s1_val >= 0.7:
            interpretation = f"Highly sensitive. Monitor {feature} as it strongly affects energy consumption."
        elif s1_val >= 0.3:
            interpretation = f"Moderately sensitive. {feature} movements may influence consumption trends."
        elif s1_val > 0:
            interpretation = f"Slight sensitivity; minor effect on energy consumption."
        else:
            interpretation = "No impact on prices."
        table_rows.append({'Feature': feature, 'S1': s1_val, 'Interpretation': interpretation})
    sensitivity_df = pd.DataFrame(table_rows)
    st.subheader(f"Sensitivity Analysis for {country}")
    st.dataframe(sensitivity_df, use_container_width=True)

elif country == 'Norway':
    predictions, trained_models = forecast_models(df, models_to_run='kNN', future_years=list(range(2024,2031)))
    model = trained_models['kNN']
    st.caption(f'Loading sensibility analysis might take up to 4 minutes')
    Si = run_sensitivity_analysis(model,  df_model=df,  country='Norway', gdp_2024=gdp_2024, crude_2024=crude_2024, crude_2025=crude_2025, oil_2024=oil_2024,
                              oil_2025= 253.59, natgas_2024=natgas_2024, natgas_2025=natgas_2025, snp_2024=snp_2024, snp_2025=snp_2025, n_samples=500)
    s1_map = dict(zip(Si['names'], Si['S1']))  # Si['names'] should have feature names
    features_list = ['GDP', 'G_oil_products', 'G_crude', 'G_natgas', 'G_S&P']
    table_rows = []
    for feature in features_list:
        s1_val = s1_map.get(feature, 0)
        if s1_val >= 0.7:
            interpretation = f"Highly sensitive. Monitor {feature} as it strongly affects energy consumption."
        elif s1_val >= 0.3:
            interpretation = f"Moderately sensitive. {feature} movements may influence consumption trends."
        elif s1_val > 0:
            interpretation = f"Slight sensitivity; minor effect on energy consumption."
        else:
            interpretation = "No impact on prices."
        table_rows.append({'Feature': feature, 'S1': s1_val, 'Interpretation': interpretation})
    sensitivity_df = pd.DataFrame(table_rows)
    st.subheader(f"Sensitivity Analysis for {country}")
    st.dataframe(sensitivity_df, use_container_width=True)

else:
    predictions, trained_models = forecast_models(df, models_to_run='Lasso', future_years=list(range(2024,2031)))
    model = trained_models['Lasso']
    st.caption(f'Loading sensibility analysis might take up to 4 minutes')
    Si = run_sensitivity_analysis(model,  df_model=df,  country='Poland', gdp_2024=gdp_2024, crude_2024=crude_2024, crude_2025=crude_2025, oil_2024=oil_2024,
                                  oil_2025= 253.59, natgas_2024=natgas_2024, natgas_2025=natgas_2025, snp_2024=snp_2024, snp_2025=snp_2025, n_samples=500)
    s1_map = dict(zip(Si['names'], Si['S1']))  # Si['names'] should have feature names
    features_list = ['GDP', 'G_oil_products', 'G_crude', 'G_natgas', 'G_S&P']
    table_rows = []
    for feature in features_list:
        s1_val = s1_map.get(feature, 0)
        if s1_val >= 0.7:
            interpretation = f"Highly sensitive. Monitor {feature} as it strongly affects energy consumption."
        elif s1_val >= 0.3:
            interpretation = f"Moderately sensitive. {feature} movements may influence consumption trends."
        elif s1_val > 0:
            interpretation = f"Slight sensitivity; minor effect on energy consumption."
        else:
            interpretation = "No impact on prices."
        table_rows.append({'Feature': feature, 'S1': s1_val, 'Interpretation': interpretation})
    sensitivity_df = pd.DataFrame(table_rows)
    st.subheader(f"Sensitivity Analysis for {country}")
    st.dataframe(sensitivity_df, use_container_width=True)


    

    




