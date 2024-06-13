from flask import Flask, jsonify, request
import os
import pandas as pd
from analysis import load_data, preprocess, apply_smoothing, create_linear_regression_model

app = Flask(__name__)

# File path to the Excel file
file_path = os.path.join(os.path.dirname(__file__), '../processed_data_rivero.xlsx')

# Verify the file exists before reading it
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load and preprocess the data once at the start
sheets = load_data(file_path)
df_macroeconomic = preprocess(sheets['Macroeconomic Data'])
df_socioeconomic = preprocess(sheets['Socioeconomic Data'])
df_sociodemographic = preprocess(sheets['Sociodemographic Data'])
df_health_wellbeing = preprocess(sheets['Health Wellbeing Data'])
df_digital_tech_adoption = preprocess(sheets['Digital Tech Adoption'])
df_social_lifestyle = preprocess(sheets['Social Lifestyle Factors'])
df_company = preprocess(sheets['Company Data'])

# Define the forecast years
forecast_years = [2025, 2026, 2027, 2028, 2029, 2030]

@app.route('/forecast/<indicator>', methods=['GET'])
def forecast(indicator):
    if indicator == 'gdp':
        model, forecast, mse, r2, mean_score, std_score = create_linear_regression_model(
            df_macroeconomic, ['Year', 'COVID'], 'GDP Growth (%)', forecast_years
        )
    elif indicator == 'inflation':
        model, forecast, mse, r2, mean_score, std_score = create_linear_regression_model(
            df_macroeconomic, ['Year', 'COVID'], 'Inflation Rate (%)', forecast_years
        )
    # Add other indicators similarly...
    else:
        return jsonify({'error': 'Indicator not found'}), 404

    return jsonify({
        'forecast': forecast.tolist(),
        'mse': mse,
        'r2': r2,
        'cross_val_mse': mean_score,
        'cross_val_std': std_score
    })

if __name__ == '__main__':
    app.run(debug=True)

