from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load and prepare your model and scalers (copied from app.py)
data = pd.read_csv('Init.csv')  # Use relative path
data = pd.get_dummies(data, columns=['State'], drop_first=True)
data['Taxa de Conversão em Vendas'] = data['Taxa de Conversão em Vendas'].str.replace('%', '').astype(float)
data['Gain per client'] = data['Gain per client'].str.replace('[\$,]', '', regex=True).astype(float)
data['Gain'] = data['Gain'].str.replace('[\$,]', '', regex=True).astype(float)
data['Investment'] = data['Investment'].str.replace('[\$,]', '', regex=True).astype(float)

features = ['Taxa de Conversão em Vendas', 'Gain per client', 'Gain']
target = 'Investment'
X = data[features]
y = np.log(data[target])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.json
        current_conversion_rate = content['current_conversion_rate']
        current_gain_per_client = content['current_gain_per_client']
        desired_gain = content['desired_gain']

        user_input = pd.DataFrame({
            'Taxa de Conversão em Vendas': [current_conversion_rate],
            'Gain per client': [current_gain_per_client],
            'Gain': [desired_gain]
        })

        user_input_scaled = scaler.transform(user_input)
        user_input_poly = poly.transform(user_input_scaled)
        investment_log = ridge.predict(user_input_poly)
        investment = np.exp(investment_log)
        return jsonify({'predicted_investment': investment[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
