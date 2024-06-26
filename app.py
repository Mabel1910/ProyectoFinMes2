from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_clima_regresion_v2.pkl')
scaler = joblib.load('standard_scaler_v2.pkl')  # Asegúrate de que 'scaler.pkl' sea un objeto de escalado como StandardScaler o MinMaxScaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        years = float(request.form['years'])
        rating = float(request.form['rating'])
        condition = float(request.form['condition'])
        economy = float(request.form['economy'])
        hp = float(request.form['hp'])

        # Escalar los datos de entrada
        input_data = [[years, rating, condition, economy, hp]]
        scaled_data = scaler.transform(input_data)
        
        # Crear un DataFrame con los datos escalados
        data_df = pd.DataFrame(scaled_data, columns=['years', 'rating', 'condition', 'economy', 'hp'])
        
        # Imprimir el DataFrame para verificar los datos
        print("Datos recibidos (escalados):")
        print(data_df)
        
        # Realizar la predicción
        prediction = model.predict(data_df)[0]
        
        # Devolver la predicción como respuesta JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
    
    
    
    
    

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('model.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        km = float(request.form['km'])
        on_road_now = float(request.form['on_road_now'])
        on_road_old = float(request.form['on_road_old'])
        condition = float(request.form['condition'])
        rating = float(request.form['rating'])
        top_speed = float(request.form['top_speed'])
        
        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[km, on_road_now, on_road_old, condition, rating, top_speed]], 
                               columns=['km', 'on road now', 'on road old', 'condition', 'rating', 'top speed'])
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Realizar predicciones
        prediction = model.predict(data_df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Convertir la predicción a float
        prediction_float = float(prediction[0])
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'prediccion': prediction_float})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

