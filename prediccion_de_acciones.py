import numpy as np
np.random.seed(4)
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error


#Esta es una función auxiliar llamada graficar_predicciones que se utilizará más adelante 
#para graficar los resultados de la predicción en comparación con los valores reales.
def graficar_predicciones(real, prediccion):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.legend()
    plt.show()

#
# Lectura de los datos
#
dataset = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
dataset.head()

#Utilizamos el .head para que nos muestre los primeros registros del csv.

set_entrenamiento = dataset[:'2016'].iloc[:,1:2]
set_validacion = dataset['2017':].iloc[:,1:2]

set_entrenamiento['High'].plot(legend=True)
set_validacion['High'].plot(legend=True)
plt.legend(['Entrenamiento (2006-2016)', 'Validación (2017)'])
plt.show()

#En estas líneas, se definen los conjuntos de entrenamiento y validación. El conjunto de entrenamiento
# contiene los datos hasta el año 2021, y el conjunto de validación contiene los datos a partir del año
# 2022. Se selecciona la columna 'High' para ambos conjuntos y se muestra un gráfico para visualizar los datos.

# Normalización del set de entrenamiento
sc = MinMaxScaler(feature_range=(0,1))
set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)

#haremos la transformación inversa para obtener valores en la escala real
#Y así poder graficarlos más adelante

# La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
# partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
time_step = 60
X_train = []
Y_train = []
m = len(set_entrenamiento_escalado)

for i in range(time_step,m):
    # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
    X_train.append(set_entrenamiento_escalado[i-time_step:i,0])

    # Y: el siguiente dato
    Y_train.append(set_entrenamiento_escalado[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Reshape X_train para que se ajuste al modelo en Keras
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#En estas líneas se realiza la preparación de los datos de entrenamiento para la red LSTM. 
#Se utiliza una ventana de tiempo de tamaño time_step para generar los datos de entrada y 
#salida correspondientes. Se recorre el conjunto de entrenamiento escalado y se crea una secuencia 
#de time_step valores consecutivos en X_train, mientras que el siguiente valor después de la secuencia 
#se almacena en Y_train. Luego, los arreglos se convierten a formato de matriz de numpy y se realiza una 
#redimensión para que se ajusten al formato requerido por la red LSTM.

# Explicacion especifica de la linea: 


#Vamos a desglosar la explicación: X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#np.reshape() es una función de la biblioteca NumPy que se utiliza para cambiar la forma (o el tamaño) de una matriz.
#Toma dos argumentos: la matriz que se va a cambiar de forma y la nueva forma deseada.
#X_train es la matriz que se va a cambiar de forma. En este caso, contiene las secuencias de datos de entrada para el entrenamiento de la red LSTM.
#(X_train.shape[0], X_train.shape[1], 1) especifica la nueva forma deseada de X_train. Aquí, X_train.shape[0] devuelve el
#tamaño del primer eje de X_train, que es el número de secuencias en el conjunto de entrenamiento. X_train.shape[1] devuelve el tamaño del segundo
#eje de X_train, que es la longitud de cada secuencia. Finalmente, se agrega un tercer eje con tamaño 1, que representa la dimensión de características.
#Entonces la línea X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) cambia la forma de X_train a una matriz tridimensional, donde 
#el primer eje representa las secuencias, el segundo eje representa los pasos de tiempo en cada secuencia, y el tercer eje representa la dimensión de 
#características (en este caso, 1). Es importante ajustar la forma de los datos de entrenamiento a la esperada por la red LSTM para que pueda procesarlos correctamente. 
#La red LSTM espera una entrada tridimensional de la forma (muestras, pasos de tiempo, características).


#
# Red LSTM
#
dim_entrada = (X_train.shape[1],1)
dim_salida = 1
na = 50

modelo = Sequential()
modelo.add(LSTM(units=na, input_shape=dim_entrada))
modelo.add(Dense(units=dim_salida))
modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=20,batch_size=32)

#En estas líneas se construye el modelo de la red LSTM utilizando Keras. 
#Se define un modelo secuencial y se agregan una capa LSTM y una capa densa. La capa LSTM
#tiene na unidades y recibe una entrada con forma dim_entrada. La capa densa tiene una unidad de salida. 
#El modelo se compila con el optimizador 'rmsprop' y la función de pérdida 'mse'. Luego, se entrena el modelo
#utilizando los datos de entrenamiento, con 20 épocas y un tamaño de lote de 32.

#
# Validación (predicción del valor de las acciones)
#
x_test = set_validacion.values
x_test = sc.transform(x_test)

X_test = []
for i in range(time_step, len(x_test)):
    X_test.append(x_test[i - time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Verificar si X_test y set_validacion tienen la misma longitud
if len(X_test) > len(set_validacion):
    X_test = X_test[:len(set_validacion)]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


prediccion = modelo.predict(X_test)
prediccion = sc.inverse_transform(prediccion)

# Graficar resultados
graficar_predicciones(set_validacion[:len(prediccion)].values, prediccion)



#Se realiza la validación del modelo utilizando el conjunto de validación. Primero,
#se escalan los datos de validación utilizando el mismo escalador que se usó para los 
#datos de entrenamiento. Luego, se preparan los datos de validación de manera similar a como 
#se prepararon los datos de entrenamiento. Se utilizan ventanas de tiempo y se obtiene la predicción del
#modelo para estos datos. Finalmente, se invierte la transformación de escala de la predicción y se utiliza la 
#función graficar_predicciones para mostrar el gráfico de los valores reales de validación y las predicciones realizadas por el modelo.

# Métricas de error
error_rmse = np.sqrt(mean_squared_error(set_validacion[:len(prediccion)].values, prediccion))
error_mae = mean_absolute_error(set_validacion[:len(prediccion)].values, prediccion)

#ERROR CUADRATICO MEDIO
#ERROR ABSOLUTO MEDIO
print("Error RMSE:", error_rmse)
print("Error MAE:", error_mae)
