#programa que implementa k-nn y tambien utiliza herramientas externas para implementarlo
#Ortega Zitle Ariel 201719454 Tratamiento de la información 
#--------librerias-------------
import csv
import pandas as pd
#-----------------librerias para utilizar knn----------------------
import numpy as np #para operaciónes en matrices
import matplotlib.pyplot as plt  #graficas
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
import random
from IPython.display import display

#% matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#-----------------------funciones------------------------
#función que lee el archivo donde se encuentra nuestro conjunto de entrenamiento
def leer_datos():
    print("dame el nombre del archivo (formato csv):")
    nombre_archivo = input()
    nombre_archivo = nombre_archivo + '.csv'
    print(nombre_archivo)
    df = pd.read_csv(nombre_archivo, sep=',', header=None) 
    df
    print(df)
    return df
def knn_implementado(df,k):
    print("Dame el numero de la columna que sera el conjunto de entrenamiento:")
    entrenamiento_colum = int(input())
    entrenamiento_colum = entrenamiento_colum-1
    print(entrenamiento_colum)
    X = df.copy() #creamos copia sin conjunto de entrenamiento 
    del(X[entrenamiento_colum]) #eliminamos la columna a predecir
    Y = df[entrenamiento_colum].values #dejamos solo el conjunto de entrenamiento 
    print(X)
    print(Y)
    #preparamos las entradas
    x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=0)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #clasificacion
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(x_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(x_test, y_test)))
    #matriz de confucion sobre set de prueba
    pred = knn.predict(x_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

def knn_creado(df,k):
    print("Dame el numero de la columna que sera el conjunto de entrenamiento:")
    entrenamiento_colum = int(input())
    entrenamiento_colum = entrenamiento_colum-1
    print(entrenamiento_colum)
    X = df.copy() #creamos copia sin conjunto de entrenamiento 
    del(X[entrenamiento_colum]) #eliminamos la columna a predecir
    Y = df[entrenamiento_colum].values #dejamos solo el conjunto de entrenamiento 
    #print(X)
    #print(Y)
#-------------------------variables-----------------------------    
f=0
op = 1
#-------------------------Clase principal--------------------------
while f!=1:
    print("MENU");
    print("1.Calcular k-nn (creado) ");
    print("2.Clacular k-nn (utilizando herramienta externa)");
    print("3.Calcular error o exactitud de los clasificadores en los puntos 1 y 2");
    print("4.Salir")
    op = int(input("Opción:"))
    if op == 1:
        print("leyendo datos")
        df=leer_datos()
        knn_creado(df,k)
        
    elif op == 2:
        print("leyendo datos")
        df=leer_datos()
        print(df.describe())
        k = int(input("dame el valor para K: "))
        knn_implementado(df,k)
    elif op == 3:
        print("implementar 3")
    elif op == 4:
        print("Saliendo")
        f=1
    else:
        print("Opnción incorrecta ingrese nuevamente")

print("Saliendo")




