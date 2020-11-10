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
import random#para inicializar k-means
from IPython.display import display

#% matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix #matriz de confusion 
from sklearn.model_selection import KFold  #validacion cruzada
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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
    #print(X)
    #print(Y)
    #preparamos las entradas
    x_train, x_test, y_train, y_test = train_test_split(X,Y, random_state=0) #entrenamiento 75% y test 25%
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #clasificacion
    knn = KNeighborsClassifier(k)
    knn.fit(x_train, y_train)#hace el entrenamiento
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(x_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(x_test, y_test)))
    #validación cruzada (solo utilizaremos el de set de entrenamiento por el momento)
    print("Dame el numero de k para la validacion cruzada")
    n_splits = int(input())
    kf = KFold(n_splits)
    scores = cross_val_score(knn, x_train, y_train, cv=kf, scoring="accuracy") 
    print("Metricas cross_validation", scores)#falta separar el mejor y el peor
    print("Media de cross_validation", scores.mean())
    #matriz de confucion sobre set de prueba (opcional cambiar al final)
    pred = knn.predict(x_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

def k_means(df,k,n_itera):
    print("Dame el numero de la columna que sera el conjunto de entrenamiento:")
    entrenamiento_colum = int(input())
    entrenamiento_colum = entrenamiento_colum-1
    print(entrenamiento_colum)
    X = df.copy() #creamos copia sin conjunto de entrenamiento 
    del(X[entrenamiento_colum]) #eliminamos la columna a predecir
    Y = df[entrenamiento_colum].values #dejamos solo el conjunto de entrenamiento 
    #empieza construccion de clasificador----------------------------
    #------------------Inicializacion (asignaremos aleatoriamente un conjuto a cada cluster)-----------------
    
    nombres_d = ["cluster0"] #'cluster'+str(0)
    i=1
    while i < k:
        nombres_d.insert(i,'cluster'+str(i))
        i = i+1
    print(nombres_d)#se creo los nombres para los cluster
    i=1
    cluster= X.iloc[random.randrange(X.shape[0]-1)]
    clusters = {'cluster1' : cluster} #CREAMOS PRIMER ELEMENTO DEL DICCIONARIO
    while i < k:
        cluster= X.iloc[random.randrange(X.shape[0]-1)]
        clusters.setdefault(nombres_d[i],cluster)
        i = i+1
    print(clusters)#imprimimos los cluster finales (diccionario)
    
    #------------------------------Asignacion de datos a centroide mas cercano-------------------

    #actualizacion del centroide a la media aritmetica del cluster

#-------------------------variables-----------------------------    
f=0
op = 1
#-------------------------Clase principal--------------------------
while f!=1:
    print("MENU");
    print("1.Calcular k-means clustering (creado) ");
    print("2.Clacular k-nn (utilizando herramienta externa)");
    print("3.Calcular error o exactitud de los clasificadores en los puntos 1 y 2 con validación cruzada");
    print("4.Salir")
    op = int(input("Opción:"))
    if op == 1:
        print("leyendo datos")
        df=leer_datos()
        print(df.describe())
        k = int(input("dame el valor para K: "))
        n_itera = int(input("dame el numero de iteraciones: "))
        k_means(df,k, n_itera)
        
    elif op == 2:
        print("leyendo datos")
        df=leer_datos()
        print(df.describe())
        k = int(input("dame el valor para K: "))
        #pedir numero de iteraciones
        n_itera = int(input("Numero de iteraciones:"))
        knn_implementado(df,k,n_itera)
    elif op == 3:
        print("implementar 3")
    elif op == 4:
        print("Saliendo")
        f=1
    else:
        print("Opnción incorrecta ingrese nuevamente")

print("Saliendo")




