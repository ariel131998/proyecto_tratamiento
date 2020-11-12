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
    distancia_c = [0.0] #creamos arreglo de distancias a clusters
    clusters_datos = np.zeros((k,X.shape[0])) #creamos arreglo que guardara agrupaciones de cluster indice representa el cluster
    #print(clusters_datos)
    i=1
    while i < k:
        nombres_d.insert(i,'cluster'+str(i))
        #distancia_c.insert(i,0.0)
        i = i+1
    print(nombres_d)#se creo los nombres para los cluster
    print(distancia_c) #se creo distancias default
    i=1
    cluster= X.iloc[random.randrange(X.shape[0]-1)]
    clusters = {'cluster0' : cluster} #CREAMOS PRIMER ELEMENTO DEL DICCIONARIO
    while i < k:
        cluster= X.iloc[random.randrange(X.shape[0]-1)]
        clusters.setdefault(nombres_d[i],cluster)
        i = i+1
    print("CLuster Centroides INICIALES:")
    for valores in clusters:
        print(clusters[valores]) #imprimimos los cluster finales (diccionario)
        prueba = clusters[valores]
    #pruebas
    #print("checando......")
    #print(clusters.get("cluster0"))
    #------------------------------Asignacion de datos a centroide mas cercano-------------------
    #sacando distancia entre punto uno y primer cluster
    #print("imprimiedno conjunto prueba")
    #print(prueba)
    #print(X.iloc[1])
    #print(X.iloc[2])
    dist = np.linalg.norm(prueba-(X.iloc[1]))#sirve para crear variable del tipo
    #print(dist)
    i = 0 #limpiamos variable
    j = 0
    r = 0
    del distancia_c[:] #limpiamos lista de distancias
    while i < n_itera: #hasta que se cumpla el numero de iteraciones (tenemos que limpiar todas las matrices)
        #limpieza de datos para cada nueva iteración(puede que arruine todo)
        #del distancia_c[:]
        clusters_datos = np.zeros((k,X.shape[0]))
        while j < X.shape[0]: #hasta que asigne cada fila del conjunto a un cluster(asignación)
            while r < k: #hasta que mida la distancia entre cada cluster
                dist = np.linalg.norm((X.iloc[j])-clusters.get(nombres_d[r]))#realiza medicion entre cluster y fila
                distancia_c.insert(r,dist)   #asignamos distancia calculada a cada cluster
                r = r+1
            #print(distancia_c) #se imprime distancias calculadas
            #print(min(distancia_c))    #buscar la menor y asignar ese conjunto al cluster
            #print(distancia_c.index((min(distancia_c)))) #retorna indice de menor distancia
            mascercano = distancia_c.index((min(distancia_c))) #guarda indice del cluster mas cercano
            #print("mas cercano")
            #print(mascercano)
            #print(nombres_d[mascercano])
            #clusters[nombres_d[mascercano]].append(X.iloc[j])#no funciono
            if j == 0 :
                clusters_datos[mascercano][j] = 0.1
            else: 
                clusters_datos[mascercano][j] = j
            #actualizacion del centroide a la media aritmetica del cluster
            z = 0
            #print(len(X.columns))
            while z < len(X.columns):
                clusters[nombres_d[mascercano]][0] = (clusters[nombres_d[mascercano]][0] + X.iloc[j][0])/2
                z = z+1
            #print(clusters[nombres_d[mascercano]][0])
            #print(X.iloc[j][0])
            j = j+1
            del distancia_c[:] #se vacia la lista de distancias
            r = 0 #limpiamos r
        
        i = i+1
        j=0#limpiamos j para siguiente iteracion
    print("CLuster Centroide finales:")
    print(clusters) #checando cluster finalesss checar-------------
    print("Evaluación de modelo de entrenamiento")
    i=0
    j=0
    #creamos varible para contar cuantos datos tiene cada cluster
    #print(clusters_datos)
    clusters_cant = [0]
    del clusters_cant[:]
    for i in range(k):
        contador = 0
        for j in range(X.shape[0]):
            if clusters_datos[i][j] !=0:
                contador = contador+1
        clusters_cant.insert(i,contador)
    print(clusters_datos) 
    print("Instancias cluster")
    i = 0   
    for datos in clusters_cant:
        print(nombres_d[i],datos)
        i = i+1
    #Asignando y comparando con Y que es el set real---------->aqui me quede 
    clus = np.zeros((X.shape[0]))
    i = 0 
    j = 0
    #print(Y[0])
    for i in range(k):
        for j in range(X.shape[0]):
            if clusters_datos[i][j] ==0.1:
                clus[Y[0]] = clus[Y[0]]+1
            elif clusters_datos[i][j] != 0:
                #print(Y[j])
                arr=int(Y[j])
                #clusters_cant[Y[j]] = clusters_cant[Y[j]]+1
                clus[arr] = clus[arr]+1
        print(np.argmax(clus))            

#-------------------------variables-----------------------------    
f=0
op = 1
#-------------------------Clase principal--------------------------
while f!=1:
    print("MENU");
    print("1.Calcular k-means clustering (creado) ");
    print("2.Calcular k-nn (utilizando herramienta externa)");
    print("3.Calcular error o exactitud de los clasificadores en los puntos 1 y 2 con validación cruzada");
    print("4.Salir")
    print("opcion")
    input()
    op = int(input())
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
        knn_implementado(df,k)
    elif op == 3:
        print("implementar 3")
    elif op == 4:
        print("Saliendo")
        f=1
    else:
        print("Opnción incorrecta ingrese nuevamente")

print("Saliendo")




