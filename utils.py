import math
import random
import numpy as np

"""@author: MarioCHF"""

#Implementation of train_test_split:
def particion_entr_prueba(X,y,test=0.20):
    labels,counts = np.unique(y,return_counts=True)
    num_data = len(y)
    indices = np.arange(num_data)
    filter = (y==labels[0])
    one_label= indices[filter]
    amount = counts[0]
    split = math.floor(amount*test) 
    random.shuffle(one_label)
    train_split_indices,test_split_indices = one_label[split:],one_label[:split]
    for label,amount in zip(labels[1:],counts[1:]):
        filter = (y==label)
        one_label= indices[filter]
        split = math.floor(amount*test)
        random.shuffle(one_label)
        train_split_indices = np.concatenate([train_split_indices,one_label[split:]])
        test_split_indices = np.concatenate([test_split_indices,one_label[:split]])
    sorted_test = np.sort(test_split_indices)
    sorted_train = np.sort(train_split_indices)
    return X[sorted_train],X[sorted_test],y[sorted_train],y[sorted_test]

#Accuracy metric:
def rendimiento(clasificador,X,y):
    num_instances = X.shape[0]
    pred_labels = np.array([clasificador.clasifica(X[i]) for i in range(num_instances)])
    return (y == pred_labels).sum()/num_instances

#Standard scaler:
class NormalizadorStandard():

    def __init__(self):
        self.ajustado = False
    
        
    def ajusta(self,X):
        self.ajustado = True
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
    def normaliza(self,X):
        try:
            if not(self.ajustado):
                raise NormalizadorNoAjustado
            return (X-self.mu)/self.sigma
        except NormalizadorNoAjustado:
            print('Excepcion: Antes de normalizar hay que ajustar los datos')

class NormalizadorNoAjustado(Exception): pass

#Columns to OneHot:
def codifica_one_hot(X):
    num_atr = X.shape[1]
    one_hotatr = [None]*num_atr
    for i in range(num_atr):
        atr = X[:,i]
        atr_labels = np.unique(atr) #etiquetas
        onehotarray = (((atr == atr_labels.reshape(-1,1))).transpose()).astype(float) #broadcast. 
                                                                            # cada fila es una columna one hot.
                                                                            #se traspone y se pasa a float
        one_hotatr[i] = onehotarray
    return np.concatenate(one_hotatr,axis=1)

#Auxiliary functions to load digits dataset:
def cargaImágenes(fichero,ancho,alto):

    def convierte_0_1(c):
        if c==" ":
            return 0
        else:
            return 1
        
    
    with open(fichero) as f:
        lista_imagenes=[]
        ejemplo=[]
        cont_lin=0
        for lin in f:
            ejemplo.extend(list(map(convierte_0_1,lin[:ancho])))
            cont_lin+=1
            if cont_lin == alto:
                lista_imagenes.append(ejemplo)   
                ejemplo=[]
                cont_lin=0
    return np.array(lista_imagenes)

def cargaClases(fichero):
    with open(fichero) as f:
        return np.array([int(c) for c in f])
