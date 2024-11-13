import math
import numpy as np
from scipy.special import expit

"""@author: MarioCHF"""

#NaiveBayes Classifier:
class NaiveBayes():

    def __init__(self,k=1):
        self.k = k
        self.entrenado = False    
                  
    def entrena(self,X,y):
        self.entrenado = True
        self.X = X
        self.y=y
        #Primero vemos cuales son las etiquetas de clasif
        labels,classes_count = np.unique(y,return_counts=True)
        num_labels = len(labels)
        num_instances,num_atr = X.shape
        #soften_constant = self.k*num_atr
        classes_prob = (labels,classes_count/num_instances) #tupla labels, Prob de cada clase
        filters = (y == labels.reshape(-1,1)) # array booleano donde cada fila representa
                                              # las instancias con etiqueta de clasif i 
                                              # (en orden de labels)  
        conditional_probs = [None]*num_atr #guarda una tupla que consiste en los valores que toma el atributo
                                           # y un array con sus probabilidades condicionadas a las clases
        featuresvalues_probs = [None]*num_atr
        #iter sobre los atrib
        for i in range(num_atr):
            atr_i = X[:,i]
            atr_values,atr_counts = np.unique(atr_i,return_counts=True)
            num_values = len(atr_values)
            soften_constant = num_values*self.k
            #crea el numpy array donde se guardan las probs
            cond_prob_matrix = np.zeros((num_labels,num_values))
            #iter sobre las clases
            for j,count in enumerate(classes_count):
                #Tomo el filtro asociado a la clase j, veo cuales son iguales a que valores de los atr
                # (haciendo broadcast) y luego sumo para ver cuantos son
                prob_v_c = (atr_i[filters[j]] == atr_values.reshape(-1,1)).sum(axis=1) #Cuenta cuantos hay con clase j
                                                                                       # de cada valor del atributo con broadcast
                prob_cond = (prob_v_c + self.k)/(count+soften_constant)
                cond_prob_matrix[j] = prob_cond
            conditional_probs[i] = (atr_values,cond_prob_matrix) #guarda el resultado de las prob del atr_i
            featuresvalues_probs[i] = atr_counts/num_instances
        self.classes_prob = classes_prob
        self.conditional_probs = conditional_probs
        self.featuresvalues_probs  = featuresvalues_probs
        


    def clasifica_prob(self,ejemplo):
        try:
            if not(self.entrenado):
                raise ClasificadorNoEntrenado()
            labels,labels_probs = self.classes_prob
            predicted_probs = labels_probs.copy() #vamos a ir acumulando los productos aqui
            denominador  =1
            conditional_probs = self.conditional_probs
            self.featuresvalues_probs
            for i,feature_value in enumerate(ejemplo):
                atr_i_features,atr_i_probs = conditional_probs[i]
                prob_filter = (atr_i_features == feature_value)
                predicted_probs *= (atr_i_probs[:,prob_filter]).flatten() #acumula productos de prob de las clases flatten para
                                                                         #aplanar la columna
                denominador *= self.featuresvalues_probs[i][prob_filter] #calculo el producto de las prob de cada atrib
            predicted_probs2 = predicted_probs/denominador
            normalized_probs = predicted_probs2/predicted_probs2.sum() #lo anterior asume atrib indep luego no es exact una distrib
            return dict(zip(labels,normalized_probs))
        except ClasificadorNoEntrenado:
            print('Antes de predecir hay que ajustar el modelo')
    def clasifica(self,ejemplo):
        try:
            if not(self.entrenado):
                raise ClasificadorNoEntrenado()
            labels,labels_probs = self.classes_prob
            predicted_probs = np.log(labels_probs) #vamos a ir acumulando los productos aqui
            conditional_probs = self.conditional_probs
            for i,feature_value in enumerate(ejemplo):
                atr_i_features,atr_i_probs = conditional_probs[i]
                prob_filter = (atr_i_features == feature_value)
                predicted_probs += np.log((atr_i_probs[:,prob_filter]).flatten()) #acumula productos de prob de las clases
            predicted_label_index  = np.argmax(predicted_probs)
            return labels[predicted_label_index]
        except ClasificadorNoEntrenado:
            print('Excepcion: Antes de predecir hay que ajustar el modelo')

class ClasificadorNoEntrenado(Exception): pass

#Sigmoid function
def sigmoide(x):
    return expit(x)

#Logistic regression binary classifier:
class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
                 batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.entrenado = False
        
        
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
                     early_stopping=False,paciencia=3):
        self.classes = np.unique(y)
        pos_class = self.classes[1]
        best_val_entr = 100000 #Para early stoping
        patience_counter = 0
        self.entrenado=True
        n_instances,n_atr = X.shape
        w = np.random.uniform(-1,1,size=(n_atr,))
        rate = self.rate
        shuffled_indices = np.arange(n_instances)
        resto = n_instances//self.batch_tam #Para ver si al final se queda un bach menor que batch_tam
        num_batches = math.floor(n_instances//self.batch_tam)
        for epoch in range(n_epochs):
            #shuffled_indices = indices.copy()
            np.random.shuffle(shuffled_indices) #aleatoriedad (in-place)
            if resto == 0:
                trainbatches = np.split(shuffled_indices,num_batches) #divide en batches porque es divisible
            else:
                b,c = np.split(shuffled_indices,[num_batches*self.batch_tam]) #si no es divisible separo los que sobran por el 
                                                                              # indice num_batches*batch_tam y luego divido los batch como antes
                trainbatches = np.split(b,num_batches)
                trainbatches.append(c)
            if self.rate_decay:
                rate = rate/(1+epoch)
            if salida_epoch and (epoch == 0):
                #calculo de entropia
                entr_pred = sigmoide(np.dot(X,w))
                val_pred = sigmoide(np.dot(Xv,w))
                pos_class_entr_filter = (y == pos_class)
                pos_class_val_filter = (yv == pos_class)
                eps = 0.005
                entr_entropyno_log = np.where(pos_class_entr_filter,entr_pred,1-entr_pred)
                entr_entropysuav = np.where(entr_entropyno_log==0, eps,entr_entropyno_log)
                entr_entropy = -np.log(entr_entropysuav)
                val_entropyno_log = np.where(pos_class_val_filter,val_pred,1-val_pred)
                val_entropysuav = np.where(val_entropyno_log==0, eps,val_entropyno_log)
                val_entropy = -np.log(val_entropysuav)
                entr_pred_class = (entr_pred >= 0.5)
                val_pred_class = (val_pred >= 0.5)
                num_aciertos_entr = (pos_class_entr_filter == entr_pred_class).sum()
                num_aciertos_val = (pos_class_val_filter == val_pred_class).sum()
                rend_entr = num_aciertos_entr/n_instances
                rend_val = num_aciertos_val/len(yv)
                print(f'Inicialmente, en entrenamiento EC: {entr_entropy.sum()}, rendimiento: {rend_entr}.')
                print(f'Inicialmente, en validacion EC: {val_entropy.sum()}, rendimiento: {rend_val}.')

            for batch in trainbatches:
                X_batch = X[batch]
                y_batch = y[batch]
                predictions = np.dot(X_batch,w)
                prob1 = sigmoide(predictions)
                sum_act_batch = np.dot((y_batch-prob1),X_batch) #prod mtricial de las (y-prob)X_batch que resulta en un vector cuya componente i es
                                                                # el coef de act del peso i
                w += rate*sum_act_batch
            if salida_epoch:
                #calculo de entropia
                val_instances = len(yv)
                entr_pred = sigmoide(np.dot(X,w))
                val_pred = sigmoide(np.dot(Xv,w))
                pos_class_entr_filter = (y == pos_class)
                pos_class_val_filter = (yv == pos_class)
                eps = 0.005
                entr_entropyno_log = np.where(pos_class_entr_filter,entr_pred,1-entr_pred)
                entr_entropysuav = np.where(entr_entropyno_log==0, eps,entr_entropyno_log)
                entr_entropy = -np.log(entr_entropysuav)
                val_entropyno_log = np.where(pos_class_val_filter,val_pred,1-val_pred)
                val_entropysuav = np.where(val_entropyno_log==0, eps,val_entropyno_log)
                val_entropy = -np.log(val_entropysuav)
                sum_entr_entropy = entr_entropy.sum()
                sum_val_entropy = val_entropy.sum()
                entr_pred_class = (entr_pred >= 0.5)
                val_pred_class = (val_pred >= 0.5)
                num_aciertos_entr = (pos_class_entr_filter == entr_pred_class).sum()
                num_aciertos_val = (pos_class_val_filter == val_pred_class).sum()
                #num_aciertos_val = (np.where(val_pred>=0.5,self.classes[1],self.classes[0]) == yv).sum()
                rend_entr = num_aciertos_entr/n_instances
                rend_val = num_aciertos_val/val_instances
                print(f'Epoch {epoch}:en entrenamiento EC: {sum_entr_entropy}, rendimiento: {rend_entr}.')
                print(f'              en validacion EC: {sum_val_entropy}, rendimiento: {rend_val}.')
            if early_stopping:
                if not(salida_epoch):
                    val_pred = sigmoide(np.dot(Xv,w))
                    pos_class_val_filter = (yv == pos_class)
                    eps = 0.005
                    val_entropyno_log = np.where(pos_class_val_filter,val_pred,1-val_pred)
                    val_entropysuav = np.where(val_entropyno_log==0, eps,val_entropyno_log) # a veces predice muy bien y sale 1/0 y por tanto un log no def
                    val_entropy = -np.log(val_entropysuav)
                    sum_val_entropy = val_entropy.sum()
                if sum_val_entropy >= best_val_entr:
                    if patience_counter == paciencia:
                        print(f'PARADA TEMPRANA en epoch {epoch}')
                        break
                    else:
                        patience_counter += 1
                else:
                    best_val_entr = sum_val_entropy
                    patience_counter = 0
        self.weights = w
        

    def clasifica_prob(self,ejemplos):
        try:
            if not(self.entrenado):
                raise ClasificadorNoEntrenado
            return sigmoide(np.dot(ejemplos,self.weights))
            #return np.where(salida>=0.5,self.classes[1],self.classes[0])
        except:
            print('Excepcion: Antes de predecir debes ajustar el modelo')

    
    def clasifica(self,ejemplos):
        try:
            if not(self.entrenado):
                raise ClasificadorNoEntrenado
            salida = sigmoide(np.dot(ejemplos,self.weights))
            return np.where(salida>=0.5,self.classes[1],self.classes[0])
        except:
            print('Excepcion: Antes de predecir debes ajustar el modelo')             


#Logistic Regression One vs Rest:
class RL_OvR():

    def __init__(self,rate=0.1,rate_decay=False,
                   batch_tam=64):
        self.rate = rate
        self.rate_decay=rate_decay
        self.batch_tam = batch_tam


    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        self.classes = np.unique(y)
        n_labels = len(self.classes)
        self.n_labels = n_labels
        clasifs = [None]*n_labels

        for i,label in enumerate(self.classes):
            onevrest_label = (y == label).astype(int) #al aplicar np.unique se mantiene el orden, el 0 va antes que el 1,
                                                                 # luego al aplicar reg log a estas etiquetas la etiqueta positiva sera el 1 
                                                                 # (la segunda), correspondiente con label.
            lr = RegresionLogisticaMiniBatch(rate=self.rate,rate_decay=self.rate_decay,batch_tam=self.batch_tam)
            #print(f'Comienza el entrenamiento del clasificador asociado a la etiqueta {label} \n')
            lr.entrena(X,onevrest_label,n_epochs=n_epochs,salida_epoch=salida_epoch) 
            clasifs[i] = lr   
        self.clasifs = clasifs

    def clasifica(self,ejemplos):
        shape = ejemplos.shape
        if len(shape) == 1:
            n_instances = 1
        else: 
            n_instances = shape[0]
        probOvR = np.zeros((self.n_labels,n_instances)) #matriz con cada fila siendo las predicciones del clasif i a cada uno de los ejemplos
        for i,clasif in enumerate(self.clasifs):
            pred = clasif.clasifica_prob(ejemplos)
            probOvR[i] = pred
        predicted_indices = probOvR.argmax(axis=0) #tomo los indices de las clases que tienen mayor probabilidad para cada ejemplo
        pred_labels = self.classes[predicted_indices] #estos indices los uso para tomar la clase correspondiente
        if n_instances == 1: #si no, daba problema cuando se usaba con rendimiento (devolvia un array 1d con 1 instancia y comparaba mal).
            return pred_labels[0]
        else:
            return pred_labels
