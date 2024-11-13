import numpy as np
from models import *
from utils import *
from carga_datos import *

print("************ TRAIN_TEST_SPLIT EXAMPLE:")
print("**********************************\n")
Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)
print("Partición votos: ",y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0])
print("Proporción original en votos: ",np.unique(y_votos,return_counts=True))
print("Estratificación entrenamiento en votos: ",np.unique(ye_votos,return_counts=True))
print("Estratificación prueba en votos: ",np.unique(yp_votos,return_counts=True))
print("\n")

Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)
print("Proporción original en cáncer: ", np.unique(y_cancer,return_counts=True))
print("Estratificación entr-val en cáncer: ",np.unique(yev_cancer,return_counts=True))
print("Estratificación prueba en cáncer: ",np.unique(yp_cancer,return_counts=True))
Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)
print("Estratificación entrenamiento cáncer: ", np.unique(ye_cancer,return_counts=True))
print("Estratificación validación cáncer: ",np.unique(yv_cancer,return_counts=True))
print("\n")

Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)
print("Estratificación entrenamiento crédito: ",np.unique(ye_credito,return_counts=True))
print("Estratificación prueba crédito: ",np.unique(yp_credito,return_counts=True))
print("\n\n\n")





print("************ NAIVEBAYES EXAMPLE:")
print("**********************************\n")

nb_tenis=NaiveBayes(k=0.5)
nb_tenis.entrena(X_tenis,y_tenis)
ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
print("NB Clasifica_prob un ejemplo tenis: ",nb_tenis.clasifica_prob(ej_tenis))
#print("NB Clasifica un ejemplo tenis: ",nb_tenis.clasifica([ej_tenis])) #En los ejemplos del ejercicio aparecia sin lista
print("NB Clasifica un ejemplo tenis: ",nb_tenis.clasifica(ej_tenis))
print("\n")

nb_votos=NaiveBayes(k=1)
nb_votos.entrena(Xe_votos,ye_votos)
print("NB Rendimiento votos sobre entrenamiento: ", rendimiento(nb_votos,Xe_votos,ye_votos))
print("NB Rendimiento votos sobre test: ", rendimiento(nb_votos,Xp_votos,yp_votos))
print("\n")


nb_credito=NaiveBayes(k=1)
nb_credito.entrena(Xe_credito,ye_credito)
print("NB Rendimiento crédito sobre entrenamiento: ", rendimiento(nb_credito,Xe_credito,ye_credito))
print("NB Rendimiento crédito sobre test: ", rendimiento(nb_credito,Xp_credito,yp_credito))
print("\n")


nb_imdb=NaiveBayes(k=1)
nb_imdb.entrena(X_train_imdb,y_train_imdb)
print("NB Rendimiento imdb sobre entrenamiento: ", rendimiento(nb_imdb,X_train_imdb,y_train_imdb))
print("NB Rendimiento imdb sobre test: ", rendimiento(nb_imdb,X_test_imdb,y_test_imdb))
print("\n")


print("************ STANDARD SCALER EXAMPLE:")
print("**********************************\n")



normst_cancer=NormalizadorStandard()
normst_cancer.ajusta(Xe_cancer)
Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

print("Normalización cancer entrenamiento: ",np.mean(Xe_cancer,axis=0))
print("Normalización cancer validación: ",np.mean(Xv_cancer,axis=0))
print("Normalización cancer test: ",np.mean(Xp_cancer,axis=0))

print("\n\n\n")



print("************ LOGISTIC REGRESSION EXAMPLE:")
print("**********************************\n")


lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)
print("LR clasifica cuatro ejemplos cáncer (y valor esperado): ",lr_cancer.clasifica(Xp_cancer_n[17:21]),yp_cancer[17:21])
print("LR clasifica_prob cuatro ejemplos cáncer: ", lr_cancer.clasifica_prob(Xp_cancer_n[17:21]))
print("LR rendimiento cáncer entrenamiento: ", rendimiento(lr_cancer,Xe_cancer_n,ye_cancer))
print("LR rendimiento cáncer prueba: ", rendimiento(lr_cancer,Xp_cancer_n,yp_cancer))

print("\n\n CON SALIDA Y EARLY STOPPING**********************************\n")

lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

print("\n\n\n")

print("************ ONE_VS_REST EXAMPLE:")
print("**********************************\n")

Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=16)

rl_iris_ovr.entrena(Xe_iris,ye_iris)

print("OvR Rendimiento entrenamiento iris: ",rendimiento(rl_iris_ovr,Xe_iris,ye_iris))
print("OvR Rendimiento prueba iris: ",rendimiento(rl_iris_ovr,Xp_iris,yp_iris))
print("\n\n\n")



print("************ BEST LOGISTIC REGRESSION RESULTS IN CREDIT, IMDB AND DIGITS")
print("*******************************************************************************\n")


# # ATENCIÓN: EN CADA CASO, USAR LA MEJOR COMBINACIÓN DE HIPERPARÁMETROS QUE SE HA 
# # DEBIDO OBTENER EN EL PROCESO DE AJUSTE

print("==== BEST LR RESULTS ON VOTES:")
RL_VOTOS=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=32) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_VOTOS.entrena(Xe_votos,ye_votos,n_epochs=6) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre votos: ",rendimiento(RL_VOTOS,Xe_votos,ye_votos))
print("Rendimiento RL test sobre votos: ",rendimiento(RL_VOTOS,Xp_votos,yp_votos))
print("\n")


print("==== BEST LR RESULTS ON BREAST CANCER:")
RL_CANCER=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_CANCER.entrena(Xe_cancer,ye_cancer,n_epochs=10) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre cáncer: ",rendimiento(RL_CANCER,Xe_cancer,ye_cancer))
print("Rendimiento RL test sobre cancer: ",rendimiento(RL_CANCER,Xp_cancer,yp_cancer))
print("\n")


print("==== BEST LR_OvR RESULTS ON CREDITS:")
X_credito_oh=codifica_one_hot(X_credito)
Xe_credito_oh,Xp_credito_oh,ye_credito,yp_credito=particion_entr_prueba(X_credito_oh,y_credito,test=0.3)

RL_CLASIF_CREDITO=RL_OvR(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_CLASIF_CREDITO.entrena(Xe_credito_oh,ye_credito,n_epochs=20) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RLOVR  entrenamiento sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xe_credito_oh,ye_credito))
print("Rendimiento RLOVR  test sobre crédito: ",rendimiento(RL_CLASIF_CREDITO,Xp_credito_oh,yp_credito))
print("\n")


print("==== BEST LR RESULTS ON IMDB:")
RL_IMDB=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_IMDB.entrena(X_train_imdb,y_train_imdb,n_epochs=20) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre imdb: ",rendimiento(RL_IMDB,X_train_imdb,y_train_imdb))
print("Rendimiento RL test sobre imdb: ",rendimiento(RL_IMDB,X_test_imdb,y_test_imdb))
print("\n")


print("==== BEST LR RESULTS ON DIGITS:")
RL_DG=RL_OvR(rate=0.1,rate_decay=True,batch_tam=64) # ATENCIÓN: sustituir aquí por los mejores parámetros tras el ajuste
RL_DG.entrena(X_entr_dg,y_entr_dg,n_epochs=20) # Aumentar o disminuir los epochs si fuera necesario
print("Rendimiento RL entrenamiento sobre dígitos: ",rendimiento(RL_DG,X_entr_dg,y_entr_dg))
print("Rendimiento RL validación sobre dígitos: ",rendimiento(RL_DG,X_val_dg,y_val_dg))
print("Rendimiento RL test sobre dígitos: ",rendimiento(RL_DG,X_test_dg,y_test_dg))
