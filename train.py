import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle
import sys

from dense import Dense
from activations import Tanh, Sigmoide, ReLu, ELU
from pertes import mse, mse_prime
import time

try :
    df = pd.read_pickle("DataSet/WfullDataSet.pkl") #on décompile les données
    data = np.array(df)
    np.random.shuffle(data) #on mélange les données pour éviter un underfit
    n, m = df.shape

    X = data[:,:-1]         #on extrait les coordonnées de la main pour chaque échantillion
    Y = data[:,m-1]         #on extrait la lettre correspondante

    X = np.reshape(X, (n, 63, 1))
    Y = np.reshape(Y, (n, 1, 1))

except Exception:
    print("Une erreur de lecture de données est survenue")


def sauvegarder_reseau(input):#fonction pour sauvegarder les poids
    print("Tentative de sauvegarde...")
    output = input
    n = 0
    try:
        for couche in reseau:
            n+=1
            #on collècte les matrices de poids
            temp_poids, temp_bias = couche.sauvegarde()
            #on enlève toutes les entrées vides
            if np.all(temp_poids) != None:
                np.savetxt("CNN/poids_"+str(n)+".CNN", temp_poids)
                np.savetxt("CNN/poids_"+str(n)+".exCNN", temp_bias)
        print("L'état du réseau a bien été sauvegardé")
    except Exception:
        print("Une erreur est survenue, le réseau n'a pas pu être sauvegardé")

def prediction(input):
    sortie = input
    for couche in reseau:
        sortie = couche.propagation_av(sortie)
    return sortie

def train(reseau, pertes, pertes_prime, x_train, y_train,
          sv =False, epochs = 1000, taux_apprentisage = 0.8, verbose = True):
    
    for e in range(epochs):
        erreur = 0
        for x, y in zip(x_train, y_train):
            
            # propagation avant
            sortie = prediction(x)

            # erreur
            erreur += pertes(y, sortie)

            # propagation arrière
            grad = pertes_prime(y, sortie)
            
            for couche in reversed(reseau):
                #application de la descente de gradient
                grad = couche.propagation_ar(grad, taux_apprentisage)
                
        erreur /= len(x_train)
        
        if verbose:
            print(f"{e + 1}/{epochs}, erreur={erreur}")
            
    if sv == True:
        print("Sauvegarde automatique :")
        sauvegarder_reseau(list(X[0])) #sauvegarde l'état actuel du reseau de neurones
    else:
        print("Aucune sauvegarde du réseau n'a été faite")


def test_prediction(n):
    print(prediction(list(X[n])))#Résultat 
    print(Y[n]) #résultat théorique attendu


# entrainement

nb_iter = input("Nombre d'itérations :")
try:
    nb_iter = int(nb_iter)
except Exception():
    sys.exit("Erreur : ce n'est pas un nombre entier")

sv=False    
if input("Sauvegarder le réseau après entrainement ? o/n") == "o":
    sv = True

if input("Continuer en utilisant les données sauvegardées ? o/n") == "o":
    try:
        print("Tentative de récupération...")
        #on tente de charger les données sauvegardées du reseau de neurones
        poids_1 = np.loadtxt('CNN/poids_1.CNN') 
        poids_3 = np.array([np.loadtxt('CNN/poids_3.CNN')][0])
        poids_5 = np.array([np.loadtxt('CNN/poids_5.CNN')][0])

        #Extraction de l'état du réseau de neurones
        bias_1 = np.array([np.loadtxt('CNN/poids_1.exCNN')]) 
        bias_3 = np.array([np.loadtxt('CNN/poids_3.exCNN')])
        bias_5 = np.array([np.loadtxt('CNN/poids_5.exCNN')])
       
        bias_1 = np.reshape(bias_1[0],(1,30,1))[0] #on remet en forme les données
        bias_3 = np.reshape(bias_3[0],(1,12,1))[0]
       

        #on charge les valeurs sauvegardées lors des apprentissages precedents
        reseau = [
            Dense(None, None,poids_1, bias_1), 
            Tanh(),
            Dense(None,None, poids_3, bias_3),
            ELU(),
            Dense(None,None, poids_5, bias_5),
            Sigmoide()]
        print("Données récupérées\n")

    except Exception:
        print("Aucune donnée n'est disponible les données basiques seront initialisées")
        reseau = [
            Dense(63, 30),
            Tanh(),
            Dense(30, 12),
            ELU(),
            Dense(12, 1),
            Sigmoide()]
else:
    print("<default>")
    reseau = [
        Dense(63, 30), #entree : 63 neurones - couche 1 : 30 neurones
        Tanh(),        #fonction d'activation pour accelerer l'apprentissage
        Dense(30, 12), #couche 2 : 12 neurones
        ELU(),        #fonction d'activation permettant de fortement reduire l'erreur
        Dense(12, 1),  #sortie : 1  neurone
        Sigmoide()]    #fonction de sortie pour une certitude entre 0 et 100%


train(reseau, mse, mse_prime, X, Y,sv, epochs=nb_iter, taux_apprentisage=0.05)
