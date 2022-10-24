import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pickle

from dense import Dense
from activations import Tanh, Sigmoide, ReLu, ELU
from pertes import mse, mse_prime

class predire():
    
    def __init__(self,lettre):
        #on tente de charger les données sauvegardées du reseau de neurones
        poids_1 = np.loadtxt(f'CNN/{lettre}/poids_1.CNN') 
        poids_3 = np.array([np.loadtxt(f'CNN/{lettre}/poids_3.CNN')][0])
        poids_5 = np.array([np.loadtxt(f'CNN/{lettre}/poids_5.CNN')][0])

        #Extraction de l'état du réseau de neurones
        bias_1 = np.array([np.loadtxt(f'CNN/{lettre}/poids_1.exCNN')]) 
        bias_3 = np.array([np.loadtxt(f'CNN/{lettre}/poids_3.exCNN')]) 
        bias_5 = np.array([np.loadtxt(f'CNN/{lettre}/poids_5.exCNN')])
        
        bias_1 = np.reshape(bias_1[0],(1,30,1))[0] #on remet en forme les données
        bias_3 = np.reshape(bias_3[0],(1,12,1))[0]

        #des 63 neurones d'entrée aux 5 neurones cachés
        self.reseau = [
            Dense(None, None,poids_1, bias_1), 
            Tanh(),
            Dense(None,None, poids_3, bias_3),
            ELU(),
            Dense(None,None, poids_5, bias_5),
            Sigmoide()]


    def predict(self, input):
        sortie = list(input)
        for layer in self.reseau:
            sortie = layer.propagation_av(sortie) 
        return sortie
