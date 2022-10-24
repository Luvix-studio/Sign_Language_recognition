import numpy as np
from activation import Activation

class Tanh(Activation):#fonction de base pour des choix simples, issue de sigmoide
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoide(Activation): #permet de "casser" l'erreur
    def __init__(self):
        def sigmoide(x):
            return 1 / (1 + np.exp(-x))

        def sigmoide_prime(x):
            s = sigmoide(x)
            return s * (1 - s)

        super().__init__(sigmoide, sigmoide_prime)#permet de ne pas se
                                                  #referer à la classe de base
        #évite ainsi un multi-appel d'une même classe et
        #donc de prendre plus d'éspace mémoire

class ReLu(Activation):
    def __init__(self):
        def ReLu(x):
            maximum = []
            for i in x:
                maximum.append([max(0.0, i[0])])
            return np.array(maximum)
        
        def ReLu_prime(x):
            liste = []
            for i in x:
                if i[0] > 0:
                    liste.append([1])
                else:
                    liste.append([0])
            return np.array(liste)
        super().__init__(ReLu, ReLu_prime)

class ELU(Activation):
    def __init__(self):
        def ELU(x):
            
            liste = []
            for i in x:
                if i[0]<0:
                    liste.append([np.exp(i[0])-1])
                else:
                    liste.append([i[0]])
            
            return np.array(liste)

        
        def ELU_prime(x):
            
            liste = []
            for i in x:
                if i[0]<0:
                    liste.append([np.exp(i[0])])
                else:
                    liste.append([1])
            
            return np.array(liste)
        
        super().__init__(ELU,ELU_prime)
