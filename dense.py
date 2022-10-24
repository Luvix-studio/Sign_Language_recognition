import numpy as np

class Dense():
    def __init__(self, taille_entree, taille_sortie, poids = None, bias = None):
        
        self.b = list()
        self.w = list()
        if np.all(poids) != None and np.all(bias) != None:
            #permet de charger l'état sauvegardé du réseau de neurones
            self.poids = poids #on définit les matrices de poids et bias
            self.bias = bias
        else:
            #on definit les matrices de poids de façon aléatoire pour débutter l'apprentisage
            self.poids = np.random.randn(taille_sortie, taille_entree) 
            self.bias = np.random.randn(taille_sortie, 1)

    def propagation_av(self, input):
        self.entree = input
        return np.dot(self.poids, self.entree) + self.bias

    def propagation_ar(self, sortie_gradient, taux_apprentisage):
        poids_gradient = np.dot(sortie_gradient, self.entree.T)
        entree_gradient = np.dot(self.poids.T, sortie_gradient)
        
        self.poids -= taux_apprentisage * poids_gradient
        self.bias -= taux_apprentisage * sortie_gradient
        
        self.b.append(self.bias)
        self.w.append(self.poids)
        return entree_gradient

    def sauvegarde(self):
        #on renvoie les matrices poids et bias pour les sauvegarder
        return(self.poids, self.bias)


