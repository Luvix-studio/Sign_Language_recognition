import numpy as np

class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation #fait appel aux fonctions d'activations
        self.activation_prime = activation_prime

    def propagation_av(self, input):
        self.input = input
        return self.activation(self.input)

    def propagation_ar(self, sortie_gradient, taux_apprentisage):
        #déscente de gradients
        return np.multiply(sortie_gradient, self.activation_prime(self.input))
    
    def sauvegarde(self): #permet d'éviter le plantage lors de la sauvegarde
        return (None,None)
