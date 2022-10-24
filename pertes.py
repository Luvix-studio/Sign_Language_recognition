import numpy as np

def mse(y_attendu, y_pred): #fonction indicatrice de l'écart entre deux valeurs
    return np.mean(np.power(y_attendu - y_pred, 2))#le carré moyen des erreurs

def mse_prime(y_attendu, y_pred):
    return 2 * (y_pred - y_attendu) / np.size(y_attendu)
