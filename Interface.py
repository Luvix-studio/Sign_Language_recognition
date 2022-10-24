import prediction

import cv2
import mediapipe
import pandas as pd
import numpy as np

def video():

    mpDraw = mediapipe.solutions.drawing_utils
    mpHands = mediapipe.solutions.hands
    dictionnaire = {0:"A",1:"B",2:"C",3:"D",4:"E",5:"F",6:"G"
                    ,7:"H",8:"I",9:"J",10:"K",11:"L",12:"M",13:"N"
                    ,14:"O",15:"P",16:"Q",17:"R",18:"S",19:"T",20:"U"
                    ,21:"V",22:"W",23:"X",24:"Y",25:"Z"}
    lettres = ["A","B","C","D","E","F","G","H", "I", "J","K","L","M","N"
               ,"O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    derniere_lettre = ""

    capture = cv2.VideoCapture(0) #on demarre le prise de vidéo en direct 
        
    with mpHands.Hands(static_image_mode=False, 
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7,
                           max_num_hands=1) as hands:#on lance la reconnaissnace des mains
        ret, frame = capture.read() #on lit une première image pour initialiser 
        image_width =  (frame).shape[0] #on prend mes dimensions image maintenant 
        image_height = (frame).shape[1] #pour ne pas les redefinir dans la boucle
            
        while (True): 
            ret, frame = capture.read() #on lit les images de la caméra
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(frame) #on fait le calcul de detection des mains
                    
            if results_hand.multi_hand_landmarks != None:
                for hand_no, handLandmarks in enumerate(results_hand.multi_hand_landmarks):
                    mpDraw.draw_landmarks(frame, handLandmarks) #on dessine les points de la main
                    donnee = []
                    for line in str(results_hand.multi_hand_landmarks[0]).split("\n"):
                        #on retire tout les caractères de formatage
                        if line != 'landmark {' and line !='}' and line !="": 
                            donnee.append([float(line.strip()[3:])]) #on enlève les derniers caractères

                    prediction_lettre = list()
                    for lettre in lettres:
                        predire = prediction.predire(lettre)
                        prediction_lettre.append(predire.predict(np.array(donnee))[0])
                        #on cherche à avoir 99% de certitude pour la lettre la plus certaine parmis les autres 
                    if max(prediction_lettre) >= 0.9: 
                        if derniere_lettre != dictionnaire[prediction_lettre.index(max(prediction_lettre))]:
                            derniere_lettre = dictionnaire[prediction_lettre.index(max(prediction_lettre))]
                            print(dictionnaire[prediction_lettre.index(max(prediction_lettre))], end='')
                    
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Test de suivi corporel', frame)#noir

            if cv2.waitKey(1) & 0xFF == 27: #appuis de [ECHAP] pour arreter l'acquisition de la vidéo
                break

    cv2.destroyAllWindows()
    capture.release()


video()
