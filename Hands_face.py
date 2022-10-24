import cv2
import mediapipe as mp

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    with mpHands.Hands(static_image_mode=False, 
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7,
                           max_num_hands=2
                       ) as hands:
        
        #
        while True:
            success, image = cap.read()

            # Flip the image horizontally and convert the color space from BGR to RGB
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            # To improve performance
            image.flags.writeable = False
            
            # Detect the face landmarks
            results = face_mesh.process(image)
            results_hand = hands.process(image)

            # To improve performance
            image.flags.writeable = True

            # Convert back to the BGR color space
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw the face mesh annotations on the image.
            if results.multi_face_landmarks:
              for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
            if results_hand.multi_hand_landmarks:
                for hand_no, handLandmarks in enumerate(results_hand.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image,
                        handLandmarks,
                        mpHands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Display the image
            cv2.imshow('MediaPipe FaceMesh', image)
            
            # Terminate the process
            if cv2.waitKey(5) & 0xFF == 27:
              break

            
cv2.destroyAllWindows()
cap.release()
