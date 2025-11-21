import cv2
import mediapipe as mp
import numpy as np


class detectMonkey:
    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)
        self.face = self.mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.draw = mp.solutions.drawing_utils
        self.img_path = {
            'monkey_finger' : "monkey_finger.jpeg",
            'monkey_normal' : "monkey_normal.jpeg",
            'monkey_question': 'monkey_question.webp',
            'monkey_surprise': 'monkey_surprise.jpeg'
        }
        self.img = {}
        for key, path in self.img_path.items():
            img = cv2.imread(path)
            if img is not None:
                self.img[key] = img


        self.current = None


    def detect_monkey_finger(self, hand_landmarks):
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]

        extended = index_tip.y < index_pip.y < index_mcp.y

        other_curl = True
        for tip_id in [12,16,20]:
            tip = hand_landmarks.landmark[tip_id]
            tip_mcp = hand_landmarks.landmark[tip_id - 2]
            if tip.y < tip_mcp.y:
                other_curl = False
                break
        return other_curl and extended

    def detect_monkey_mouth(self, face_landmarks):
        upper = face_landmarks.landmark[13]
        lower = face_landmarks.landmark[14]

        openness = abs(upper.y - lower.y)

        return openness > 0.04

    def detect_monkey_normal(self, face_landmarks):
        upper = face_landmarks.landmark[13]
        lower = face_landmarks.landmark[14]

        openness = abs(upper.y - lower.y)

        return openness < 0.01

    #def detect_monkey_question(self, face_landmarks, hand_landmarks):

    def get_face_bounding_box(self, face_landmarks, frame_width, frame_height):

        x_coords = []
        y_coords = []

        for landmark in face_landmarks.landmark:
            x_coords.append(landmark.x * frame_width)
            y_coords.append(landmark.y * frame_height)

        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        return (x_min, y_min, x_max, y_max)

    def is_finger_touching_face(self, hand_landmarks, face_landmarks, frame_width, frame_height):

        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        finger_x = int(index_tip.x * frame_width)
        finger_y = int(index_tip.y * frame_height)


        x_min, y_min, x_max, y_max = self.get_face_bounding_box(face_landmarks, frame_width, frame_height)

        return x_min <= finger_x <= x_max and y_min <= finger_y <= y_max

    def is_pointer_finger_intersecting_face(self, hand_res, face_res, frame_width, frame_height):
        if not hand_res.multi_hand_landmarks or not face_res.multi_face_landmarks:
            return False
        for hand_landmarks in hand_res.multi_hand_landmarks:
            if self.detect_monkey_finger(hand_landmarks):
                for face_landmarks in face_res.multi_face_landmarks:
                    return self.is_finger_touching_face(hand_landmarks, face_landmarks, frame_width, frame_height)

    def process_frame(self, frame):
        colorFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_res = self.hands.process(colorFrame)
        face_res = self.face.process(colorFrame)

        h, w, c = frame.shape
        
        gest_detect = False
        detected = None

        if self.is_pointer_finger_intersecting_face(hand_res, face_res, w, h):
            gest_detect = True
            detected = 'monkey_question'

        elif hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                if self.detect_monkey_finger(hand_landmarks):
                        gest_detect = True
                        detected = 'monkey_finger'
                        break


        elif face_res.multi_face_landmarks:
            for face_landmarks in face_res.multi_face_landmarks:
                if self.detect_monkey_mouth(face_landmarks):
                    gest_detect = True
                    detected = 'monkey_surprise'
                    break
                elif self.detect_monkey_normal(face_landmarks):
                    gest_detect = True
                    detected = 'monkey_normal'
                    break



        if gest_detect and detected in self.img:
            self.current = detected


        if self.current and self.current in self.img:
            monk_img = self.img[self.current]
            monk_img_resized = cv2.resize(monk_img, (w, h))
            split = np.hstack((frame, monk_img_resized))
        else:
            monk_img = self.img['monkey_normal']
            monk_img_resized = cv2.resize(monk_img, (w, h))
            split = np.hstack((frame, monk_img_resized))


        return split


    def run(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            result = self.process_frame(frame)
            cv2.imshow('frame', result)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.face.close()

if __name__ == "__main__":
    detector = detectMonkey()
    detector.run()





