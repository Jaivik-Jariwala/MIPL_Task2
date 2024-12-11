import face_recognition
import cv2

class FaceRecognizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def predict(self, frame):
        """Recognize faces in a video frame."""
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        predictions = []

        for face_encoding in face_encodings:
            matches = []
            for unique_id, data in self.encodings.items():
                is_match = face_recognition.compare_faces([data['encoding']], face_encoding)
                if is_match[0]:
                    matches.append(data['name'])

            name = matches[0] if matches else "UNKNOWN"
            predictions.append(name)

        return face_locations, predictions

    @staticmethod
    def draw_boxes(frame, face_locations, names):
        """Draw bounding boxes and labels on the frame."""
        for (top, right, bottom, left), name in zip(face_locations, names):
            # Draw the box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Label the box
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        return frame
