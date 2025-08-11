import cv2
from src.pca_model import load_pca
from src.classifier import load_clf
from src.preprocess import load_lfw

def face_detector(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces, gray

if __name__ == '__main__':
    pca = load_pca('models/pca.joblib')
    clf = load_clf('models/clf.joblib')
    _, _, target_names, img_shape = load_lfw(min_faces_per_person=50, resize=0.4)
    h, w = img_shape

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces, gray = face_detector(frame, face_cascade)
        for (x, y, fw, fh) in faces:
            face_img = gray[y:y+fh, x:x+fw]
            face_resized = cv2.resize(face_img, (w, h)).reshape(1, -1)
            face_pca = pca.transform(face_resized)
            pred = clf.predict(face_pca)[0]
            name = target_names[pred]
            cv2.rectangle(frame, (x, y), (x+fw, y+fh), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow('PCA Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
