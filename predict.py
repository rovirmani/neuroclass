import keras
import cv2
img = cv2.imread('img.png')
model = load_model('classifier.h5')
pred = model.predict_generator(img)
print(pred)
