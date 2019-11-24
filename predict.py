import keras
import cv2


trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1 / 255.0)
valAug = ImageDataGenerator(rescale=1 / 255.0)
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=1)
model = load_model('classifier.h5')
INIT_LR=1e-1
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

totalTest = len(list(paths.list_images(config.TEST_PATH)))
print(totalTest)
pred = model.predict_generator(testGen,
	steps=(totalTest) + 1)

print(pred)
