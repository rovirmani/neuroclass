
import matplotlib
matplotlib.use("Agg")

import keras
model = ResNet.build(64, 128, 4, 4,
	(64, 128, 256, 512), reg=0.00010)
opt = SGD(lr=1e-2, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
j = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
	callbacks=callbacks)
dicts = model.predict_generator(testGen,
	steps=totalTest + 1)

dicts = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

model.save('classifier.h5')

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
