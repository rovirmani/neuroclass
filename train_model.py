
import matplotlib

import keras
model = ResNet.build(64, 128, 4, 4,
	(64, 128, 256, 512), reg=0.00010)
opt = SGD(lr=1e-2, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
j = model.fit_generator(#fit gens here )
dicts = model.predict_generator(#gensteps)

dicts = np.argmax(d, axis=1)
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

model.save('classifier.h5')

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
