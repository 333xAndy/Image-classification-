Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
#import packages
>>> import tensorflow as tf
>>> from tensorflow import keras
>>> from tensorflow.keras import layers
>>> import os
>>> os.getcwd()
'your current directory'
>>> os.chdir('directory to change to')
>>> num_skipped = 0
>>> for folder_name in ("Cat", "Dog"):
	folder_path = os.path.join("PetImages", folder_name)
	for fname in os.listdir(folder_path):
		fpath = os.path.join(folder_path, fname)
		try:
			fobj = open(fpath, "rb")
			#filtering out badly-encoded images not featuring "JFIF" header
			is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
		finally:
			fobj.close()
		if not is_jfif:
			num_skipped += 1
			os.remove(fpath)

	      
>>> print("deleted %d images" % num_skipped)
deleted 1578 images

#generating datasets
>>> image_size = (180, 180)
>>> batch_size = 32

#raw data from disk to a Dataset(tf) object that can be used to train
>>> train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"PetImages",
	validation_split=0.2,
	subset="training",
	seed=1337,
	image_size=image_size,
	batch_size=batch_size,
)
Found 23422 files belonging to 2 classes.
Using 18738 files for training.
#validation set would be structured the same way.
>>> val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	"PetImages",
	validation_split=0.2,
	subset="validation"
	,
	seed=1337,
	image_size=image_size,
	batch_size=batch_size,
)
Found 23422 files belonging to 2 classes.
Using 4684 files for validation.

#for MATLAB
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(10,10))
<Figure size 1000x1000 with 0 Axes>

>>> plt.figure(figsize=(10,10))
<Figure size 1000x1000 with 0 Axes>
>>> for images, labels in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3,3,i+1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(int(labels[i]))
		plt.axis("off")

		
<matplotlib.image.AxesImage object at 0x000001AAD820B760>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7C83FD0>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7C6BC10>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7CA7910>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7CE41C0>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7D34CD0>
Text(0.5, 1.0, '1')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7D00460>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7D006D0>
Text(0.5, 1.0, '0')
(-0.5, 179.5, 179.5, -0.5)
<matplotlib.image.AxesImage object at 0x000001AAD7D70AC0>
Text(0.5, 1.0, '1')
(-0.5, 179.5, 179.5, -0.5)

#Using transformations to augment data. This will give greater diversity
#to the sample. GOOD for smaller sample sizes
#using sequential model for linear stack of layers
>>> data_augmentation = keras.Sequential(
	[
		layers.experimental.preprocessing.RandomFlip("horizontal"),
		layers.experimental.preprocessing.RandomRotation(0.1),
	]
)

>>> train_ds = train_ds.prefetch(buffer_size=32)
>>> val_ds = val_ds.prefetch(buffer_size=32)

>>> def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

>>> model = make_model(input_shape=image_size + (3,), num_classes=2)

#number of trials
>>> epochs = 50
>>> callbacks = [
	keras.callbacks.ModelCheckPoint("save_at_{epoch}.h5"),
]

>>> model.compile(
	optimizer=keras.optimizers.Adam(1e-3),
	loss="binary_crossentropy",
	metrics=["accuracy"],
)
>>> model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

#training happens here

>>> img = keras.preprocessing.image.load_img(
	"PetImages/Cat/6779.jpg", target_size=image_size
)

>>> img_array = keras.preprocessing.image.img_to_array(img)
>>> img_array = tf.expand_dims(img_array,0)
>>> predictions = model.predict(img_array)
>>> score = predictions[0]
>>> print(
	"This image is %.2f precent cat and %.2f percent dog."
	% (100 * (1 - score), 100 * score)
)
This image is 51.66 precent cat and 48.34 percent dog.

#Training was stopped premature, had training not been interupted
#would have gotten accurate reading
