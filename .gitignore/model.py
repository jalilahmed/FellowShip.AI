from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Activation, Flatten, LeakyReLU, Lambda
from keras.initializers import RandomNormal
from keras.regularizers import l2
import numpy as np


def initializer(mean=0., std_dev=1., seed=867154):
	return RandomNormal(mean=mean, stddev=std_dev, seed=seed)


def create_model(input_size=(105, 105, 1)):
	model = Sequential()

	model.add(Convolution2D(filters=64, kernel_size=(10, 10), strides=1, padding='valid', input_shape=input_size,
							kernel_initializer=initializer(0, 1e-2), bias_initializer=initializer(0.5, 1e-2),
							kernel_regularizer=l2(2e-4)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D())

	model.add(Convolution2D(filters=128, kernel_size=(7, 7), strides=1, padding='valid',
							kernel_initializer=initializer(0, 1e-2), bias_initializer=initializer(0.5, 1e-2),
							kernel_regularizer=l2(2e-4)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D())

	model.add(Convolution2D(filters=128, kernel_size=(4, 4), strides=1, padding='valid',
							kernel_initializer=initializer(0, 1e-2), bias_initializer=initializer(0.5, 1e-2),
							kernel_regularizer=l2(2e-4)))
	model.add(Activation('relu'))

	model.add(MaxPooling2D())

	model.add(Convolution2D(filters=256, kernel_size=(4, 4), strides=1, padding='valid',
							kernel_initializer=initializer(0, 1e-2), bias_initializer=initializer(0.5, 1e-2),
							kernel_regularizer=l2(2e-4)))
	model.add(Activation('relu'))

	model.add(Flatten())

	model.add(Dense(units=4096, activation = 'sigmoid', kernel_initializer=initializer(0, 0.2),
					bias_initializer=initializer(0.5, 1e-2), kernel_regularizer=l2(1e-3)))
	return model


def get_both_legs(lft, rgt, input_size):
	model = create_model(input_size=input_size)
	left_encoding = model(lft)
	right_encoding = model(rgt)
	return left_encoding, right_encoding


def create_siamese(input_size=(105, 105, 1)):
	lft_input = Input(shape=input_size)
	rgt_input = Input(shape=input_size)

	lft_encoding, rgt_encoding = get_both_legs(lft_input, rgt_input, input_size)

	# Create a Layer to Calculate Distance of Feature Map Created
	l1_dist_layer_wrapper = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
	l1_dist_layer = l1_dist_layer_wrapper([lft_encoding, rgt_encoding])

	# get prediction
	prediction = Dense(units=1, activation='sigmoid')(l1_dist_layer)

	return Model(inputs=[lft_input, rgt_input], outputs=[prediction])


def create_one_shot_data(x, n_way):
	n_characters, n_rendition, x_size, y_size = x.shape

	random_char_idx = np.random.choice(range(n_characters), size=(n_way), replace=False)
	random_rendition_idx = np.random.randint(0, n_rendition, size=(n_way + 1,))

	test_char_idx = random_char_idx[0]
	check_char_idx = random_char_idx[1:]

	test_char = np.array([x[test_char_idx, random_rendition_idx[0], :, :]]*n_way).reshape(n_way, x_size, y_size, 1)
	check_char = x[random_char_idx, random_rendition_idx[1:], :, :].reshape(n_way, x_size, y_size, 1)

	test_pair = [test_char, check_char]
	test_target = np.zeros((n_way))
	test_target[0] = 1

	return test_pair, test_target


def one_shot(x, model, e_times=1, n_way=2):
	correct_predictions = 0
	for i in np.arange(0, e_times):
		pairs, targets = create_one_shot_data(x, n_way)
		predictions = model.predict(pairs)
		ground_truth = np.argmax(targets)
		predictions = np.argmax(predictions)
		if ground_truth == predictions:
			correct_predictions = correct_predictions + 1

	return (100. * correct_predictions) / e_times




