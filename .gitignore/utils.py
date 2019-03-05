import numpy as np
from scipy.misc import imread
import os
from matplotlib import pyplot as plt
from keras import backend as K
from wordcloud import WordCloud


def load_imgs(path, n=0):
	x = []
	y = []
	char_dict = {}
	lang_dict = {}
	current_y = n

	# Load Alphabet
	for alphabet in os.listdir(path):
		lang_dict[alphabet] = [current_y, None]
		alphabet_path = os.path.join(path, alphabet)

		# Load every character in Alphabet
		for character in os.listdir(alphabet_path):
			char_dict[current_y] = (alphabet, character)
			char_images = []
			character_path = os.path.join(alphabet_path, character)

			# Read All Images for the Character
			for image in os.listdir(character_path):
				image_path = os.path.join(character_path, image)
				image = imread(image_path)
				char_images.append(image)
				y.append(current_y)

			# Append collected images of the character in array x
			try:
				x.append(np.stack(char_images))
			except ValueError as err:
				print(err)
				print("Error Occucred while loading: " + alphabet + '-' + character)

			lang_dict[alphabet][1] = current_y
			current_y = current_y + 1

	y = np.vstack(y)
	x = np.stack(x)

	return x, y, lang_dict


def get_batch_in_pairs(x, batch_size):
	n_characters, n_rendition, size_x, size_y = x.shape

	# Classes to choose images from in this batch
	char_id_list = np.random.choice(n_characters, size=(batch_size,), replace=False)


	# Empty Array to hold the batch
	batch_pairs = [np.zeros((batch_size, size_x, size_y, 1)) for i in range(2)]

	# Empty Array to hold target of the batch
	batch_targets = np.zeros((batch_size, ))

	# Balance dataset to contain half valid pairs and half non-valid pairs
	batch_targets[batch_size // 2:] = 1

	# Creating pairs to feed in batch
	for i in range(0, batch_size):
		# Setting first image in pair

		char_id_0 = char_id_list[i]
		rendition_id_0 = np.random.randint(0, n_rendition)

		# Set the first image of of pair withi rendition = rendition_id and character = char_id
		batch_pairs[0][i, :, :, :] = x[char_id_0, rendition_id_0].reshape(size_x, size_y, 1)

		# Setting second image in pair
		# Get rendition id for second image in pair
		rendition_id_1 = np.random.randint(0, n_rendition)

		# Choose whether you want true pairs or false pairs
		if i >= batch_size // 2:
			#print('Choosing Same Image')
			char_id_1 = char_id_0
		else:
			# Select a new character
			char_id_1 =  (char_id_0 + np.random.randint(1, n_characters)) % n_characters
		# Set second image in pair with rendition= rendition_id_1 and character = character_id_1
		batch_pairs[1][i, :, :, :] = x[char_id_1, rendition_id_1].reshape(size_x, size_y, 1)

	return batch_pairs, batch_targets


def data_generator(x, batch_size=2):
	while True:
		batch_pairs, batch_targets = get_batch_in_pairs(x, batch_size)
		yield (batch_pairs, batch_targets)


def show_batch(batch):
	batch_pairs, batch_target = batch

	fig, axes = plt.subplots(1, 2)

	axes[0].set_title('Image 0')
	axes[0].imshow(batch_pairs[0][0, ...].reshape(105, 105))

	axes[1].set_title('Pair 0 - Image 1')
	axes[1].imshow(batch_pairs[0][1, ...].reshape(105, 105))

	fig, axes = plt.subplots(1, 2)
	axes[0].set_title('Pair 1 - Image 0')
	axes[0].imshow(batch_pairs[1][0, ...].reshape(105, 105))

	axes[1].set_title('Pair 1 - Image 1')
	axes[1].imshow(batch_pairs[1][1, ...].reshape(105, 105))


def gradient_norm(model):
	with K.name_scope('gradient_norm'):
		grads = K.gradients(model.total_loss, model.trainable_weights)
		norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
		return norm


def generate_wordcloud(text):
    wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                          relative_scaling = 1.0,
                          stopwords = {'to', 'of'} # set or space-separated string
                          ).generate(text)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()