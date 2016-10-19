import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools

def create():
	lines = open('decks.txt').readlines()

	decks = []


	unique_cards = set()

	cur_deck = []
	for line in lines:
		if not line.strip():
			decks.append(cur_deck)
			cur_deck = []
		else:
			card = line[3:].strip()
			if (line[0] == '2'):
				cur_deck.append(card)
				cur_deck.append(card)
			else:
				cur_deck.append(card)
			unique_cards.add(card)




	card_to_int = dict((c, i) for i, c in enumerate(unique_cards))
	int_to_card = dict((i, c) for i, c in enumerate(unique_cards))

	input_len = 2;
	output_len = 1;
	total_len = input_len + output_len

	train_inputs = []
	train_outputs = []

	for deck in decks:
		#Should use permutations if input order matters, but use combinations to save time
		combinations = itertools.combinations((card_to_int[card] for index, card in enumerate(deck)), total_len)
		for combination in combinations:
			train_inputs.append(combination[:input_len])
			train_outputs.append(combination[input_len:])

	X = numpy.reshape(train_inputs, (len(train_inputs), input_len, 1))
	X = X / float(len(unique_cards))
	y = np_utils.to_categorical(train_outputs)
	# print("*****\n\n\n\n\n")
	# print(train_outputs)

	model = Sequential()
	model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))

	return model, int_to_card, card_to_int, X, y, input_len