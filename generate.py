import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools
from create_dataset import create

model, int_to_card, card_to_int, X, y, input_len = create()
filename = "weights-improvement-19-4.3681.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# # pick a random seed
# test_input = [numpy.random.randint(0, len(int_to_card)), numpy.random.randint(0, len(int_to_card))]

test_input_text = ["Mounted Raptor LoE", "Savage Roar"]
test_input = list(card_to_int[card] for card in test_input_text)

generated_deck = [] + test_input
for i in range(30 - input_len):
	test_input = numpy.reshape(test_input, (1, len(test_input), 1))
	test_input = test_input/float(len(int_to_card))
	prediction = model.predict(test_input)
	card_int = numpy.argmax(prediction)
	test_input = [generated_deck[numpy.random.randint(0, len(generated_deck))], card_int]
	generated_deck.append(card_int)

for card_int in generated_deck:
	print(int_to_card[card_int])
# pattern = dataX[start]
# print "Seed:"
# print "\"", ''.join([int_to_char[value] for value in pattern]), "\""

# #Pick 100 chars from the text as seed, then generate the next one. Then use that new char plus the 99 prev chars as the new seed. Do this 1000 times.
# # generate characters
# for i in range(1000):
# 	x = numpy.reshape(pattern, (1, len(pattern), 1))
# 	x = x / float(n_vocab)
# 	prediction = model.predict(x, verbose=0)
# 	index = numpy.argmax(prediction)
# 	result = int_to_char[index]
# 	seq_in = [int_to_char[value] for value in pattern]
# 	sys.stdout.write(result)
# 	pattern.append(index)
# 	pattern = pattern[1:len(pattern)]
# print "\nDone."