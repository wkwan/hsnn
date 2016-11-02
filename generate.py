import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools
from create_dataset import create
from operator import itemgetter
from heapq import nlargest

model, int_to_card, card_to_int, Xs, ys, input_len = create()
filename = "weights-improvement-00-1.4557.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
# # pick a random seed
# test_input = [numpy.random.randint(0, len(int_to_card)), numpy.random.randint(0, len(int_to_card))]

test_input_text = ["Mounted Raptor LoE", "Savage Roar", "Living Roots TGT", "Swipe"]
# test_input_text = ["Earthen Ring Farseer", "Argent Squire", "Bloodmage Thalnos"]
# test_input_text = ["Mounted Raptor LoE", "Mad Scientist Naxx", "Alexstrasza"]
# test_input_text = ["Ice Barrier", "Fireball", "Acolyte of Pain"]

test_input = list(card_to_int[card] for card in test_input_text)

print("Test input indices", test_input)

# generated_deck = [] + test_input
generated_deck = {}
for test_card in test_input:
    generated_deck[test_card] = 1

for i in range(30 - input_len):
    pruned_test_input = numpy.reshape(test_input, (1, len(test_input), 1))
    pruned_test_input = pruned_test_input/float(len(int_to_card))
    prediction = model.predict(pruned_test_input)
    prediction = numpy.argsort(prediction[0])
    print("prediction is ", prediction)
    for i in range(len(prediction) - 1, -1, -1):
        if prediction[i] not in generated_deck or generated_deck[prediction[i]] < 2:
            card_int = prediction[i]
            test_input.append(card_int)
            test_input = test_input[1:]
            print("New test input", test_input)
            # generated_deck.append(card_int)
            if card_int in generated_deck:
                generated_deck[card_int] = 2
            else:
                generated_deck[card_int] = 1
            break
    # card_int = numpy.argmax(prediction)
    # test_input.append(card_int)
    # test_input = test_input[1:]
    # print("New test input", test_input)
    # # generated_deck.append(card_int)
    # if card_int in generated_deck:
    #     generated_deck[card_int] = 2
    # else:
    #     generated_deck[card_int] = 1

for card_int in generated_deck:
    print(int_to_card[card_int], generated_deck[card_int])


