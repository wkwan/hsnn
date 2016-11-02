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


    print(len(unique_cards))


    card_to_int = dict((c, i) for i, c in enumerate(unique_cards))
    int_to_card = dict((i, c) for i, c in enumerate(unique_cards))

    input_len = 4;
    output_len = 1;
    total_len = input_len + output_len

    train_inputs = []
    train_outputs = []

    train_inputs_all = []
    train_outputs_all = []

    for deck in decks:
        #Should use permutations if input order matters, but use combinations to save time
        combinations = itertools.combinations((card_to_int[card] for index, card in enumerate(deck)), total_len)
        for combination in combinations:
            train_inputs.append(combination[:input_len])
            train_outputs.append(combination[input_len:])

            if len(train_inputs) >= 10000:
                train_inputs_all.append(train_inputs)
                train_outputs_all.append(train_outputs)
                train_inputs = []
                train_outputs = []

        # deck_ints = list(card_to_int[card] for card in deck)
        # # print(deck_ints)
        # for i in range(30 - total_len):
        # 	train_inputs.append(deck_ints[i:i+input_len])
        # 	train_outputs.append(deck_ints[i+input_len:i+total_len])


    print("num train outputs", len(train_outputs_all) * len(train_outputs_all[0]))

    Xs = []
    ys = []

    for i in range(len(train_inputs_all)):
        print("ok alloc")
        train_inputs = train_inputs_all[i]
        train_outputs = train_outputs_all[i]
        X = numpy.reshape(train_inputs, (len(train_inputs), input_len, 1))
        X = X / float(len(unique_cards))
        Xs.append(X)
        # y = np_utils.to_categorical(train_outputs, len(unique_cards))
        # ys.append(y)
    # print(y.shape[1])
    # print("*****\n\n\n\n\n")
    # print(train_outputs)


    print(Xs[0].shape)

    model = Sequential()
    model.add(LSTM(32, input_shape=(Xs[0].shape[1], Xs[0].shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(len(unique_cards), activation='softmax'))

    return model, int_to_card, card_to_int, Xs, train_outputs_all, input_len