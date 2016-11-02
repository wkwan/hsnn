import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import itertools
from create_dataset import create

model, int_to_card, card_to_int, Xs, train_outputs_all, input_len = create()

model.compile(loss='categorical_crossentropy', optimizer='adam')
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

for epoch in range(10):
	for i in range(len(Xs)):
		train_outputs = train_outputs_all[i]
		y = np_utils.to_categorical(train_outputs, len(int_to_card))		
		X = Xs[i]
		print("SHAPE ", X.shape, y.shape)
		model.fit(X, y, nb_epoch=1, batch_size=256, callbacks=callbacks_list)



