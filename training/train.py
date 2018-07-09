from random import randint
import numpy as np
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

n_features = 6569
n_sentences = 8000

X1, X2, y = list(), list(), list()
#Xsrc = pd.read_csv('/df-enc22/df-enc22.csv')
#Ysrc = pd.read_csv('/df-enc22/df-dec22.csv')
Xsrc = pd.read_csv('/df-enc3/df-enc3.csv')
Ysrc = pd.read_csv('/df-enc3/df-dec3.csv')
#Xsrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-enc3.csv')
#Ysrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-dec3.csv')
print('Ucitao')
maxlenX = 15
maxlenY = 15
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

def get_dataset(n_in, n_out, cardinality, n_samples):
		X1, X2, y = list(), list(), list()
		i = 0
		for row in Xsrc.loc[1:n_sentences, 'Text']:
				#Xrow = [int(num) for num in row.split()]
				Xrow = []
				for num in row.split():
					elem = int(num.replace(".0",""))
					Xrow.append(elem)
				if len(Xrow) < maxlenX:
					for i in range(1, maxlenX-len(Xrow)+1):
						Xrow.append(0)
				src_encoded = to_categorical([Xrow], num_classes=cardinality)
				X1.append(src_encoded)
				print(i)
				i+=1
		i = 0
		for row in Ysrc.loc[1:n_sentences, 'Text']:
				#Yrow = [int(num) for num in row.split()]
				Yrow = []
				for num in row.split():
					elem = int(num.replace(".0",""))
					Yrow.append(elem)
				if len(Yrow) < maxlenY:
					for i in range(1, maxlenY-len(Yrow)+1):
						Yrow.append(0)
				trg_encoded = to_categorical([Yrow], num_classes=cardinality)
				y.append(trg_encoded)
				X2row = [0] + Yrow[:-1]
				trg2_encoded = to_categorical([X2row], num_classes=cardinality)
				X2.append(trg2_encoded)
				print(i)
				i+=1
		X1 = np.squeeze(array(X1), axis=1) 
		X2 = np.squeeze(array(X2), axis=1) 
		y = np.squeeze(array(y), axis=1) 
		return X1, X2, y
		
# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = Input(shape=(None, n_input))
	encoder = LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = Input(shape=(None, n_output))
	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = Input(shape=(n_units,))
	decoder_state_input_c = Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

# configure problem
n_steps_in = maxlenX #maxlen od pitanja
n_steps_out = maxlenY #maxlen od odgovora
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
print('Definisao modele')
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print('Kompajlovao')

# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100)
print('Napravio X,Y')

print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=30)
print('Fitted...')
print('Saving models...')
train.save_weights("/output/train-weights.h5", overwrite=True)
infenc.save_weights("/output/infenc-weights.h5", overwrite=True)
infdec.save_weights("/output/infdec-weights.h5", overwrite=True)

train_json = train.to_json()
with open("/output/train.json", "w") as json_file:
    json_file.write(train_json)
infenc_json = infenc.to_json()
with open("/output/infenc.json", "w") as json_file:
    json_file.write(infenc_json)
infdec_json = infdec.to_json()
with open("/output/infdec.json", "w") as json_file:
    json_file.write(infdec_json)
print('Saved models.')
