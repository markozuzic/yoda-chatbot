from random import randint
import numpy as np
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import model_from_json

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

n_steps = 15 #maxlenY
cardinality = 6569
n_features = 6569
maxlenX = 15;
maxlenY = 15;
#source = X1

X1, X2, y = list(), list(), list()
Xsrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-enc3.csv')
Ysrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-dec3.csv')

train_enc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\train-enc.csv', sep=";")
train_dec = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\train-dec.csv', sep=";")

mx = []
my=[]
for row in Xsrc.loc[1:50,'Text']:
	n = 0
	for num in row.split():
		n+=1
	mx.append(n)

maxlenX = max(mx)

Y=list()
for row in Ysrc.loc[1:50,'Text']:
	n = 0
	for num in row.split():
		n+=1
	my.append(n)

maxlenY = max(my)
if maxlenY < maxlenX:
	maxlenY = maxlenX
else:
	maxlenX = maxlenY
print('Datasets ready...')

def create_vocabulary():
	vocabulary_enc = {}
	for sentence in train_enc.loc[:,'train_enc2']:
		for word in sentence.split():
			if word not in vocabulary_enc:
				vocabulary_enc[word] = 1
			else:
				vocabulary_enc[word] += 1

	vocabulary_dec = {}
	for sentence in train_dec.loc[:,'train_dec2']:
		for word in sentence.split():
			if word not in vocabulary_dec:
				vocabulary_dec[word] = 1
			else:
				vocabulary_dec[word] += 1
	
	vocabulary_enc_to_int = {}
	vocabulary_dec_to_int = {}
	codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']
	word_num = 0
	threshold = 4

	for code in codes:
		vocabulary_enc_to_int[code] = word_num
		word_num += 1

	for word, count in vocabulary_enc.items():
		if count >= threshold:
			vocabulary_enc_to_int[word] = word_num
			word_num += 1
        
        
	word_num = 0
	for code in codes:
		vocabulary_dec_to_int[code] = word_num
		word_num += 1

	for word, count in vocabulary_dec.items():
		if count >= threshold:
			vocabulary_dec_to_int[word] = word_num
			word_num += 1
			
	int_to_vocabulary = {}
	for word, num in vocabulary_dec_to_int.items():
		int_to_vocabulary[num] = word
	
	return vocabulary_enc_to_int, vocabulary_dec_to_int, int_to_vocabulary

def get_dataset(n_in, n_out, cardinality, n_samples):
		X1, X2, y = list(), list(), list()
		for row in Xsrc.loc[1:3, 'Text']:
				Xrow = [int(num) for num in row.split()]
				if len(Xrow) != maxlenX:
					for i in range(1, maxlenX-len(Xrow)+1):
						Xrow.append(0)
				src_encoded = to_categorical([Xrow], num_classes=cardinality)
				#X1.append([Xrow])
				X1.append(src_encoded)
		X2row = np.zeros(maxlenX)
		src_encoded = to_categorical([X2row], num_classes=cardinality)
		X2.append(src_encoded)
		for row in Xsrc.loc[2:3, 'Text']:
				X2row = [int(num) for num in row.split()]
				if len(X2row) != maxlenX:
					for i in range(1, maxlenX-len(X2row)+1):
						X2row.append(0)
				src_encoded = to_categorical([X2row], num_classes=cardinality)
				#X1.append([Xrow])
				X2.append(src_encoded)
		for row in Ysrc.loc[1:3, 'Text']:
				Yrow = [int(num) for num in row.split()]
				if len(Yrow) != maxlenY:
					for i in range(1, maxlenY-len(Yrow)+1):
						Yrow.append(0)
				trg_encoded = to_categorical([Yrow], num_classes=cardinality)
				y.append(trg_encoded)
		X2=X1
		X1 = np.squeeze(array(X1), axis=1) 
		X2 = np.squeeze(array(X2), axis=1) 
		y = np.squeeze(array(y), axis=1) 
		#return array(X1), array(X2), array(y) 
		return X1, X2, y

def decode(sequence, int_to_vocabulary):
	sen = []
	for num in sequence:
		word = int_to_vocabulary[num]
		sen.append(word)
	return sen
	
def encode(sentence, vocabulary_enc_to_int, cardinality):
	seq = list()
	for word in sentence.split():
		seqRow = []
		if word not in vocabulary_enc_to_int:
			seqRow = [2]
		else:
			seqRow = [vocabulary_enc_to_int[word]]
		if len(seqRow) != maxlenX:
			for i in range(1, maxlenX-len(seqRow)+1):
				seqRow.append(0)
		seq_encoded = to_categorical([seqRow], num_classes=cardinality)
		seq.append(seq_encoded)
	seq = np.squeeze(array(seq), axis=1) 
	return seq
		
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
	# encode
	state = infenc.predict(source)
	# start of sequence input
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
	output = list()
	for t in range(n_steps):
		# predict next char
		yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
		output.append(yhat[0,0,:])
		# update state
		state = [h, c]
		# update target sequence
		target_seq = yhat
	return array(output)
	
def predict_sequence2(encoder_model, decoder_model, input_seq, int_to_vocabulary):
	states_value = encoder_model.predict(input_seq)
	num_decoder_tokens = cardinality
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	stop_condition = False
	decoded_sentence = ''
	i = 0
	while not stop_condition:
		output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		if sampled_token_index != 2:
			sampled_char = int_to_vocabulary[sampled_token_index]
			if sampled_token_index != 0 :
				decoded_sentence += sampled_char
				decoded_sentence += ' '

        # Exit condition: either hit max length or find stop character.
		if sampled_char == '<PAD>':
			stop_condition = True
        # Update the target sequence
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.

        # Update states
		states_value = [h, c]

	return decoded_sentence

vocabulary_enc_to_int, vocabulary_dec_to_int, int_to_vocabulary = create_vocabulary()
print('Created vocabularies...')

#X1, X2, y = get_dataset(n_steps, n_steps, n_features, 1)

print('Loading models...')

json_file = open('train.json', 'r')
train_json = json_file.read()
json_file.close()
train = model_from_json(train_json)

json_file = open('infenc.json', 'r')
infenc_json = json_file.read()
json_file.close()
infenc = model_from_json(infenc_json)

json_file = open('infdec.json', 'r')
infdec_json = json_file.read()
json_file.close()
infdec = model_from_json(infdec_json)

#train = load_model('train.h5')
#train_weights = load_weights('train-weights.h5')
#train.set_weights(train_weights)
train.load_weights('train-weights.h5')
infenc.load_weights('infenc-weights.h5')
infdec.load_weights('infdec-weights.h5')

#infenc = load_model('infenc.h5')
#infenc_weights = load_model('infenc_weights.h5')
#infenc.set_weights(infenc_weights)

#infdec = load_model('infdec.h5')
#infdec_weights = load_model('infdec_weights.h5')
#infdec.set_weights(infdec_weights)

print('Loaded models...')

#X1, X2, y = get_dataset(n_steps, n_steps, n_features, 1)
while(1):
	question = input(">>")
	sequence = encode(question, vocabulary_enc_to_int, n_features)
	print('Predicting....')
	#target = predict_sequence(infenc, infdec, sequence, n_steps, n_features)
	target = predict_sequence2(infenc, infdec, sequence, int_to_vocabulary)
	print(target)
	
	print('Decoding...')
	#target = one_hot_decode(target)
	#sentence = decode(target, int_to_vocabulary)
	#print(sentence)

	