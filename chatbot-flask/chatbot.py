from flask import Flask
from flask import request
from flask import Response
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

n_steps = 15 #maxlenY
cardinality = 6569
n_features = 6569
maxlenX = 15;
maxlenY = 15;

app = Flask("a")


@app.route("/")
def hello():
    return "Hello world!"
	
@app.route('/chat', methods=['POST'])
def api_message():
	question = request.data.decode("utf-8") 
	sequence = encode(question, vocabulary_enc_to_int, n_features)
	target = predict_sequence2(infenc, infdec, sequence, int_to_vocabulary)
	resp = Response(target, status=200, mimetype='text/plain')
	return resp
	



#source = X1

X1, X2, y = list(), list(), list()
Xsrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-enc3.csv')
Ysrc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\df-dec3.csv')

train_enc = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\train-enc.csv', sep=";")
train_dec = pd.read_csv('D:\\Fakultet\\Master\\SIA\\results\\train-dec.csv', sep=";")

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

train.load_weights('train-weights.h5')
infenc.load_weights('infenc-weights.h5')
infdec.load_weights('infdec-weights.h5')
print('Loaded models')