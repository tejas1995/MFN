import numpy as np
seed = 123
np.random.seed(seed)
import random
import copy

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import math
import h5py
import time
import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import cPickle as pickle
import time
import json, os, ast, h5py

from keras.models import Model
from keras.layers import Input
from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys

PROJECT_DIR = '/work/tsriniva/MFN/'

def get_data(args,config):
	tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
	val_split = 0.1514                      # fixed. 52 training 10 validation
	use_pretrained_word_embedding = True    # fixed. use glove 300d
	embedding_vecor_length = 300            # fixed. use glove 300d
	# 115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114 
	max_segment_len = config['seqlength']
	end_to_end = True                       # fixed

	word2ix = loader.load_word2ix()
	word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
	train, valid, test = loader.load_word_level_features(max_segment_len, tr_split)

	ix2word = inv_map = {v: k for k, v in word2ix.iteritems()}
	print len(word2ix)
	print len(ix2word)
	print word_embedding[0].shape

	feature_str = ''
	if args.feature_selection:
		with open('/media/bighdd5/Paul/mosi/fs_mask.pkl') as f:
			[covarep_ix, facet_ix] = pickle.load(f)
		facet_train = train['facet'][:,:,facet_ix]
		facet_valid = valid['facet'][:,:,facet_ix]
		facet_test = test['facet'][:,:,facet_ix]
		covarep_train = train['covarep'][:,:,covarep_ix]
		covarep_valid = valid['covarep'][:,:,covarep_ix]
		covarep_test = test['covarep'][:,:,covarep_ix]
		feature_str = '_t'+str(embedding_vecor_length) + '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
	else:
		facet_train = train['facet']
		facet_valid = valid['facet']
		covarep_train = train['covarep'][:,:,1:35]
		covarep_valid = valid['covarep'][:,:,1:35]
		facet_test = test['facet']
		covarep_test = test['covarep'][:,:,1:35]

	text_train = train['text']
	text_valid = valid['text']
	text_test = test['text']
	y_train = train['label']
	y_valid = valid['label']
	y_test = test['label']

	lengths_train = train['lengths']
	lengths_valid = valid['lengths']
	lengths_test = test['lengths']

	#f = h5py.File("out/mosi_lengths_test.hdf5", "w")
	#f.create_dataset('d1',data=lengths_test)
	#f.close()
	#assert False

	facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
	facet_train_max[facet_train_max==0] = 1
	#covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
	#covarep_train_max[covarep_train_max==0] = 1

	facet_train = facet_train / facet_train_max
	facet_valid = facet_valid / facet_train_max
	#covarep_train = covarep_train / covarep_train_max
	facet_test = facet_test / facet_train_max
	#covarep_test = covarep_test / covarep_train_max

	text_input = Input(shape=(max_segment_len,), dtype='int32', name='text_input')
	text_eb_layer = Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_eb_layer', trainable=False)(text_input)
	model = Model(text_input, text_eb_layer)
	text_train_emb = model.predict(text_train)
	print text_train_emb.shape      # n x seq x 300
	print covarep_train.shape       # n x seq x 5/34
	print facet_train.shape         # n x seq x 20/43
	X_train = np.concatenate((text_train_emb, covarep_train, facet_train), axis=2)

	text_valid_emb = model.predict(text_valid)
	print text_valid_emb.shape      # n x seq x 300
	print covarep_valid.shape       # n x seq x 5/34
	print facet_valid.shape         # n x seq x 20/43
	X_valid = np.concatenate((text_valid_emb, covarep_valid, facet_valid), axis=2)

	text_test_emb = model.predict(text_test)
	print text_test_emb.shape      # n x seq x 300
	print covarep_test.shape       # n x seq x 5/34
	print facet_test.shape         # n x seq x 20/43
	X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)

	return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_saved_data():
	h5f = h5py.File(PROJECT_DIR+'mosei_data/X_train.h5','r')
	X_train = h5f['data'][:]
	h5f.close()
	h5f = h5py.File(PROJECT_DIR+'mosei_data/y_train.h5','r')
	y_train = h5f['data'][:]
	h5f.close()
	h5f = h5py.File(PROJECT_DIR+'mosei_data/X_valid.h5','r')
	X_valid = h5f['data'][:]
	h5f.close()
	h5f = h5py.File(PROJECT_DIR+'mosei_data/y_valid.h5','r')
	y_valid = h5f['data'][:]
	h5f.close()
	h5f = h5py.File(PROJECT_DIR+'mosei_data/X_test.h5','r')
	X_test = h5f['data'][:]
	h5f.close()
	h5f = h5py.File(PROJECT_DIR+'mosei_data/y_test.h5','r')
	y_test = h5f['data'][:]
	h5f.close()
	return X_train, y_train, X_valid, y_valid, X_test, y_test

#parser = argparse.ArgumentParser(description='')
#parser.add_argument('--config', default='configs/mosi.json', type=str)
#parser.add_argument('--type', default='mgddm', type=str)	# d, gd, m1, m3
#parser.add_argument('--fusion', default='mfn', type=str)	# ef, tf, mv, marn, mfn
#parser.add_argument('-s', '--feature_selection', default=1, type=int, choices=[0,1], help='whether to use feature_selection')

#args = parser.parse_args()
#config = json.load(open(args.config), object_pairs_hook=OrderedDict)


class MFNPhantom(nn.Module):
	def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig,mode):
		super(MFNPhantom, self).__init__()
		[self.d_l,self.d_a,self.d_v] = config["input_dims"]
		[self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
		total_h_dim = self.dh_l+self.dh_a+self.dh_v
		self.mem_dim = config["memsize"]
		window_dim = config["windowsize"]
		output_dim = 1
		attInShape = total_h_dim*window_dim
		gammaInShape = attInShape+self.mem_dim
		final_out = total_h_dim+self.mem_dim
		h_att1 = NN1Config["shapes"]
		h_att2 = NN2Config["shapes"]
		h_gamma1 = gamma1Config["shapes"]
		h_gamma2 = gamma2Config["shapes"]
		h_out = outConfig["shapes"]
		att1_dropout = NN1Config["drop"]
		att2_dropout = NN2Config["drop"]
		gamma1_dropout = gamma1Config["drop"]
		gamma2_dropout = gamma2Config["drop"]
		out_dropout = outConfig["drop"]

		self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
		self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
		self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

		self.phantom_num_layers = 3
		self.d_v_phantom = 128
		self.d_a_phantom = 128

		self.modality_drop = config["modality_drop"]

		self.phantom_v = nn.LSTM(self.d_l, self.d_v_phantom, num_layers=self.phantom_num_layers)
		self.phantom_a = nn.LSTM(self.d_l, self.d_a_phantom, num_layers=self.phantom_num_layers)
		self.phantom_ff_v = nn.Linear(self.d_v_phantom, self.d_v)
		self.phantom_ff_a = nn.Linear(self.d_a_phantom, self.d_a)
		# Freeze params relevant to phantom acoustic and visual features
		# x_a = x_a.detach()
		# x_v = x_v.detach()
		# frozen_layers = [self.phantom_a, self.phantom_v, self.phantom_ff_a, self.phantom_ff_v]
		# for layer in frozen_layers:
		#  	for param in layer.parameters():
		#  		param.requires_grad = False

		if mode == 'PhantomIntermD':
			self.phantom_lstm_a = nn.LSTMCell(self.dh_l, self.dh_a)
			self.phantom_lstm_v = nn.LSTMCell(self.dh_l, self.dh_v)
		elif mode == 'PhantomIntermInputD':
			self.phantom_lstm_a = nn.LSTMCell(self.d_l, self.dh_a)
			self.phantom_lstm_v = nn.LSTMCell(self.d_l, self.dh_v)
		

		self.att1_fc1 = nn.Linear(attInShape, h_att1)
		self.att1_fc2 = nn.Linear(h_att1, attInShape)
		self.att1_dropout = nn.Dropout(att1_dropout)

		self.att2_fc1 = nn.Linear(attInShape, h_att2)
		self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
		self.att2_dropout = nn.Dropout(att2_dropout)

		self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
		self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
		self.gamma1_dropout = nn.Dropout(gamma1_dropout)

		self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
		self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
		self.gamma2_dropout = nn.Dropout(gamma2_dropout)

		self.out_fc1 = nn.Linear(final_out, h_out)
		self.out_fc2 = nn.Linear(h_out, output_dim)
		self.out_dropout = nn.Dropout(out_dropout)

		self.criterion = nn.L1Loss()
		self.phantom_v_criterion = nn.L1Loss()
		self.phantom_a_criterion = nn.L1Loss()

		self.mode = mode


	def forward(self,x_l, x_a, x_v, isPhantom=False, avZero=False, intermPhantom=False, phantomInput=False):

		pred_x_a = x_a
		pred_x_v = x_v

		# x is t x n x d
		n = x_l.shape[1]
		t = x_l.shape[0]

		#print('n:', n)
		#print('t:', t)
		#print('x_l shape:', x_l.shape)
		#print('x_a shape:', x_a.shape)
		#print('x_v shape:', x_v.shape)

		if isPhantom is True:
			self.phantom_h_a = torch.zeros(self.phantom_num_layers, n, self.d_a_phantom).cuda()
			self.phantom_c_a = torch.zeros(self.phantom_num_layers, n, self.d_a_phantom).cuda()
			x_a_int, _ = self.phantom_a(x_l, (self.phantom_h_a, self.phantom_c_a))
			x_a = self.phantom_ff_a(x_a_int)

			self.phantom_h_v = torch.zeros(self.phantom_num_layers, n, self.d_v_phantom).cuda()
			self.phantom_c_v = torch.zeros(self.phantom_num_layers, n, self.d_v_phantom).cuda()
			x_v_int, _ = self.phantom_v(x_l, (self.phantom_h_v, self.phantom_c_v))
			x_v = self.phantom_ff_v(x_v_int)

			pred_x_a = x_a
			pred_x_v = x_v

		if avZero is True:
			x_a = torch.zeros(x_a.shape).cuda()
			x_v = torch.zeros(x_v.shape).cuda()



		self.h_l = torch.zeros(n, self.dh_l).cuda()
		self.h_a = torch.zeros(n, self.dh_a).cuda()
		self.h_v = torch.zeros(n, self.dh_v).cuda()
		self.c_l = torch.zeros(n, self.dh_l).cuda()
		self.c_a = torch.zeros(n, self.dh_a).cuda()
		self.c_v = torch.zeros(n, self.dh_v).cuda()
		self.mem = torch.zeros(n, self.mem_dim).cuda()

		all_h_ls = []
		all_h_as = []
		all_h_vs = []
		all_c_ls = []
		all_c_as = []
		all_c_vs = []
		all_mems = []

		for i in range(t):
			# prev time step
			prev_c_l = self.c_l
			prev_c_a = self.c_a
			prev_c_v = self.c_v
			# curr time step
			new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
			if intermPhantom is True:
				if phantomInput is True:
					new_h_a, new_c_a = self.phantom_lstm_a(x_l[i], (self.h_a, self.c_a))
					new_h_v, new_c_v = self.phantom_lstm_v(x_l[i], (self.h_v, self.c_v))
				else:
					new_h_a, new_c_a = self.phantom_lstm_a(new_c_l, (self.h_a, self.c_a))
					new_h_v, new_c_v = self.phantom_lstm_v(new_c_l, (self.h_v, self.c_v))
			else:
				new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
				new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
			# concatenate
			prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
			new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
			cStar = torch.cat([prev_cs,new_cs], dim=1)
			attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
			attended = attention*cStar
			cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
			both = torch.cat([attended,self.mem], dim=1)
			gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
			gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
			self.mem = gamma1*self.mem + gamma2*cHat
			all_mems.append(self.mem)
			# update
			self.h_l, self.c_l = new_h_l, new_c_l
			self.h_a, self.c_a = new_h_a, new_c_a
			self.h_v, self.c_v = new_h_v, new_c_v
			all_h_ls.append(self.h_l)
			all_h_as.append(self.h_a)
			all_h_vs.append(self.h_v)
			all_c_ls.append(self.c_l)
			all_c_as.append(self.c_a)
			all_c_vs.append(self.c_v)

		# last hidden layer last_hs is n x h
		last_h_l = all_h_ls[-1]
		last_h_a = all_h_as[-1]
		last_h_v = all_h_vs[-1]
		last_mem = all_mems[-1]
		last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
		output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
		return output, pred_x_a, pred_x_v

	def calc_phantom_loss(self, batchsize, X_test, y_test, mode='before'):

		total_loss_a = [0.0]*74
		total_loss_v = [0.0]*35

		self.eval()
		total_n = X_test.shape[1]
		num_batches = total_n / batchsize
		for batch in xrange(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			#optimizer.zero_grad()

			# isPhantom = True if random.random() < 0.5 else False

			batch_X = torch.Tensor(X_test[:,start:end]).cuda()
			batch_y = torch.Tensor(y_test[start:end]).cuda()
			x_l = batch_X[:,:,:self.d_l]
			x_a = batch_X[:,:,self.d_l:self.d_l+self.d_a]
			x_v = batch_X[:,:,self.d_l+self.d_a:]

			predictions, pred_x_a, pred_x_v = self.forward(x_l, x_a, x_v, True)

			for i in range(74):
				total_loss_a[i] += self.phantom_a_criterion(pred_x_a[:,:,i], x_a[:,:,i]).item()/74
			for i in range(35):
				total_loss_v[i] += self.phantom_v_criterion(pred_x_v[:,:,i], x_v[:,:,i]).item()/35

		audio_loss = 0.0
		video_loss = 0.0
		for i in range(74):
			print mode, "audio loss dim", i, ":", total_loss_a[i]
			audio_loss += total_loss_a[i]
		for i in range(35):
			print mode, "video loss dim", i, ":", total_loss_v[i]
			video_loss += total_loss_v[i]

		print mode, "loss a:", audio_loss
		print mode, "loss v:", video_loss


	def train_epoch(self, batchsize, X_train, y_train, optimizer):
		epoch_loss = 0
		self.train()
		total_n = X_train.shape[1]
		num_batches = total_n / batchsize	
		for batch in xrange(num_batches):
			start = batch*batchsize
			end = (batch+1)*batchsize
			optimizer.zero_grad()


			batch_X = torch.Tensor(X_train[:,start:end]).cuda()
			batch_y = torch.Tensor(y_train[start:end]).cuda()
			x_l = batch_X[:,:,:self.d_l]
			x_a = batch_X[:,:,self.d_l:self.d_l+self.d_a]
			x_v = batch_X[:,:,self.d_l+self.d_a:]

			if self.mode in ['PhantomDG', 'PhantomD']:
				isPhantom = True if random.random() < self.modality_drop else False
			else:
				isPhantom = False

			if self.mode in ['PhantomIntermD', 'PhantomIntermInputD']:
				intermPhantom = True if random.random() < self.modality_drop else False
				phantomInput = True if self.mode == 'PhantomIntermInputD' else False
			else:
				intermPhantom = False
				phantomInput = False

			if self.mode == 'PhantomBlind':
				avZero = True
			else:
				avZero = False

			predictions, pred_x_a, pred_x_v = self.forward(x_l, x_a, x_v, isPhantom, avZero, intermPhantom, phantomInput)
			predictions = predictions.squeeze(1)

			loss = self.criterion(predictions, batch_y)
			torch.set_printoptions(profile="full")



			if self.mode == 'PhantomDG':
				loss += self.phantom_a_criterion(pred_x_a, x_a)
				loss += self.phantom_v_criterion(pred_x_v, x_v)

			#if math.isnan(loss.item()):
				#continue
				#print x_a[:,:,7]
				#sys.exit(0)

			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / num_batches

	def evaluate(self, X_valid, y_valid):
		epoch_loss = 0
		self.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_valid).cuda()
			batch_y = torch.Tensor(y_valid).cuda()
			x_l = batch_X[:,:,:self.d_l]
			x_a = batch_X[:,:,self.d_l:self.d_l+self.d_a]
			x_v = batch_X[:,:,self.d_l+self.d_a:]

			if self.mode in ['PhantomDG', 'PhantomD']:
				isPhantom = True
			else:
				isPhantom = False

			if self.mode in ['PhantomIntermD', 'PhantomIntermInputD']:
				intermPhantom = True
				phantomInput = True if self.mode == 'PhantomIntermInputD' else False
			else:
				intermPhantom = False
				phantomInput = False

			if self.mode in ['PhantomBlind', 'PhantomICL']:
				avZero = True
			else:
				avZero = False

			predictions, pred_x_a, pred_x_v = self.forward(x_l, x_a, x_v, isPhantom, avZero, intermPhantom, phantomInput)
			predictions = predictions.squeeze(1)

			epoch_loss = self.criterion(predictions, batch_y)
		return epoch_loss.item()

	def predict(self, X_test):
		epoch_loss = 0
		self.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			x_l = batch_X[:,:,:self.d_l]
			x_a = batch_X[:,:,self.d_l:self.d_l+self.d_a]
			x_v = batch_X[:,:,self.d_l+self.d_a:]

			if self.mode in ['PhantomDG', 'PhantomD']:
				isPhantom = True
			else:
				isPhantom = False

			if self.mode in ['PhantomIntermD', 'PhantomIntermInputD']:
				intermPhantom = True
				phantomInput = True if self.mode == 'PhantomIntermInputD' else False
			else:
				intermPhantom = False
				phantomInput = False


			if self.mode in ['PhantomBlind', 'PhantomICL']:
				avZero = True
			else:
				avZero = False

			if self.mode in ['PhantomBlind', 'PhantomICL']:
				avZero = True
			else:
				avZero = False

			predictions, _, _ = self.forward(x_l, x_a, x_v, isPhantom, avZero, intermPhantom, phantomInput)
			predictions = predictions.squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions



def train_mfn_phantom(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, mode, save_path):
	p = np.random.permutation(X_train.shape[0])
	X_train = X_train[p]
	y_train = y_train[p]

	X_train = X_train.swapaxes(0,1)
	X_valid = X_valid.swapaxes(0,1)
	X_test = X_test.swapaxes(0,1)

	d = X_train.shape[2]
	h = 128
	t = X_train.shape[0]
	output_dim = 1
	dropout = 0.5

	[config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs

	#model = EFLSTM(d,h,output_dim,dropout)
	model = MFNPhantom(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig, mode)

	#optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

	# optimizer = optim.SGD([
	#                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
	#                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
	#             ], momentum=0.9)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)

	optimizer = optim.Adam(model.parameters(),lr=config["lr"])
	scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)


	best_valid = 999999.0
	rand = random.randint(0,100000)

	model.calc_phantom_loss(config["batchsize"], X_test, y_test, 'before')

	best_model = None
	for epoch in range(config["num_epochs"]):
		train_loss = model.train_epoch(config["batchsize"], X_train, y_train, optimizer)
		train_loss = model.evaluate(X_train, y_train)
		valid_loss = model.evaluate(X_valid, y_valid)
		test_loss = model.evaluate(X_test, y_test)
		scheduler.step(valid_loss)
		print 'Epoch', epoch, ':\t Training loss:', train_loss
		if valid_loss <= best_valid:
			# save model
			print 'Epoch', epoch, ':\t Validation loss:', valid_loss, 'saving model'
			best_valid = valid_loss
			torch.save(model, '{}/mfn_phantom_{}.pt'.format(save_path, args.hparam_iter))
			#best_model = copy.deepcopy(model).cpu().gpu()
		else:
			print 'Epoch', epoch, ':\t Validation loss:', valid_loss
		print 'Epoch', epoch, ':\t Test loss:', test_loss


	print 'model number is:', rand
	model = torch.load('{}/mfn_phantom_{}.pt'.format(save_path, args.hparam_iter))
	#model = copy.deepcopy(best_model).cpu().gpu()

	model.calc_phantom_loss(config["batchsize"], X_test, y_test, 'after')

	for split in ['train', 'valid', 'test']:

		if split is 'train':
			X, y = X_train, y_train
		elif split is 'valid':
			X, y = X_valid, y_valid
		else:
			X, y = X_test, y_test

		predictions = model.predict(X)
		mae = np.mean(np.absolute(predictions-y))
		print split, "mae: ", mae
		corr = np.corrcoef(predictions,y)[0][1]
		print split, "corr: ", corr
		mult = round(sum(np.round(predictions)==np.round(y))/float(len(y)),5)
		print split, "mult_acc: ", mult
		f_score = round(f1_score(np.round(predictions),np.round(y),average='weighted'),5)
		print split, "mult f_score: ", f_score
		true_label = (y >= 0)
		predicted_label = (predictions >= 0)
		print split, "Confusion Matrix :"
		print confusion_matrix(true_label, predicted_label)
		print split, "Classification Report :"
		print classification_report(true_label, predicted_label, digits=5)
		print split, "Accuracy ", accuracy_score(true_label, predicted_label)
	sys.stdout.flush()

def test(X_test, y_test, metric):
	X_test = X_test.swapaxes(0,1)
	def predict(model, X_test):
		epoch_loss = 0
		model.eval()
		with torch.no_grad():
			batch_X = torch.Tensor(X_test).cuda()
			predictions = model.forward(batch_X).squeeze(1)
			predictions = predictions.cpu().data.numpy()
		return predictions
	dev = torch.cuda.device_count()
	if metric == 'mae':
		# model = torch.load('best/mfn_mae.pt', map_location='cuda:'+str(dev-1))
		model = torch.load('temp_models/mfn_99577.pt', map_location='cuda:'+str(dev-1))
	if metric == 'acc':
		model = torch.load('best/mfn_acc.pt', map_location='cuda:'+str(dev-1))
	model = model.cpu().cuda()
	
	predictions = predict(model, X_test)
	print predictions.shape
	print y_test.shape
	mae = np.mean(np.absolute(predictions-y_test))
	print "mae: ", mae
	corr = np.corrcoef(predictions,y_test)[0][1]
	print "corr: ", corr
	mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
	print "mult_acc: ", mult
	f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
	print "mult f_score: ", f_score
	true_label = (y_test >= 0)
	predicted_label = (predictions >= 0)
	print "Confusion Matrix :"
	print confusion_matrix(true_label, predicted_label)
	print "Classification Report :"
	print classification_report(true_label, predicted_label, digits=5)
	print "Accuracy ", accuracy_score(true_label, predicted_label)
	sys.stdout.flush()

local = False

if local:
	X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args,config)

	h5f = h5py.File('data/X_train.h5', 'w')
	h5f.create_dataset('data', data=X_train)
	h5f = h5py.File('data/y_train.h5', 'w')
	h5f.create_dataset('data', data=y_train)
	h5f = h5py.File('data/X_valid.h5', 'w')
	h5f.create_dataset('data', data=X_valid)
	h5f = h5py.File('data/y_valid.h5', 'w')
	h5f.create_dataset('data', data=y_valid)
	h5f = h5py.File('data/X_test.h5', 'w')
	h5f.create_dataset('data', data=X_test)
	h5f = h5py.File('data/y_test.h5', 'w')
	h5f.create_dataset('data', data=y_test)

	sys.stdout.flush()

X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()
X_train[X_train==float("-inf")] = 0.0
X_valid[X_valid==float("-inf")] = 0.0
X_test[X_test==float("-inf")] = 0.0


#test(X_test, y_test, 'mae')
#test(X_test, y_test, 'acc')
#assert False

#config = dict()
#config["batchsize"] = 32
#config["num_epochs"] = 100
#config["lr"] = 0.01
#config["h"] = 128
#config["drop"] = 0.5
#train_ef(X_train, y_train, X_valid, y_valid, X_test, y_test, config)
#assert False
parser = argparse.ArgumentParser(description='Phantom Modality training')
parser.add_argument('--mode', type=str)
parser.add_argument('--g_loss_weight', type=float)
parser.add_argument('--modality_drop', type=float)
parser.add_argument('--hparam_iter', default=0, type=int)

args = parser.parse_args()

i = args.hparam_iter
print "Hparam iter:", i



config = dict()
config["input_dims"] = [300,74,35]
config["memsize"] = 128
config["windowsize"] = 2
config["batchsize"] = 32
config["num_epochs"] = 200
config["lr"] = 0.00001
config["momentum"] = 0.9

random.seed(123*i+456)

network_dropout = random.choice([0.0, 0.2, 0.5])
modality_dropout = random.choice([0.1, 0.3, 0.5])

hl = random.choice([32,64,128,256])
ha = random.choice([8,32,64,80])
hv = random.choice([8,32,64,80])
config["h_dims"] = [hl,ha,hv]
# config["modality_drop"] = modality_dropout
config["modality_drop"] = args.modality_drop
config["g_loss_weight"] = args.g_loss_weight

NN1Config = dict()
NN1Config["shapes"] = random.choice([64,128,256])
NN1Config["drop"] = network_dropout
NN2Config = dict()
NN2Config["shapes"] = random.choice([64,128,256])
NN2Config["drop"] = network_dropout
gamma1Config = dict()
gamma1Config["shapes"] = random.choice([64,128,256])
gamma1Config["drop"] = network_dropout
gamma2Config = dict()
gamma2Config["shapes"] = random.choice([64,128,256])
gamma2Config["drop"] = network_dropout
outConfig = dict()
outConfig["shapes"] = random.choice([64, 128,256])
outConfig["drop"] = network_dropout

configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
print configs

mode = args.mode
if mode not in ['PhantomDG', 'PhantomD', 'PhantomBlind', 'PhantomICL', 'MFN', 'PhantomIntermD', 'PhantomIntermInputD']:
	print "Mode argument is invalid!"
	sys.exit(0)

save_path = PROJECT_DIR + 'gridsearch_models_mosei_200epochs/'+mode
if mode in ['PhantomD', 'PhantomDG', 'PhantomIntermD', 'PhantomIntermInputD']:
	save_path += '_D'+str(args.modality_drop)
if mode == 'PhantomDG':
	save_path += '_G'+str(args.g_loss_weight)
print "Save path: " + save_path


# mode = 'PhantomDG'      # loss = task + avloss during training (only task during val) (phantom modality during val and test)
# mode = 'PhantomD'     # loss = task during training and val (phantom modality during val and test)
# mode = 'PhantomBlind' # loss = task during training and val (0 inputs during training, val and test)
# mode = 'PhantomICL'   # loss = task during training and val (ground truth inputs during training, 0 inputs during val and test)

train_mfn_phantom(X_train, y_train, X_valid, y_valid, X_test, y_test, configs, mode, save_path)

