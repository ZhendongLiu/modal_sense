import torch
from torch import nn
import random
from transformers import BertTokenizer, BertModel
from util import *
from modules import *
from loss_functions import *
from models import *
import numpy as np
import sys
import time


def GAN_train(mode,
			  gen_loss,
	          discr_loss,
	          gen_optimizer,
	          discr_optimizer,
	          generator,
	          discriminator,
	          bert_tokenizer,
	          bert,
	          modals,
	          modal_idx,
	          modal_sents,
	          epoch,
	          sub_epoch,
	          batch_size,
	          device = 'cpu',
	          serialize_dir = None):
	
	'''

	'''
	gen_loss_log = list()
	discr_loss_log = list()

	print("Training starts")

	for i in range(epoch):
		start_time = time.time()
		print("preparing batch embedding tensors")
		while True:
			try:
				batch_sentences = {modal: random.sample(modal_sents[modal],batch_size) for modal in modals}
				batch_embeddings = {modal: get_bert_embeddings(tokenizer,bert,sents,modal,device)[0] for modal, sents in batch_sentences.items()}
				break
			except:
				continue

		print("Fitting discriminator")
		for j in range(sub_epoch):
			gen_out = generator(batch_embeddings)
			probs = discriminator(gen_out["sense_embeddings"])
			loss = discr_loss(probs,gen_out["labels"])
			discr_optimizer.zero_grad()
			loss.backward(retain_graph = True)
			discr_optimizer.step()
			print("epoch:{}.{}, loss_gen:{:7f}".format(i, j, loss.item()))
			discr_loss_log.append(loss.item())

		print("Fitting generator")
		for j in range(sub_epoch):
			gen_out = generator(batch_embeddings)
			probs = discriminator(gen_out['sense_embeddings'])
			if mode == 'token level':
				loss = gen_loss(probs)
			else:
				loss = gen_loss(probs, gen_out['sense_embeddings'],gen_out['segments'])
			gen_optimizer.zero_grad()
			loss.backward(retain_graph = True)
			gen_optimizer.step()
			print("epoch:{}.{}, loss_gen:{:7f}".format(i, j, loss.item()))
			gen_loss_log.append(loss.item())

		epoch_time = (time.time() - start_time)/60
		print("\n\n\nlast epoch time usage: {} minutes\n\n\n".format(int(epoch_time)))

	if serialize_dir:
		torch.save(generator.state_dict(), "{}/generator.pth".format(serialize_dir))
		torch.save(discriminator.state_dict(),"{}/discriminator.pth".format(serialize_dir))
		print("model saved")

		np.save(open("{}/training_data/genloss-log.npy".format(serialize_dir),'wb'), np.array(gen_loss_log))
		np.save(open("{}/training_data/discrloss-log.npy".format(serialize_dir),'wb'),np.array(discr_loss_log))

	else:
		return generator, discriminator




if __name__ == "__main__":
	import json
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("config")
	parser.add_argument("num_epoch")
	parser.add_argument("num_sub_epoch")
	parser.add_argument("batch_size")
	parser.add_argument("sent_data_path")
	parser.add_argument("serialize_dir")

	args = parser.parse_args()
	config = json.load(open(args.config,'r'))
	num_epoch = int(args.num_epoch)
	num_sub_epoch = int(args.num_sub_epoch)
	batch_size = int(args.batch_size)
	sent_data_path = args.sent_data_path
	serialize_dir = args.serialize_dir


	device = "cuda" if torch.cuda.is_available() else "cpu"
	print("Using {} device".format(device))
	
	modals = config['modals']
	#modals = ["can","could","may","might","should","shall","must","will"]
	modal_idx = {modal: modals.index(modal) for modal in modals}
	
	modal_sents = dict()
	for modal in modals:
		modal_sents[modal] = get_sents_from_file("{}/{}.txt".format(sent_data_path,modal))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
	bert.to(device)

	if config['model'] == 'token level':
		#def __init__(self,modals, token_dim,sense_dim, hidden_dim, modal_idx):
		generator = get_sense_for_all_token_level(modals, config['generator']['token_dim'],config['generator']['sense_dim'],config['generator']['hid_dim'],modal_idx)
		discriminator = Discriminator(config['discriminator']['input_dim'],config['discriminator']['hid_dim'],len(modals))

		generator.to(device)
		discriminator.to(device)

		discr_loss = nn.CrossEntropyLoss()
		gen_loss = neg_entropy

		gen_optimizer = torch.optim.Adam(generator.parameters())
		discr_optimizer = torch.optim.Adam(discriminator.parameters())

		GAN_train(config['model'],
				  gen_loss,
				  discr_loss,
				  gen_optimizer,
				  discr_optimizer,
				  generator,
				  discriminator,
				  tokenizer,
				  bert,
				  modals,
				  modal_idx,
				  modal_sents,
				  num_epoch,
				  num_sub_epoch,
				  batch_size,
				  device,
				  serialize_dir
				  )

	else:
		pass


























