import torch
from torch import nn
torch.manual_seed(0)
import random
random.seed(0)
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

	#cnt for learning rate scheduling
	cnt = 0
	print("Training starts")

	for i in range(epoch):
		start_time = time.time()
		print("preparing batch embedding tensors")
		
		batch_sentences = {modal: random.sample(modal_sents[modal],batch_size) for modal in modals}
		batch_embeddings = {modal: get_bert_embeddings(tokenizer,bert,sents,modal,device)[0] for modal, sents in batch_sentences.items()}

		print("Fitting generator")
		for j in range(sub_epoch):
			start_2 = time.time()
			gen_out = generator(batch_embeddings,device)
			probs = discriminator(gen_out['sense_embeddings'])
			if mode == 'token level':
				loss = gen_loss(probs)
			else:
				loss, neg_entropy, dist = gen_loss(probs, gen_out['sense_embeddings'],gen_out['segments'],device,cnt)
				cnt += 1
			if torch.isnan(loss):
				print(probs)
				break
				

			gen_optimizer.zero_grad()
			loss.backward(retain_graph = True)
			gen_optimizer.step()
			print("epoch:{}.{}, loss_gen:{:7f}, neg_entropy:{:7f}, dist:{:7f},  time usage:{}".format(i+1, j+1, loss.item(), neg_entropy.item(), dist.item(), (time.time() - start_2)/60))
			
			gen_loss_log.append(loss.item())

		print("Fitting discriminator")
		for j in range(sub_epoch):
			start_1 = time.time()
			gen_out = generator(batch_embeddings,device)
			probs = discriminator(gen_out["sense_embeddings"])
			loss = discr_loss(probs,gen_out["labels"])
			discr_optimizer.zero_grad()
			loss.backward(retain_graph = True)
			discr_optimizer.step()
			print("epoch:{}.{}, loss_discr:{:7f}, time usage:{:3f} minutes".format(i+1, j+1, loss.item(),(time.time() - start_1)/60))
			discr_loss_log.append(loss.item())

		
		epoch_time = (time.time() - start_time)/60
		print("\n\n\nlast epoch time usage: {:3f} minutes\n\n\n".format(int(epoch_time)))

	if serialize_dir:
		torch.save(generator.state_dict(), "{}/joint_align_1.pth".format(serialize_dir))
		torch.save(discriminator.state_dict(),"{}/discriminator.pth".format(serialize_dir))
		print("model saved")

		np.save(open("{}/training_stats/genloss-log.npy".format(serialize_dir),'wb'), np.array(gen_loss_log))
		np.save(open("{}/training_stats/discrloss-log.npy".format(serialize_dir),'wb'),np.array(discr_loss_log))

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
	parser.add_argument("device")
	

	args = parser.parse_args()
	config = json.load(open(args.config,'r'))
	num_epoch = int(args.num_epoch)
	num_sub_epoch = int(args.num_sub_epoch)
	batch_size = int(args.batch_size)
	sent_data_path = args.sent_data_path
	serialize_dir = args.serialize_dir

	device = args.device if torch.cuda.is_available() else "cpu"
	print("Using {} device".format(device))
	
	modals = config['modals']
	#modals = ["can","could","may","might","should","shall","must","will"]
	modal_idx = {modal: modals.index(modal) for modal in modals}
	
	modal_sents = dict()
	for modal in modals:
		modal_sents[modal] = get_sents_from_file("{}/{}.txt".format(sent_data_path,modal))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
	generator = None
	if config['mode'] == 'token level':
		generator = joint_align(config['mode'], modals=modals,modal_idx=modal_idx, token_dim=config['generator']['token_dim'], sense_dim=config['generator']['sense_dim'], hid_dim=config['generator']['hid_dim']).to(device)
	else:
		print("using sense level model")
		generator = joint_align(config['mode'], modals=modals,modal_idx=modal_idx, token_dim=config['generator']['token_dim'], sense_dim=config['generator']['sense_dim'], hid_dim=config['generator']['hid_dim'],k_assignment=config['k_assignment']).to(device)

	discriminator = Discriminator(config['discriminator']['input_dim'],config['discriminator']['hid_dim'],len(modals)).to(device)

	discr_loss = nn.CrossEntropyLoss()
	gen_loss = combine_loss

	gen_optimizer = torch.optim.Adam(generator.parameters())
	discr_optimizer = torch.optim.Adam(discriminator.parameters())

	GAN_train(config['mode'],
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


























