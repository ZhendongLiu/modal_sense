import torch
from torch import nn
torch.manual_seed(0)
import random
random.seed(0)
from transformers import BertTokenizer, BertModel
from util import *
from loss_functions import *
from models import *
import numpy as np
import sys
import os
import time


def GAN_train(gen_loss,
	          discr_loss,
	          gen_optimizer,
	          discr_optimizer,
	          generator,
	          discriminator,
	          bert_tokenizer,
	          bert,
	          modals,
	          modal_sents,
	          iter,
	          gen_iter,
			  discr_iter,
	          batch_size,
			  gen_share_centroids,
	          device = 'cpu',
	          serialize_dir = None):
	
	#gen_loss_log = list()
	discr_loss_log = list()
	gen_loss_log = list()
	neg_entropy_log = list()
	mean_centroid_dist_log = list()
	mean_dist_to_centroid_log = list()

	#cnt for learning rate scheduling
	cnt = 1
	print("Training starts")
	start_time = time.time()
	for i in range(iter):
		print("iteration #{}".format(i))
		#print("Fitting discriminator")
		for j in range(discr_iter):
			batch_sentences = {modal: random.sample(modal_sents[modal],batch_size) for modal in modals}
			batch_embeddings = {modal: get_bert_embeddings(tokenizer,bert,sents,modal,device)[0] for modal, sents in batch_sentences.items()}
			start_1 = time.time()
			gen_out = generator(batch_embeddings)
			discr_in = gen_out["sense_embeddings"]
			discr_in = discr_in.detach()
			probs = discriminator(discr_in)
			loss = discr_loss(probs,gen_out["word_class"].to(device))

			discr_optimizer.zero_grad()
			loss.backward()
			discr_optimizer.step()
			#     "generator loss:"
			print("discrimin loss:{:7f}".format(loss.item()))
			discr_loss_log.append(loss.item())

		#print("Fitting generator")
		for j in range(gen_iter):
			batch_sentences = {modal: random.sample(modal_sents[modal],batch_size) for modal in modals}
			batch_embeddings = {modal: get_bert_embeddings(tokenizer,bert,sents,modal,device)[0] for modal, sents in batch_sentences.items()}
			start_2 = time.time()
			gen_out = generator(batch_embeddings)
			probs = discriminator(gen_out['sense_embeddings'])
			
			loss, neg_entropy, mean_centroid_dist, mean_dist_to_centroid = gen_loss(probs, gen_out['centroids'],gen_out['embeddings_grouped_by_sense'],cnt,device,gen_share_centroids)
			cnt+=1
			gen_optimizer.zero_grad()
			loss.backward(retain_graph = True)
			gen_optimizer.step()
			print("generator loss:{:7f}, neg_entropy:{:7f}, mean_centroid_dist:{:7f}, mean_dist_to_centroid:{:7f}".format(loss.item(), neg_entropy.item(), mean_centroid_dist.item(), mean_dist_to_centroid.item()))
			
			gen_loss_log.append(loss.item())
			neg_entropy_log.append(neg_entropy.item())
			mean_centroid_dist_log.append(mean_centroid_dist.item())
			mean_dist_to_centroid_log.append(mean_dist_to_centroid.item())

		if i%10 == 0 and i > 0:

			iter_time = (time.time() - start_time)
			start_time = time.time()
			print("\n\n\ntime for 10 iterations: {:3f} seconds \n\n\n".format(iter_time))

	if serialize_dir:
		if not os.path.exists(serialize_dir):
			os.makedirs(serialize_dir)
			os.makedirs(serialize_dir + '/training_stats')
		torch.save(generator.state_dict(), "{}/generator.pth".format(serialize_dir))
		torch.save(discriminator.state_dict(),"{}/discriminator.pth".format(serialize_dir))
		print("model saved")

		np.save(open("{}/training_stats/genloss-log.npy".format(serialize_dir),'wb'), np.array(gen_loss_log))
		np.save(open("{}/training_stats/discrloss-log.npy".format(serialize_dir),'wb'),np.array(discr_loss_log))
		np.save(open("{}/training_stats/neg_entropy_log.npy".format(serialize_dir),'wb'), np.array(neg_entropy_log))
		np.save(open("{}/training_stats/mean_centroid_dist_log.npy".format(serialize_dir),'wb'),np.array(mean_centroid_dist_log))
		np.save(open("{}/training_stats/mean_dist_to_centroid_log.npy".format(serialize_dir),'wb'),np.array(mean_dist_to_centroid_log))


	else:
		return generator, discriminator




if __name__ == "__main__":
	import json
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("config")
	parser.add_argument("num_iter")
	parser.add_argument("num_gen_iter")
	parser.add_argument("num_discr_iter")
	parser.add_argument("batch_size")
	parser.add_argument("sent_data_path")
	parser.add_argument("serialize_dir")
	parser.add_argument("device")
	

	args = parser.parse_args()
	config = json.load(open(args.config,'r'))
	num_iter = int(args.num_iter)
	num_gen_iter = int(args.num_gen_iter)
	num_discr_iter = int(args.num_discr_iter)
	batch_size = int(args.batch_size)
	sent_data_path = args.sent_data_path
	serialize_dir = args.serialize_dir

	device = args.device if torch.cuda.is_available() else "cpu"
	print("Using {} device".format(device))
	
	modals = config['generator']['modals']
	#modals = ["can","could","may","might","should","shall","must","will"]
	#modal_idx = {modal: modals.index(modal) for modal in modals}
	
	modal_sents = dict()
	for modal in modals:
		modal_sents[modal] = get_sents_from_file("{}/{}.txt".format(sent_data_path,modal))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
	generator = None
	#if config['mode'] == 'token level':
	#	generator = joint_align(config['mode'], modals=modals,modal_idx=modal_idx, token_dim=config['generator']['token_dim'], sense_dim=config['generator']['sense_dim'], hid_dim=config['generator']['hid_dim']).to(device)
	#else:
	#	print("using sense level model")
	#	generator = joint_align(config['mode'], modals=modals,modal_idx=modal_idx, token_dim=config['generator']['token_dim'], sense_dim=config['generator']['sense_dim'], hid_dim=config['generator']['hid_dim'],k_assignment=config['k_assignment']).to(device)
	gen_config = config['generator']
	discr_config = config['discriminator']
	#print(type(gen_config['sub_module_use_mlp']))
	if gen_config['share_centroids']:
		generator = Mapping_and_Shared_Centroids(gen_config['k_assignment'],
									  gen_config['token_dim'],
									  gen_config['hid_dim'],
									  gen_config['sense_dim'],
									  gen_config['modals'],
									  gen_config['sub_module_use_mlp'],
									  device
									).to(device)
	else:
		generator = Mapping_and_Centroids(gen_config['k_assignment'],
										gen_config['token_dim'],
										gen_config['hid_dim'],
										gen_config['sense_dim'],
										gen_config['modals'],
										gen_config['sub_module_use_mlp'],
										device
										).to(device)

	discriminator = Discriminator(discr_config['input_dim'],
								  discr_config['hid_dim'],
								  len(modals)).to(device)

	discr_loss = nn.CrossEntropyLoss()
	gen_loss = combine_loss

	gen_optimizer = torch.optim.Adam(generator.parameters())
	discr_optimizer = torch.optim.Adam(discriminator.parameters())

	GAN_train(gen_loss,
			  discr_loss,
			  gen_optimizer,
			  discr_optimizer,
			  generator,
			  discriminator,
			  tokenizer,
			  bert,
			  modals,
			  modal_sents,
			  num_iter,
			  num_gen_iter,
			  num_discr_iter,
			  batch_size,
			  gen_config['share_centroids'],
			  device,
			  serialize_dir
			  )