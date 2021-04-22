import torch

def neg_entropy(dist):
    logp = torch.log(dist)
    return torch.neg(torch.sum(logp * torch.neg(dist)))

def distance_regularization_loss(sense_embeddings, segments):
    distance_sum = torch.zeros(1)
    for start, end in segments:
         distance_sum += torch.sum(torch.cdist(sense_embeddings[start:end], sense_embeddings[start:end]))/2
    
    return torch.neg(distance_sum)

def combine_loss(dist, sense_embeddings,segments):
	return neg_entropy(dist) + distance_regularization_loss(sense_embeddings, segments)
        