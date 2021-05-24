import torch
from torch import nn
cos = nn.CosineSimilarity(dim=1, eps=1e-7)

def neg_entropy(dist):
    logp = torch.log(dist + 1e-5)
    return torch.neg(torch.sum(logp * torch.neg(dist)))

def cos_sim_regularization_loss(sense_embeddings, segments, device):
    loss_sum = torch.zeros(1).to(device)
    
    for start, end in segments:
        segment = sense_embeddings[start:end]
        for i in range(0,end-start):
            for j in range(i+1,end-start):
                loss_sum += cos(segment[i][None,:],segment[j][None,:])+1
    loss_sum = loss_sum/len(segments)
    return loss_sum

def distance_regularization_loss(sense_embeddings, segments,device):
    distance_sum = torch.zeros(1).to(device)
    for start, end in segments:
         distance_sum += torch.sum(torch.cdist(sense_embeddings[start:end], sense_embeddings[start:end]))/2
    
    return torch.neg(distance_sum)/len(segments)

def combine_loss(dist, sense_embeddings,segments,device,t):
    l1 = neg_entropy(dist)
    #l2 = cos_sim_regularization_loss(sense_embeddings,segments, device)
    l2 = distance_regularization_loss(sense_embeddings,segments, device)
    
    return l1 + (1/t)*l2, l1, l2
    #+ (0.01/t**2) * l2, l1, l2
        