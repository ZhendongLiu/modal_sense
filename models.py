import torch
from torch import nn
from modules import *



class Discriminator(nn.Module):
    def __init__(self,in_dim, hid_dim, n_class):
        super(Discriminator,self).__init__()
        self.n_class = n_class
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,n_class),
            nn.ReLU()
        )
    
    def forward(self,batch):
        return torch.softmax(self.mlp(batch),dim = 1)


class get_sense_for_all_token_level(nn.Module):
    
    def __init__(self,modals, token_dim,sense_dim, hidden_dim, modal_idx):
        super(get_sense_for_all_token_level,self).__init__()
        self.modals = modals
        self.modal_idx = modal_idx
        modules = {modal: sense_for_token(token_dim,sense_dim,hidden_dim) for modal in modals}
        self.submodules = nn.ModuleDict(modules)
    
    def forward(self,batch_embeddings):
        '''
        batch_embeddings: Dict(str, tensor)
        '''
        sense_embeddings = None
        labels = None
        
        for modal, embeddings in batch_embeddings.items():
            modal_idx = self.modal_idx[modal]
            module = self.submodules[modal]
            sense_embedding = module(embeddings)
            label = torch.tensor([modal_idx] * embeddings.shape[0])
            
            if sense_embeddings == None:
                sense_embeddings = sense_embedding
                labels = label
            else:
                sense_embeddings = torch.cat((sense_embeddings, sense_embedding),dim=0)
                labels = torch.cat((labels,label))
        
        return {"sense_embeddings": sense_embeddings, 
                "labels": labels}



class get_sense_for_all(nn.Module):
    def __init__(self, k_assignment, token_dim, hid_dim, sense_dim, modals, modal_idx):
        super(get_sense_for_all,self).__init__()
        self.modals = modals
        self.modal_idx = modal_idx
        self.k_assignment = k_assignment
        self.submodules = nn.ModuleDict({modal: sense_embedding(token_dim,hid_dim, sense_dim, k_assignment[modal]) for modal in modals})
        
        
    def forward(self, batch_embeddings):
        '''
        batch_embeddings: Dict(str, tensor)
        '''
        sense_embeddings = None
        labels = None
        segments = list()
        last = 0
        
        for modal, embeddings in batch_embeddings.items():
            modal_idx = self.modal_idx[modal]
            module = self.submodules[modal]
            sense_embedding = module(embeddings)
            label = torch.tensor([modal_idx] * sense_embedding.shape[0])
            segments.append((last, last + sense_embedding.shape[0]))
            last += sense_embedding.shape[0]
            
            if sense_embeddings == None:
                sense_embeddings = sense_embedding
                labels = label
            else:
                sense_embeddings = torch.cat((sense_embeddings, sense_embedding),dim=0)
                labels = torch.cat((labels,label))
        
        return {"sense_embeddings":sense_embeddings, 
                "labels":labels, 
                "segments":segments}

    