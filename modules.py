import torch
from torch import nn

class sense_for_token(nn.Module):
    '''
    generate sense embeddings for a batch of modal representations
    '''
    def __init__(self, token_dim, sense_dim, hidden_dim):
        super(sense_for_token,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Linear(hidden_dim,sense_dim),
        )
    
    def forward(self, batch):
        return self.mlp(batch)

class sense_embedding(nn.Module):
    '''
    compute the distribution over K senses for each token
    get the weighted average of tokens by the distribution as sense embeddings before mapping.

    '''
    def __init__(self, token_dim, hid_dim,sense_dim, k_senses):
        super(sense_embedding,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,k_senses),
            nn.ReLU()
        )
        
        self.sense_map = nn.Sequential(
            nn.Linear(token_dim, hid_dim),
            nn.Linear(hid_dim,hid_dim),
            nn.Linear(hid_dim, sense_dim),
        )
    def forward(self, batch):
        
        ##double check this implementation
        distribution = torch.softmax(self.mlp(batch),dim = 1)
        weighted_token_ave = distribution.T.matmul(batch)/torch.sum(distribution,dim=0)[:,None]
        sense_repr = self.sense_map(weighted_token_ave)
        
        
        return sense_repr

    def predict_sense(self, batch):

        return torch.softmax(self.mlp(batch),dim = 1)










        