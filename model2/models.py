import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self,in_dim, hid_dim, n_class):
        super(Discriminator,self).__init__()
        self.n_class = n_class
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_dim, hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim,hid_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hid_dim,n_class),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self,batch):
        return torch.softmax(self.mlp(batch),dim = 1)

class Sense_Mapping(nn.Module):
    def __init__(self,token_dim, hid_dim, sense_dim,k_senses, use_mlp = True):
        super(Sense_Mapping, self).__init__()
        self.sense_map = nn.Sequential(
            nn.Linear(token_dim,hid_dim,bias = False),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim,bias = False),
            nn.Tanh(),
            nn.Linear(hid_dim, sense_dim,bias = False),
            nn.Tanh()
        )
        
        self.use_mlp = use_mlp
        
        if use_mlp:
            #print("use MLP for clustering")
            self.mlp = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(sense_dim, hid_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hid_dim,hid_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hid_dim,k_senses),
                nn.LeakyReLU(0.1),
                nn.Softmax(dim = 1)
            )
        

            #print("use mindist for clustering")
    
    def forward(self, batch,centroids = None):
        sense_embeddings = self.sense_map(batch)
        
        if self.use_mlp:
            sense_distribution = self.mlp(sense_embeddings)
            which_sense = torch.argmax(sense_distribution,dim=1)
        else:
            sqdist_to_center = torch.sum(torch.square(centroids[:,None,:] - sense_embeddings[None,:,:]), axis=2)
            mindist_to_center = torch.min(sqdist_to_center, axis=0).values
            which_sense = torch.argmax(torch.where(sqdist_to_center.T==mindist_to_center[:,None],1,0),dim=1)       
        
        return {'sense_embeddings':sense_embeddings,
                #'sense_distribution':sense_distribution,
                'which_sense': which_sense}

class Mapping_and_Shared_Centroids(nn.Module):
    def __init__(self, 
                 k_assignment, 
                 token_dim, 
                 hid_dim, 
                 sense_dim, 
                 modals,
                 sub_module_use_mlp,
                 device):
        super(Mapping_and_Shared_Centroids, self).__init__()
        self.modals = modals
        self.k_assignment = k_assignment
        
        self.centroids = nn.Parameter(torch.rand(k_assignment[modals[0]],sense_dim,requires_grad=True))

        #self.centroids = nn.ParameterDict({modal:nn.Parameter(torch.rand(k_assignment[modal],sense_dim,requires_grad=True).to(device)) for modal in modals})
        self.submodules = nn.ModuleDict({modal:Sense_Mapping(token_dim, hid_dim, sense_dim, k_assignment[modal],sub_module_use_mlp) for modal in modals})
        self.tanh = nn.Tanh()
        
    def forward(self, batch_embeddings):
        
        #keep centroids within radius of 1
        #centroids = {modal:self.tanh(self.centroids[modal]) for modal in self.modals}
        centroids = self.tanh(self.centroids)
        sense_embeddings = None
        word_class = None
        sense_class = None
        embeddings_grouped_by_sense = dict()
        
        #for testing and plotting
        embeddings_dict = dict()
        sense_class_dict = dict()
        
        for modal in self.modals:
            
            #for discriminator:
                #get sense_embeddings for a batch of instances of a modal [batch, sense_dim]
                #record word class index for this chunk of sense_embeddings (for the crossentropy loss of discriminator)
                #record sense class index for this chunk of sense_embeddings
            
            embeddings = batch_embeddings[modal]
            submodule = self.submodules[modal]
            submodule_out = submodule(embeddings, centroids)
            _sense_embeddings = submodule_out['sense_embeddings']
            _word_class = torch.tensor([self.modals.index(modal)] * submodule_out['sense_embeddings'].shape[0])
            
            
            #for distance loss:
                
            _sense_class = submodule_out['which_sense']
            
            _embeddings_grouped_by_sense = dict()
            for i in range(self.k_assignment[modal]):
                indices = (_sense_class == i).nonzero(as_tuple=False).squeeze(-1)
                embeddings_of_a_sense = _sense_embeddings.index_select(0,indices)
                _embeddings_grouped_by_sense[i] = embeddings_of_a_sense
                
            embeddings_grouped_by_sense[modal] = _embeddings_grouped_by_sense
            
            if sense_embeddings == None:
                sense_embeddings = _sense_embeddings
                word_class = _word_class
                sense_class = _sense_class
            else:
                sense_embeddings = torch.cat((sense_embeddings, _sense_embeddings),dim=0)
                word_class = torch.cat((word_class,_word_class))
                sense_class = torch.cat((sense_class,_sense_class))
            
            sense_class_dict[modal] = _sense_class
            embeddings_dict[modal] = _sense_embeddings
        
        return {"centroids":centroids,
                "sense_embeddings": sense_embeddings,
                "word_class":word_class,
                "sense_class":sense_class,
                "embeddings_dict":embeddings_dict,
                "sense_class_dict":sense_class_dict,
                "embeddings_grouped_by_sense":embeddings_grouped_by_sense}

class Mapping_and_Centroids(nn.Module):
    def __init__(self, 
                 k_assignment, 
                 token_dim, 
                 hid_dim, 
                 sense_dim, 
                 modals,
                 sub_module_use_mlp,
                 device):
        super(Mapping_and_Centroids, self).__init__()
        self.modals = modals
        self.k_assignment = k_assignment
        
        self.centroids = nn.ParameterDict({modal:nn.Parameter(torch.rand(k_assignment[modal],sense_dim,requires_grad=True).to(device)) for modal in modals})
        self.submodules = nn.ModuleDict({modal:Sense_Mapping(token_dim, hid_dim, sense_dim, k_assignment[modal],sub_module_use_mlp) for modal in modals})
        self.tanh = nn.Tanh()
        
    def forward(self, batch_embeddings):
        
        #keep centroids within radius of 1
        centroids = {modal:self.tanh(self.centroids[modal]) for modal in self.modals}
        
        sense_embeddings = None
        word_class = None
        sense_class = None
        embeddings_grouped_by_sense = dict()
        
        #for testing and plotting
        embeddings_dict = dict()
        sense_class_dict = dict()
        
        for modal in self.modals:
            
            #for discriminator:
                #get sense_embeddings for a batch of instances of a modal [batch, sense_dim]
                #record word class index for this chunk of sense_embeddings (for the crossentropy loss of discriminator)
                #record sense class index for this chunk of sense_embeddings
            
            embeddings = batch_embeddings[modal]
            submodule = self.submodules[modal]
            submodule_out = submodule(embeddings, centroids[modal])
            _sense_embeddings = submodule_out['sense_embeddings']
            _word_class = torch.tensor([self.modals.index(modal)] * submodule_out['sense_embeddings'].shape[0])
            
            
            #for distance loss:
                
            _sense_class = submodule_out['which_sense']
            
            _embeddings_grouped_by_sense = dict()
            for i in range(self.k_assignment[modal]):
                indices = (_sense_class == i).nonzero(as_tuple=False).squeeze(-1)
                embeddings_of_a_sense = _sense_embeddings.index_select(0,indices)
                _embeddings_grouped_by_sense[i] = embeddings_of_a_sense
                
            embeddings_grouped_by_sense[modal] = _embeddings_grouped_by_sense
            
            if sense_embeddings == None:
                sense_embeddings = _sense_embeddings
                word_class = _word_class
                sense_class = _sense_class
            else:
                sense_embeddings = torch.cat((sense_embeddings, _sense_embeddings),dim=0)
                word_class = torch.cat((word_class,_word_class))
                sense_class = torch.cat((sense_class,_sense_class))
            
            sense_class_dict[modal] = _sense_class
            embeddings_dict[modal] = _sense_embeddings
        
        return {"centroids":centroids,
                "sense_embeddings": sense_embeddings,
                "word_class":word_class,
                "sense_class":sense_class,
                "embeddings_dict":embeddings_dict,
                "sense_class_dict":sense_class_dict,
                "embeddings_grouped_by_sense":embeddings_grouped_by_sense}