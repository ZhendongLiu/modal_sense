import torch
from tqdm import tqdm

def get_sents_from_file(file_path):

    file = open(file_path,'r')
    sents = file.read()
    sents = sents.split('\n')
    return sents[:-1]
    
def save_sents(sents, file_path):
    with open(file_path,"w") as file:
        for sent in tqdm(sents):
            file.write(sent + '\n')

def get_bert_embeddings(tokenizer, model, sents, word, device, layer = 12):
    '''
    Get embeddings and return as a tensor
    '''
    embeddings = None
    sentences = list()
    
    
    for sent in sents:

        tokens = tokenizer.tokenize(sent)
        if len(tokens) > 512:
            continue
        encoded_input = tokenizer(sent, return_tensors='pt')
        output = model(**encoded_input)
        vectors = output.hidden_states[layer]
        
        for i in range(len(tokens)):
            if tokens[i] == word:
                sentences.append(sent)
                embedding = vectors[0][i+1]
                
                if embeddings == None:
                    embeddings = embedding[None,:]
                    
                else:
                    
                    embeddings = torch.cat((embeddings,embedding[None,:]),dim = 0)
    
                    
    return embeddings.to(device),sentences