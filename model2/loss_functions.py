import torch
from torch.distributions import Categorical

def neg_entropy_loss(prob):
    entropy = -Categorical(probs = prob).entropy()
    return entropy.mean()

def centroid_distance_loss(centroids,device = 'cpu',share_centroids = False):
    distance_sum = torch.zeros(1).to(device)
    
    if share_centroids:
        distance_sum = -torch.sum(torch.cdist(centroids,centroids))
    else:
        for modal, centds in centroids.items():
            distance_sum += torch.sum(torch.cdist(centds, centds))/2
        distance_sum = -distance_sum/len(centroids)

    return distance_sum

def dist_to_centroid_loss(centroids, embeddings_grouped_by_sense,device = 'cpu',share_centroids = False):
    
    
    distance_sum = torch.zeros(1).to(device)
    
    for modal, embeddings in embeddings_grouped_by_sense.items():
        if share_centroids:
            _centroids = centroids
        else:
            _centroids = centroids[modal]
        for idx, _embeddings in embeddings.items():
            centroid = _centroids[idx]
            dist = torch.sum(torch.cdist(_embeddings, centroid[None,:],p=2))
            distance_sum += dist
    return distance_sum/len(embeddings_grouped_by_sense)    

def combine_loss(prob, centroids, embeddings_grouped_by_sense, t, device = 'cpu',share_centroids = False):
    entropy_loss = neg_entropy_loss(prob)
    centroid_loss = centroid_distance_loss(centroids,device,share_centroids)
    dist_loss = dist_to_centroid_loss(centroids,embeddings_grouped_by_sense,device,share_centroids)

    if t > 20:
        return (entropy_loss + (1/20) * centroid_loss + (1/20) * dist_loss), entropy_loss, centroid_loss, dist_loss
    else:
        return (entropy_loss + (1/t) * centroid_loss + (1/t) * dist_loss), entropy_loss, centroid_loss, dist_loss