import torch
import torch.nn as nn

class Proto_Net(nn.Module):
    def __init__(self, input_size, num_class):
        super(Proto_Net, self).__init__()
        self.prototypes = nn.Embedding(num_class, input_size)

    def __distance__(self, reps, protos):
        # similarity = 1-1/(1+torch.exp((torch.sum(protos*reps, 1)-384)/100))
        similarity = 1-1/(1+torch.exp(torch.cosine_similarity(reps, protos)))
        return similarity
    def forward(self, reps, labels, neg_labels):
        pos_protos = self.prototypes(labels)
        neg_protos = self.prototypes(neg_labels)
        p_similarity = self.__distance__(reps, pos_protos)
        n_similarity = self.__distance__(reps.unsqueeze(1).permute(0,2,1), neg_protos.permute(0,2,1))
        cluster_loss = -(torch.mean(torch.log(p_similarity+1e-5))+torch.mean(torch.log(1-n_similarity+1e-5)))
        return cluster_loss, pos_protos

    def get_protos(self, labels):
        labels = torch.tensor(labels).cuda()
        return self.prototypes(labels)

    def get_all_protos(self):
        return self.prototypes.weight