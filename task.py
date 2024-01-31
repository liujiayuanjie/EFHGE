import torch
import torch.nn as nn
import numpy as np

class LogisticRegression:
    def __init__(self, input_dim, class_num, device = None, lr = 0.001):
        linear = nn.Linear(input_dim, class_num)
        if device != None:
            linear = linear.to(device)
        self.linear = linear

        self.optimizer = torch.optim.Adam([{'params': linear.parameters(), 'weight_decay': 1e-4}], lr = lr)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def fit(self, x, y, epoch_num = 1000):
        for i in range(epoch_num):
            outputs = self.linear(x)
            loss = self.criterion(outputs, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        outputs = self.linear(x).detach()
        y = torch.argmax(outputs, dim = -1)
        return y

def kmeans(x, cluster_num, max_iters = 100):
    x_num, _ = x.shape
    clr_idx = np.random.choice(x_num, cluster_num, replace = False)
    clr_idx = torch.tensor(clr_idx, device = x.device).long()
    centroids = x[clr_idx]

    for i in range(max_iters):
        distances = torch.cdist(x, centroids)
        cluster_assignments = torch.argmin(distances, dim = -1)
        new_centroids = torch.stack([x[cluster_assignments == k].mean(dim = 0) for k in range(cluster_num)])
        
        if torch.all(torch.eq(new_centroids, centroids)):
            break
        
        centroids = new_centroids

    return cluster_assignments