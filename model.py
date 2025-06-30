import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import modules
#from Torch.ntools import VGG16


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        # 36 * 60
        # 18 * 30
        # 9  * 15
        self.img_feature_dim = 128
        self.base_model = modules.resnet50(pretrained= True) #(batchsize, img_feature_dim)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        
        self.GazeZone9 = nn.Linear(self.img_feature_dim, 9)
        self.GazeZone16 = nn.Linear(self.img_feature_dim, 16)
        self.GazeZone25 = nn.Linear(self.img_feature_dim, 25)
        self.GazeZone36 = nn.Linear(self.img_feature_dim, 36)

        self.gazeEs = nn.Linear(self.img_feature_dim, 2)
        self.drop = nn.Dropout()

        #self._init_weights(self.weightStream)

    def forward(self, x_in):
        baseout = self.base_model(x_in["face"])
        gazefeature = self.drop(baseout)

        gazezone9 = self.GazeZone9(gazefeature)
        gazezone16 = self.GazeZone16(gazefeature)
        gazezone25 = self.GazeZone25(gazefeature)
        gazezone36 = self.GazeZone36(gazefeature)
        gazezone = [gazezone9, gazezone16, gazezone25, gazezone36]

        gaze = self.gazeEs(gazefeature)


        return gazefeature, gaze, gazezone
    
    '''
    def _init_weights(self, net):
        k = 0
        for m in net.modules():    
            if isinstance(m, nn.Conv2d):
                if k ==0 or k == 1:
                  nn.init.normal_(m.weight, mean=0, std=0.01)
                  nn.init.constant_(m.bias, 0.1)
                if k == 2:
                  nn.init.normal_(m.weight, mean=0, std=0.001)
                  nn.init.constant_(m.bias, 1)
                if k > 2:
                  print("ERROR IN WEIGHT INITIAL")
                  exit()
                k += 1
    '''
    

class FeatureLoss():
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def __call__(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss
    

class Gloss():
    def __init__(self):
        self.gloss = torch.nn.L1Loss()

    def __call__(self, gaze, gaze_trans, gaze_pre) :
        loss1 =   self.gloss(gaze, gaze_pre)
        loss2 = self.gloss(gaze, gaze_trans) 
        loss = loss1 + loss2
        #loss = loss2
        return loss

'''
class Auxiliarytask():
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()

    def __call__(self, label, pesudo_label):

        return self.CE(label, pesudo_label)
'''
class Auxiliarytask():
    def __init__(self):
        self.CE = torch.nn.CrossEntropyLoss()

    def __call__(self, gazezone, pesudo_label):
        g1_label = gazezone[0]
        g2_label = gazezone[1]
        g3_label = gazezone[2]
        g4_label = gazezone[3]
        
        pesudo_label1 = pesudo_label[0]
        pesudo_label2 = pesudo_label[1]
        pesudo_label3 = pesudo_label[2]
        pesudo_label4 = pesudo_label[3]
        

        loss1 = self.CE(g1_label, pesudo_label1)
        loss2 = self.CE(g2_label, pesudo_label2)
        loss3 = self.CE(g3_label, pesudo_label3)
        loss4 = self.CE(g4_label, pesudo_label4)

        loss = torch.mean(torch.stack([loss1, loss2, loss3, loss4]))

        return loss


class TripletLoss():
    def __init__(self) :
        self.tripletloss = nn.TripletMarginLoss()
        
    def generate_derangement(self, n):
        while True:
            perm = np.random.choice(n, size=n, replace=True)
            if np.all(perm != np.arange(n)):
                return perm
    
    def sampleselect(self, tensor1, gaze):
        
        anchor = tensor1.view(tensor1.size(0), -1) #(batchisze, c)
        batchsize = tensor1.size(0)
        inital_index = torch.arange(batchsize)
        derangement = self.generate_derangement(batchsize)
        derangement_tensor = torch.tensor(derangement, dtype=torch.long)
        positive_sample = anchor[derangement_tensor]

        gaze_anchor = gaze
        #gaze_positive = gaze[derangement_tensor]

        distances = torch.cdist(gaze_anchor, gaze_anchor) #(b,b)

        indices = []

        for i in range(batchsize):
            current_anchor = inital_index[i] #当前样本索引
            pos_anchor = derangement_tensor[i]
            current_dis = distances[i]
            pos = current_dis[derangement_tensor[i]]
            
            mask = torch.ones_like(current_dis).bool()
            mask[current_anchor] = 0
            mask[pos_anchor] = 0
            mask = mask & (current_dis > pos)

            if torch.sum(mask.int()) == 0:
                continue
            else:
                valid_indice = torch.where(mask)[0]
                _, a = torch.min(current_dis[valid_indice], dim=0)
                indices.append((i, pos_anchor, valid_indice[a]))
            
            if len(indices) == 0:
                return None, None, None
            
        indices = torch.tensor(indices, dtype=torch.long)
        valid_anchors = indices[:, 0]
        valid_positives = indices[:, 1]
        valid_negatives = indices[:, 2]

        anchor_sample = anchor[valid_anchors]
        positive_sample = positive_sample[valid_positives]
        negative_sample = anchor[valid_negatives]

        return anchor_sample, positive_sample, negative_sample
    
    def __call__(self, tensor1, gaze) : #tensor1: (batchsize,7,7) gaze: (batchsize,2)

        anchor, positive_sample, negative_sample = self.sampleselect(tensor1, gaze)
        if anchor is None:
            return torch.tensor(0.0)
        
        loss = self.tripletloss(anchor, positive_sample, negative_sample)
        #loss = loss/anchor.size(0)
        return loss



class TotalLoss():
    def __init__(self, lambda1, lambda2, lambda3, lambda4):
        self.feature_loss = FeatureLoss()
        self.gloss = Gloss()
        self.tripletloss = TripletLoss()
        self.CE = Auxiliarytask()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4

    def __call__(self, tensor1, tensor2, gaze, pesudo_label, gaze1, gaze2, gazezone1):
        ps_loss = self.CE(gazezone1, pesudo_label)
        loss1 = self.feature_loss(tensor1, tensor2)
        loss2 = self.gloss(gaze, gaze1, gaze2)
        loss31 = self.tripletloss(tensor1, gaze)
        loss32 = self.tripletloss(tensor2, gaze)
        #loss3 = loss31
        loss3 = (loss31 + loss32)
        #loss3 = 0

        total_loss = self.lambda1 * loss1 + self.lambda2 * loss2 + self.lambda3 * loss3 +self.lambda4 * ps_loss
        
        return total_loss, loss1, loss2, loss3, ps_loss

if __name__ == '__main__':


    #covariance = f(tensor1, tensor2)
    #print(covariance)
    
    '''


    m = model().cuda()
    feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }
    feature = {"head_pose": torch.zeros(10, 2).cuda(),
               "left": torch.zeros(10, 3, 36, 60).cuda(),
               "right": torch.zeros(10, 3, 36, 60).cuda(),
               "face": torch.zeros(10, 3, 448, 448).cuda()
               }
    a, img = m(feature)
    print(m)
    print(a.size())

    '''
    batchsize = 32  # 假设 batchsize 为 32
    tensor1 = torch.randn(batchsize, 20).cuda()
    tensor2 = torch.randn(batchsize, 20).cuda()
    tensor1_mean = torch.mean(tensor1, dim=1, keepdim=True)
    tensor2_mean = torch.mean(tensor2, dim=1, keepdim=True)
    tensor1_std = torch.std(tensor1, dim=1, keepdim=True)
    tensor2_std = torch.std(tensor2, dim=1, keepdim=True)

    a = (tensor1 - tensor1_mean)

    covariance = torch.mean( a * (tensor2 - tensor2_mean), dim=1)

    b = tensor1_std * tensor2_std
    correlation = covariance / (b)

    print(tensor1_mean.size())
    print(tensor1_std.size())
    print(a.size())
    print(covariance)
    print(b)
    print(correlation)


    '''
    gaze = torch.randn(batchsize, 2).cuda()
    triplet_loss = TripletLoss()
    loss = triplet_loss(tensor1, gaze)
    print(loss)
    '''
