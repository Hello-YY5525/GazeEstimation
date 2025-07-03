import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list): #检查path路径是否为列表
      for i in path: #便利path中的每个文件
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)#返回数据集的长度

  def __getitem__(self, idx):
    #获取数据集中索引为idx项目
    line = self.lines[idx]
    #print(line)
    #print('----')
    #按空格分割
    line = line.strip().split(" ")
    #print(line)

    name = line[3]
    gaze2d = line[7]
    head2d = line[8]
    lefteye = line[1]
    righteye = line[2]
    face = line[0]

    label = np.array(gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    headpose = np.array(head2d.split(",")).astype("float")
    headpose = torch.from_numpy(headpose).type(torch.FloatTensor)

    # rimg = cv2.imread(os.path.join(self.root, righteye))/255.0
    # rimg = rimg.transpose(2, 0, 1)

    # limg = cv2.imread(os.path.join(self.root, lefteye))/255.0
    # limg = limg.transpose(2, 0, 1)

    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
    fimg = cv2.resize(fimg, (224, 224))/255.0 #resize，并将像素值缩放到0-1之间
    
    fimg = fimg.transpose(2, 0, 1) #转置以适应pytoch，变成(通道数*高度*宽度)

    #创建一个dic，并将fimg转换为torch.tensor类型
    img = {"face":torch.from_numpy(fimg).type(torch.FloatTensor),
            "head_pose":headpose,
            "name":name}
    #print(label)


    # img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
    #        "right":torch.from_numpy(rimg).type(torch.FloatTensor),
    #        "face":torch.from_numpy(fimg).type(torch.FloatTensor),
    #        "head_pose":headpose,
    #        "name":name}

    return img, label

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header) #创建数据集对象
  #打印数据集相关信息
  print(f"[Read Data]: Total num: {len(dataset)}")
  print(f"[Read Data]: Label path: {labelpath}")
  #利用torch的DataLoader类创建数据加载对象load
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load


if __name__ == "__main__":
  path = '/home/gz/datasets/MPIIFaceGaze/Label/p00.label'
  d = loader(path,'/home/gz/datasets/MPIIFaceGaze/Image')
  #print(len(d))
  (data, label) = d.__getitem__(0)
  #print(data['face'].size())
  #print('---------------')
  #print(data)

