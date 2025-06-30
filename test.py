import model
import importlib
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
#import ctools

def eth_gazeto3d(gaze):
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
  gaze_gt[1] = -np.sin(gaze[0])
  gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
  return gaze_gt


def gazeto3d(gaze):
  
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def angular(gaze, label):
  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader = yaml.FullLoader)
  readername = config["reader"]
  dataloader = importlib.import_module("reader." + readername)

  config = config["test"]
  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["load"]["model_name"] 
  
  loadpath = os.path.join(config["load"]["load_path"])
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  savepath = os.path.join(loadpath, f"checkpoint")
  
  if not os.path.exists(os.path.join(loadpath, f"evaluation")):
    os.makedirs(os.path.join(loadpath, f"evaluation"))

  if os.path.isdir(labelpath):
    folder = os.listdir(labelpath)
    folder.sort()
    testlabel = copy.deepcopy(folder)
    testlabelpath = [os.path.join(labelpath, m) for m in testlabel]
  else:
    testlabelpath = labelpath
  #testlabelpath = os.path.join(testlabel, folder[0])

  print("Read data")
  dataset = dataloader.txtload(testlabelpath, imagepath, 32, num_workers=4, header=True)

  begin = config["load"]["begin_step"]
  end = config["load"]["end_step"]
  step = config["load"]["steps"]

  for saveiter in range(begin, end+step, step):
    print("Model building")
    net = model.model()
    statedict = torch.load(os.path.join(savepath, f"Iter_{saveiter}_{modelname}.pt"), map_location="cuda:0")

    net.to(device)
    net.load_state_dict(statedict)
    net.eval()

    print(f"Test {saveiter}")
    length = len(dataset)
    accs = 0
    count = 0
    with torch.no_grad():
      with open(os.path.join(loadpath, f"evaluation/diap_{saveiter}.log"), 'w') as outfile:
        outfile.write("name results gts\n")
        for j, (data, label) in enumerate(dataset):
          img = data["face"].to(device) 
          names =  data["name"]
          #print(img.size())

          img = {"face":img}

          
          gts = label.to(device)
           
          _ , gazes, _ = net(img)
          #print(type(gazes))
          
          for k, gaze in enumerate(gazes):
            gaze = gaze.cpu().detach().numpy()
            #print(k)
            #print(gaze.size)
            count += 1
            #print(gts.size())
            #gts[:, [0, 1]] = gts[:, [1, 0]]
            accs += angular(gazeto3d(gaze), gazeto3d(gts.cpu().numpy()[k]))
            
            name = [names[k]]
            gaze = [str(u) for u in gaze]
            gaze = [str(gaze[1]), str(gaze[0])] 
            gt = [str(u) for u in gts.cpu().numpy()[k]] 
            log = name + [",".join(gaze)] + [",".join(gt)]
            outfile.write(" ".join(log) + "\n")

        loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
        outfile.write(loger)
        print(loger)

