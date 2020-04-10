# loader.py

!pip install medicaltorch

import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from medicaltorch import transforms as mt_transforms
import PIL
from random import sample

from torch.autograd import Variable

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class Dataset(data.Dataset):
    def __init__(self, datadirs, diagnosis, use_gpu, transformbool):
        super().__init__()
        self.use_gpu = use_gpu

        self.transformbool = transformbool
        self.transforms = torchvision.transforms.Compose([
        #    torchvision.transforms.ToPILImage(),
            #torchvision.transforms.Resize((224,224)),
            #torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        #    torchvision.transforms.RandomHorizontalFlip(p=1.0),
        #    torchvision.transforms.RandomRotation((20,20), resample=PIL.Image.BILINEAR),
            #torchvision.transforms.RandomCrop((224,224),pad_if_needed=True),
            #torchvision.transforms.Resize((224,224)),
        #    torchvision.transforms.ToTensor()
        
            mt_transforms.CenterCrop2D((200,200)),
            mt_transforms.RandomAffine(degrees = 0.0, scale = (1, 1), translate=(0,0)),
            mt_transforms.ToTensor()

         ])
        
        self.train_transforms = torchvision.transforms.Compose([
            mt_transforms.Resample(0.25, 0.25),
            mt_transforms.ElasticTransform(alpha_range=(40.0, 60.0),
                                           sigma_range=(2.5, 4.0),
                                           p=0.3),
            mt_transforms.ToTensor()]
        )
        
        label_dict = {}
        self.paths = []
        print(datadirs)

        for i, line in enumerate(open('metadata.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            path = line[10]
            label = line[2]
            label_dict[path] = int(int(label) > diagnosis)

        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir+'/'+file)

        self.labels = [label_dict[path[6:]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

      
    # Data augmentation section
    # can go through each cases, looking at the histogram of 3T vs 1.5T (naive distribution of contrast data?)
    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as file_handler: # Must use 'rb' as the data is binary
            vol = pickle.load(file_handler).astype(np.int32)
        
        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        #print('pad', pad)
        vol = vol[:,pad:-pad,pad:-pad]
        #vol = vol[pad:-pad,pad:-pad,:]
  
        # see if theres a way to reformat an image from 196 to 224 
        # something called interpolate, scikit image. 
        # consider scipy zoom too?
        problemflag = False
        issue = True

        if not(vol.shape[1] == 224) or not(vol.shape[2] == 224):
          #print('problem vol shape', vol.shape)
          delta_1 = (INPUT_DIM - vol.shape[1]) // 2
          delta_2 = (INPUT_DIM - vol.shape[2]) // 2
          padding = (delta_1, delta_2)
          
          new_vol = np.zeros((vol.shape[0], 224, 224), dtype=np.int32)
          for slice in range(vol.shape[0]):
            vol_slice = vol[slice,:,:]
            img_slice = PIL.Image.fromarray(vol_slice)
            new_vol[slice,:,:] = np.array(PIL.ImageOps.fit(img_slice, [224, 224]), dtype='i')
          vol = new_vol  
          vol.astype(np.int32)
          problemflag = True
          #print('vol shape', vol.shape)
          #print('vol type', vol.dtype)
        flag = False
        randomangle = 0
        if self.transformbool:
          if np.random.rand(1) < 0.5:
            flag = True
            randomangle = np.random.uniform(-20,20)
            self.transforms = torchvision.transforms.Compose([
              torchvision.transforms.ToPILImage(),
              #torchvision.transforms.Resize((224,224)),
              torchvision.transforms.RandomHorizontalFlip(p=1.0),
              torchvision.transforms.RandomRotation((randomangle,randomangle), resample=PIL.Image.BILINEAR),
              #torchvision.transforms.RandomCrop((224,224),pad_if_needed=True),
              torchvision.transforms.ToTensor()
          ])

        """
        # see if this transform policy can take in a 3d volume
        #vol = np.asarray(vol).astype(np.uint8)
        # save the 15th slice to see if transformation happened
        #print('volume mean', np.mean(vol, axis=0))
        #save_fig = PIL.Image.fromarray(np.uint8(np.array(vol[15,:,:] * 255)))
        save_fig = PIL.Image.fromarray(np.uint8(np.array(vol[15,:,:])))
        save_fig = save_fig.convert("L")
        newpath = str(path).replace('/', ' ')
        file_name = "path" + newpath + " transform" + str(flag) + " angle" + str(round(randomangle)) + "_0.png"
        save_path = Path(rundir) / "images1" /  file_name
        save_fig.save(save_path)
        """

        #vol = vol.transpose(2,1,0)
        #print('vol shape', vol.shape)
        #print('vol sum', np.sum(vol))

        #print('vol type', vol.dtype)
        if flag:
          for sliceindex in range(vol.shape[0]):
            vol[sliceindex] = self.transforms(np.array(vol[sliceindex]))


        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol) + 1.0e-6) * MAX_PIXEL_VAL

        """
        save_fig = PIL.Image.fromarray(np.uint8(np.array(vol[15,:,:])))
        save_fig = save_fig.convert("L") 
        #save_fig.show()
        #file_name = f'path{path}_transform{flag}_angle{randomangle:0.4f}'
        newpath = str(path).replace('/', ' ')
        file_name = "path" + newpath + " transform" + str(flag) + " angle" + str(round(randomangle)) + "_1.png"
        save_path = Path(rundir) / "images2" /  file_name
        save_fig.save(save_path)

        if problemflag:
          save_fig = PIL.Image.fromarray(np.uint8(np.array(vol[15,:,:])))
          save_fig = save_fig.convert("L") 
          #save_fig.show()
          #file_name = f'path{path}_transform{flag}_angle{randomangle:0.4f}'
          newpath = str(path).replace('/', ' ')
          file_name = "path" + newpath + " transform" + str(flag) + " angle" + str(round(randomangle)) + "_1.png"
          save_path = Path(rundir) / "problems" /  file_name
          save_fig.save(save_path)

        """  
        # normalize
        # problems with the normalization, fix
        # vol = (vol - MEAN) / STDDEV


        #print('vol1', vol.shape)

        #print('vol shape', vol.shape)

        # convert to RGB
        #vol = np.stack((vol,)*3, axis=1)
        #print('vol2', vol.shape)

        #new_vol = self.transforms(vol).float()
        #print('new_vol', new_vol.shape)

        #assert(1==2)
        #if self.transformbool:
        #  vol = np.mean(self.transforms(vol).float(), axis=(0,1))

        """
        save_fig = PIL.Image.fromarray(np.uint8(np.array(vol[15,:,:])))
        save_fig = save_fig.convert("L")
        #save_fig.show()
        #file_name = f'path{path}_transform{flag}_angle{randomangle:0.4f}'
        newpath = str(path).replace('/', ' ')
        file_name = "path" + newpath + " transform" + str(flag) + " angle" + str(round(randomangle)) + "_2.png"
        save_path = Path(rundir) / "images1" /  file_name
        save_fig.save(save_path)
        """

        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])
        #print('vol tensor shape', vol_tensor.shape)
        #print('label_tensor shape', label_tensor.shape)
        #assert(1==3)
        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(self, diagnosis, use_gpu=True):

    #vol_list = ['vol01', 'vol02', 'vol03', 'vol04', 'vol05', 'vol06', 'vol07', 'vol08', 'vol09', 'vol10']
    #vol_ind = [0,1,2,3,4,5,6,7,8,9]
    
    #train_ind = sample(vol_ind, 6)
    #train_dirs = [vol_list[i] for i in train_ind]

    #val_test_ind = [index for index in vol_ind if index not in train_ind]
    #val_ind = sample(val_test_ind, 2)
    #valid_dirs = [vol_list[i] for i in val_ind]

    #test_ind = [index for index in val_test_ind if index not in val_ind]
    #test_dirs = [vol_list[i] for i in test_ind]
    train_dirs = ['vol08','vol04','vol03','vol09','vol06','vol07']
    valid_dirs = ['vol10','vol05']
    test_dirs = ['vol01','vol02']
    
    #train_dataset = Dataset(train_dirs, diagnosis, use_gpu)
    train_dataset = Dataset(train_dirs, diagnosis, use_gpu, True)
    valid_dataset = Dataset(valid_dirs, diagnosis, use_gpu, False)
    test_dataset = Dataset(test_dirs, diagnosis, use_gpu, False)

    train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, num_workers=8, shuffle=False)

    return train_loader, valid_loader, test_loader
