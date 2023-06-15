from dv import AedatFile
import cv2,torch,os
from tqdm import tqdm
import numpy as np
from readdata import *
from torchvision import transforms
import torch.nn.functional as F

class Bullying10k(torch.utils.data.Dataset):
    def __init__(self,root,train,transform=None,target_transform=None,step=10,gap=2):
        super().__init__()
        self.transform=transform
        self.target_transform=target_transform
        rawroot=os.path.join(root,"Bullying10k_final/")
        nproot=os.path.join(root,"Bullying10k_processed/")
        self.length=step
        self.gap = gap
        if not os.path.exists(nproot):aedat42eventnp(rawroot,self.length)
        classes=os.listdir(nproot)
        classes=[os.path.join(nproot,i) for i in classes]
        
        self.sample=[]
        for label,c in enumerate(classes):
            data=[(i,label)for i in allfile(c)]
            self.sample.extend(data)
        self.sample=np.array(self.sample)
        
        loc=np.zeros(len(self.sample),dtype=bool)
        loc[range(0,len(self.sample),5)]=1
        
        if train:self.sample=self.sample[~loc]
        else:self.sample=self.sample[loc]
 
    def __len__(self):
        return len(self.sample)
        
        
    def __getitem__(self,idx):
        # data=np.load(self.sample[idx][0],allow_pickle=True)
         
        data=readnplong(self.sample[idx][0])
        data=[i for i  in data]
        # data=torch.from_numpy(data).float()
        if self.length>0:
            loc=np.random.choice(len(data)-self.length*self.gap+1)
            # print(loc,data.shape[0],loc+self.length*self.gap)
            data=data[loc:loc+self.length*self.gap :self.gap]
        frame=np.zeros((len(data),2,260,346),dtype=bool)
        
        
        for i in range(len(data)):
            
            frame=event2frame(data[i],i,frame)
        data=frame
        label= int(self.sample[idx][1]) 
        if self.transform is not None :data=self.transform(data)
        if self.target_transform is not None :label=self.target_transform(label)
        
        return data,label
        
 def get_bullying10k_data(batch_size, num_workers=32, same_da=False, **kwargs):


    size=kwargs["size"] if "size" in  kwargs else 48
    
    train_transform = transforms.Compose([lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),
                                          ])
    test_transform = transforms.Compose([lambda x: torch.tensor(x, dtype=torch.float),
        lambda x: F.interpolate(x, size=[size, size], mode='bilinear', align_corners=True),])

    train_datasets = Bullying10k(r"/data/datasets/",train=True , transform=test_transform if same_da else train_transform ,step=kwargs["step"],gap=kwargs["gap"])
    test_datasets = Bullying10k(r"/data/datasets/",train=False , transform=test_transform,step=kwargs["step"] ,gap=kwargs["gap"])

    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_datasets, batch_size=batch_size,
        pin_memory=True, drop_last=False, num_workers=num_workers
    )

    return train_loader, test_loader, False, None
    
 if __name__=="__main__":        
    a=Bullying10klong(r"/data/datasets/",train=True,step=10,gap=10)
    print(a[0][0].shape)
    print(a[0][1])
    # a=Bullying10klong(r"/data/datasets/",train=True,step=-1,gap=1)
    torch.set_num_threads(12 )


    l = torch.utils.data.DataLoader(
        a, batch_size=10,
        pin_memory=True, drop_last=True, shuffle=True, num_workers=0
    )
    # print(    [i[0].shape[0] for i in l])
    sumnum=0
    minnum=9999
    maxnum=0
    for idx,(i,label) in enumerate(l):
        # print(111,i)
        sumnum+=i[0].shape[0]
        if i[0].shape[0]<minnum:minnum=i[0].shape[0]
        if i[0].shape[0]>maxnum:maxnum=i[0].shape[0]
        # print(    idx,i[0].shape[0],sumnum/(idx+1),sumnum ,minnum,maxnum)