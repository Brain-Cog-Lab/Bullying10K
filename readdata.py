
from dv import AedatFile
import cv2,torch,os
from tqdm import tqdm
import numpy as np
import time
import cv2 as cv
 
def event2frame(event,index,frame):#[t,x,y,p] h w
      
 
    if event is not None:
        frame[index,event["polarity"],event["y"],event["x"]]=1
          
    return frame#[n,h,w]
 
def readaedat4(filename):
   
     
    frame=[]     
    whole=[]

    start=0
    empty=None
     
    with AedatFile(filename) as f:

        for index,e in enumerate(f["events"].numpy() ):
            t=e

             
            if   t[0][0] -start>15000 and index>0:
                whole.extend( [empty for i  in range((t[0][0] -start)//10000-1)])
                if (t[0][0] -start)%10000>5000:whole.append(empty )
            whole.append(t )
                
            start=t[0][0]
        t=None
        frame=np.zeros((len(whole),2,260,346),dtype=bool)    
        for index,e in enumerate(whole)   :  
            if e is None :
                frame=event2frame(t ,index,frame) 
            else:
                frame= event2frame(e ,index,frame)  
                t=e
       
    return frame
def readnp(filename):
   
     
    frame=[]     
    whole=[]

    start=0
    empty=None
     
    f=np.load(filename,allow_pickle=True)   

    for index,e in enumerate(f):
        t=e

            
        if   t[0][0] -start>15000 and index>0:
            whole.extend( [empty for i  in range((t[0][0] -start)//10000-1)])
            if (t[0][0] -start)%10000>5000:whole.append(empty )
        whole.append(t )
            
        start=t[0][0]
    t=None
    frame=np.zeros((len(whole),2,260,346),dtype=bool)    
    for index,e in enumerate(whole)   :  
        if e is None :
            frame=event2frame(t ,index,frame) 
        else:
            frame= event2frame(e ,index,frame)  
            t=e
       
    return frame
def readnplong(filename):
   
     
    frame=[]     
    whole=[]

    start=0
    empty=None
     
    f=np.load(filename,allow_pickle=True)   

    for index,e in enumerate(f):
        t=e

            
        if   t[0][0] -start>15000 and index>0:
            whole.extend( [empty for i  in range((t[0][0] -start)//10000-1)])
            if (t[0][0] -start)%10000>5000:whole.append(empty )
        whole.append(t )
            
        start=t[0][0]
    t=None
    frame=[] 
    for index,e in enumerate(whole)   :  
        if e is None :
            frame.append(t  ) 
        else:
            frame.append(e  )  
            t=e
       
    return frame
def readevent(filename):
 
     
    with AedatFile(filename) as f:
 
       
        return [i for i in f["events"].numpy()]
    
def allfile(path):
    fileall=[]
    for curpath,dirlist,filelist in os.walk(path):
        for i in filelist:
            fileall.append(os.path.join(curpath, i))
             
            
    return fileall
    
def aedat42np(root,length):
        rawroot=os.path.join(root)
        nproot=os.path.join("/data/datasets/Bullying10k_processed")
        classes=os.listdir( rawroot)
        

        for label,c in enumerate(classes):
             
            craw=os.path.join(rawroot,c) 
            cnp=os.path.join(nproot,c)
            for i in tqdm(allfile(craw)) :
                s=os.path.join(cnp,*i.split("/")[-2:-1])
                
                if not os.path.exists(s):os.makedirs(s)
                s=os.path.join(s,i.split("/")[-1][:-7])
                #print(s)
                data=readaedat4(i)
                if data.shape[0]>=length:
                    np.save(s,data)
def aedat42eventnp(root,length):
        rawroot=os.path.join(root)
        nproot=os.path.join("/data/datasets/Bullying10k_processed")
        classes=os.listdir( rawroot)
        

        for label,c in enumerate(classes):
             
            craw=os.path.join(rawroot,c) 
            cnp=os.path.join(nproot,c)
            for i in tqdm(allfile(craw)) :
                s=os.path.join(cnp,*i.split("/")[-2:-1])
                
                if not os.path.exists(s):os.makedirs(s)
                s=os.path.join(s,i.split("/")[-1][:-7])
                #print(s)
                data=readevent(i)
                # if data.shape[0]>=length:
                np.save(s,data)
def readvideo(filename,s):
    i=0
    frame=None
    allframe=None    
    
    with AedatFile(filename) as f:
     
            # 获取视频帧的宽和高
        w = 346
        h = 260

        # 获取视频总帧数和fps
     
        fps = 24
     
        # 视频保存
        # fourcc = cv.VideoWriter_fourcc('P', 'I', 'M', '1')
        # fourcc = cv.VideoWriter_fourcc(*'XVID')
        # 视频编码格式
        
        fourcc = cv.VideoWriter_fourcc(*'mp4v')

        out = cv.VideoWriter(s+'.mp4', fourcc, 24, (int(w), int(h)), True)
        i=0
        tmp=None  
        for e in f["frames"] :#一张图10ms 4个组为一张
        
            # if i >86:break
            # out.write (e.image)
            if i==0:
                out.write(e.image)
                i=e.timestamp
                continue
            for j  in range( (e.timestamp-i)//50000):
                out.write(e.image) 
                if (e.timestamp-i)//50000>5:break
            if (e.timestamp-i)%50000>30000:
                out.write(e.image)   
            i=e.timestamp
 
