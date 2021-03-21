import torch
from torch.utils.data import Dataset
from torchvision import transforms
import config
from utils import *




class XrayImgDataset(Dataset):
  def __init__(self,json_file,phase):
    self.data=read_json(json_file)
    self.img_paths=list(self.data.keys())
    self.bbxs=list(self.data.values())
    self.phase=phase
    self.transforms=torchvision.transforms.ToTensor()

  def convert_offsets(self,bbx,img):
    bboxs=get_offset(bbx,img)
    new_bbx=[[int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2),int(bbox[0] + bbox[2] / 2),int(bbox[1] + bbox[3] / 2)] for bbox in bboxs]
    return new_bbx

    
  def __getitem__(self,idx):
    img_path=self.img_paths[idx]
    img=cv2.imread(img_path)
    bbxs=self.convert_offsets(self.bbxs[idx],img)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    for bbx in bbxs:
      bbx[0]=bbx[0]/img.shape[1]*WIDTH
      bbx[1]=bbx[1]/img.shape[0]*HEIGHT
      bbx[2]=bbx[2]/img.shape[1]*WIDTH
      bbx[3]=bbx[3]/img.shape[0]*HEIGHT
    img=cv2.resize(img,(WIDTH,HEIGHT))
    img=img.astype(float)
    img=self.transforms(img)
    img=img/255
    bbxs=torch.FloatTensor(bbxs)


    targets={}
    targets['boxes']=bbxs
    targets['labels']=torch.ones(bbxs.shape[0]).long()
    return img,targets

  def __len__(self):
    return len(self.data)






if __name__ == '__main__':
	train_dataset=XrayImgDataset(config.train_data_json,'train')
	my_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=8,
    shuffle=True,
    #collate_fn=my_collate_fn,
    )

	item=iter(my_loader).next()

	print(item[1]['labels'])