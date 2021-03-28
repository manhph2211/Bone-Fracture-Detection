import torch
import torchvision
import config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import XrayImgDataset


def view(images,labels,k,std=1,mean=0):
  figure=plt.figure(figsize=(30,30))
  images=list(images)
  labels=list(labels)
  for i in range(k):
    out=torchvision.utils.make_grid(images[i])
    inp=out.cpu().numpy().transpose((1,2,0))
    inp=np.array(std)*inp+np.array(mean)
    inp=np.clip(inp,0,1)  
    ax = figure.add_subplot(2,2, i + 1)
    ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
    l=labels[i]['boxes'].cpu().numpy()
    l[:,2]=l[:,2]-l[:,0]
    l[:,3]=l[:,3]-l[:,1]
    for j in range(len(l)):
      ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=2,edgecolor='w',facecolor='none')) 



test_data=XrayImgDataset(config.test_data_json,'val')
test_dataloader = torch.utils.data.DataLoader(
   test_data,
   batch_size=config.BATCH_SIZE,
   num_workers=8,
   shuffle=True,
   collate_fn=lambda x: list(zip(*x)),
   )

images,targets=next(iter(test_dataloader))

model=torchvision.models.detection.fastersrcnn_resnet50_fpn(pretrained=True)
in_features=model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,config.CLASS_N)
model=model.to(config.device)
model.load_state_dict(torch.load(config.model_save_path))


images = list(image.to(device) for image in images)
targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

model.eval()
model=model.double()
output=model(images)

with torch.no_grad():
    view(images,output,2)


