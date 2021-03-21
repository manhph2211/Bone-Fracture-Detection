import torch
import torchvision
from dataset import XrayImgDataset
import config
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


train_data=XrayImgDataset(config.train_data_json,'train')
train_dataloader = torch.utils.data.DataLoader(
   train_data,
   batch_size=config.BATCH_SIZE,
   num_workers=8,
   shuffle=True,
   collate_fn=lambda x: list(zip(*x)),
   )


val_data=XrayImgDataset(config.test_data_json,'val')
val_dataloader = torch.utils.data.DataLoader(
   val_data,
   batch_size=config.BATCH_SIZE,
   num_workers=8,
   shuffle=True,
   collate_fn=lambda x: list(zip(*x)),
   )


model=torchvision.models.detection.fastersrcnn_resnet50_fpn(pretrained=True)
in_features=model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,config.CLASS_N)
model=model.to(config.device)
params=[p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)



best_point=9999
for epoch in range(config.EPOCH_N):
	print("Training...")
	train_loss=0
	val_loss=0
	for imgs,targets in tqdm(train_dataloader):
		model.train()
		model=model.double()
		optimizer.zero_grad()
		imgs=list(img.to(device) for img in imgs)
		targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
		loss_dict=model(imgs,targets)
		losses=sum(loss for loss in loss_dict.values())   	
		losses.backward()
		optimizer.step()

		train_loss+=losses.item()

	for imgs,targets in tqdm(val_dataloader):
		model.eval()
		imgs=list(img.to(device) for img in imgs)
		targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
		loss_dict_val=model(imgs,targets)
		losses_val=sum(loss for loss in loss_dict.values())
		val_loss+=losses_val
		if losses.item()<best_point:
			best_point=losses.item()
			torch.save(model.state_dict(), config.model_save_path)

		#print(losses.item())

	print("------------------------->", train_loss,val_loss)




