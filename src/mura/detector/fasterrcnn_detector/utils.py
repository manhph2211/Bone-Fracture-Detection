import os
import glob
import cv2
import matplotlib.pyplot as plt
import json
import config
import cv2


def get_anno(txt_path):
    annos=[]
    with open(txt_path) as f:
      for line in f.readlines():
        anno=line[:-1].split(" ")
        anno = [float(num) for num in anno]
        annos.append(anno[1:])
    return annos



def get_offset(annos,image): # for single image
  new_annos=[]
  for annotation in annos:
    bbox = [annotation[0] * image.shape[1], annotation[1] * image.shape[0], 
            annotation[2] * image.shape[1], annotation[3] * image.shape[0]]
    new_annos.append(bbox)
  return new_annos


def get_path(data_path):
  img_paths = glob.glob(os.path.join(data_path,"*.png"))
  dic={}
  for img_path in img_paths:
    txt_path = img_path.replace('.png','.txt')
    dic[img_path]=get_anno(txt_path)
  return dic


  # Predict label: [0 cx cy w h]
def my_draw(img_path,anno):
  image=cv2.imread(img_path)
  print(image.shape)
  bboxs=get_offset(anno,image)
  for bbox in bboxs:
    cv2.rectangle(image, (int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)), (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)), (225, 0, 0), 2)
  plt.figure(figsize = (10, 10))
  plt.imshow(image)
  plt.show()


def read_json(path):
  with open(path,'r') as f:
    data = json.load(f)
  return data

def write_json(dic,path):
  with open(path,'w') as f:
    json.dump(dic,f,indent=4)



if __name__ == '__main__':
	dic_train=get_path(config.data_train_path)
	dic_test=get_path(config.data_test_path)
	write_json(dic_train,config.train_data_json)
	write_json(dic_test,config.test_data_json)
	test_dic=read_json('../data/test_data.json')
	item=next(iter(test_dic.items()))
	my_draw(item[0],item[1])