import os
import torch
import numpy as np
import faiss
from PIL import Image

PATH_IMAGE = []

def offline_process_text2pic(dataset_path, preprocess_train, model):
  file_list_off = []
  PATH_IMAGE.append(dataset_path)
  label = os.listdir(dataset_path)
  for i in label:
    test = os.listdir(dataset_path + '/' + i)
    file_list_1 = [i + '/' + item for item in test]
    file_list_off.extend(file_list_1)

  images = []
  for file_name in file_list_off:
    file_path = os.path.join(dataset_path, file_name)
    img = Image.open(file_path)
    images.append(img)

  index = faiss.IndexFlatL2(768)

  image_input = torch.cat([preprocess_train(img).unsqueeze(0) for img in images])
  with torch.no_grad():
      image_features = model.encode_image(image_input)

  index.add(np.array(image_features.cpu()).astype(np.float32))
  faiss.write_index(index, "vectorstores/index_text2pic.bin")


def calculate_cosine_similarity(text_features, image_features):
  similarity = torch.nn.functional.cosine_similarity(text_features, image_features, dim=-1)
  return similarity


def find_similar_text2pic(caption, loaded_index, model, tokenizer):
  file_list_on = []
  if PATH_IMAGE != []: label = os.listdir(PATH_IMAGE[-1])
  else: label = os.listdir("../data/data_process_full/Images")
  for i in label:
    if PATH_IMAGE != []: test = os.listdir(PATH_IMAGE[-1] + '/' + i)
    else: test = os.listdir("../data/data_process_full/Images" + '/' + i)
    file_list_1 = [i + '/' + item for item in test]
    file_list_on.extend(file_list_1)
  
  num_features = loaded_index.ntotal
  image_features = np.zeros((num_features, loaded_index.d), dtype=np.float32)
  loaded_index.reconstruct_n(0, num_features, image_features)
  image_features = torch.from_numpy(image_features)

  text_inputs = torch.cat([tokenizer(caption)])
  text_features = model.encode_text(text_inputs)

  similarity = calculate_cosine_similarity(text_features, image_features)
  values, indices = similarity.topk(5)

  similar = []
  for value, index in zip(values, indices):
    if PATH_IMAGE != []: similar.append([PATH_IMAGE[-1] + '/' + file_list_on[index.item()], value.item()])
    else: similar.append(["../data/data_process_full/Images" + '/' + file_list_on[index.item()], value.item()])

  return similar
