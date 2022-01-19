import torch

from transformers import BertTokenizer
from PIL import Image

from models import caption, build_model
from datasets import coco, utils
from configuration import Config
import os
import sys 
import matplotlib.pyplot as plt
import numpy as np
import scipy

image_path = sys.argv[1]
save_path = sys.argv[2]
version = 'v3'
checkpoint_path = None

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])

my_model, _ =  build_model(config)
my_model.load_state_dict(model.state_dict())

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

image = Image.open(image_path)
eval_img = np.array(image)
image = coco.val_transform(image)
image = image.unsqueeze(0)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

att_w_list = []
shapes = []

@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions, att_w, shape = my_model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == 102:
            return caption

        att_w_list.append(att_w[-1,0,i])
        shapes.append(shape)
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        
    return caption


output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

#result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result.capitalize())

words = result.split(' ')
atts = att_w_list[:-1]  #Skip the end dot

fig = plt.figure(figsize=(13,13))
columns = 5
rows = int(np.ceil(len(words)/5))

for i, (att, word, shape) in enumerate(zip(atts, words, shapes)):
    att = att.detach().cpu().numpy().reshape(shape[0], shape[1])
    im = Image.fromarray(att)
    att = np.array(im.resize((eval_img.shape[1], eval_img.shape[0]), resample=Image.BILINEAR))
    fig.add_subplot(rows, columns, i+1)
    plt.title(word)
    plt.imshow(eval_img)
    plt.imshow(att, cmap='jet', alpha=0.3)
    plt.axis('off')
plt.savefig(save_path, bbox_inches='tight')
 