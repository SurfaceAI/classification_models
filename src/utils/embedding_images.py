import sys
sys.path.append('.')

from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torchvision import transforms, models
from functools import partial
from sentence_transformers import SentenceTransformer, util
import torch
import os
from pathlib import Path
from PIL import Image
import pickle

from src.utils import helper, preprocessing
from src.models import prediction

config = {
    'embedding_model_name': 'surface-efficientNetV2SLinear-20240408_135216-sd61xphn_epoch5.pt', # 'dino', 'mask2former', 'efficientnet_v2_s', 'clip'
    # 'embedding_model_name': 'dino',
    'feature_layer': 'features', # for effnet only!
    'root_path': str(Path().resolve()),
    'root_data': "data/training",
    'dataset': 'V200/sorted_images',
    # 'dataset': 'V11/annotated/asphalt/bad',
    'model_folder': 'trained_models',
    'crop': 'lower_middle_half',
    'root_output': 'embeddings',
    'name_output': 'V200_effnet_cropped_sim_embeddings.pkl',
    'gpu_kernel': 0,
}

device = torch.device(
    f"cuda:{config.get('gpu_kernel')}" if torch.cuda.is_available() else "cpu"
)

embedding_model_name = config.get('embedding_model_name')

root_data = os.path.join(config.get('root_path'), config.get('root_data'))

images_path = os.path.join(config.get('root_path'), config.get('root_data'), config.get('dataset'))
embeddings_path = os.path.join(config.get('root_path'), config.get('root_output'))
if not os.path.exists(embeddings_path):
    os.mkdir(embeddings_path)
embeddings_file = os.path.join(embeddings_path, config.get('name_output'))

crop = config.get('crop')

# output is a 2D matrix!
class mask2former_embed():
    def __init__(self) -> None:

        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-semantic")
        self.feature_layer = self.mask2former.model.pixel_level_module.encoder.encoder.layers[3].blocks[1].output
        # self.feature_layer = self.mask2former.model.pixel_level_module.encoder.embeddings

    def embed(self, image):
        img = self.processor(images=image, return_tensors="pt").to(device)

        # Move mask2former to GPU
        self.mask2former.to(device)
        self.mask2former.eval()

        # run inference
        with torch.no_grad() and helper.ActivationHook(self.feature_layer) as activation_hook:
            outputs = self.mask2former(**img)
            embedding = activation_hook.activation[0]

        return embedding
    
class dino_embed():
    def __init__(self) -> None:

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

    @staticmethod
    def transform(image):
        transform = transforms.Compose([ 
            transforms.Lambda(partial(preprocessing.custom_crop, crop_style=crop)),
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ]) 
        return transform(image)

    def embed(self, image):
        img = self.transform(image).unsqueeze(0).to(device)

        # Move model to GPU
        self.model.to(device)
        self.model.eval()

        # run inference
        with torch.no_grad():
            embedding = self.model(img)

        return embedding.squeeze(0)
    
class clip_embed():
    def __init__(self) -> None:

        self.model = SentenceTransformer('clip-ViT-B-32')

    def embed(self, image):
        # run inference
        image = preprocessing.custom_crop(image, crop_style=crop)
        embedding = self.model.encode(image, convert_to_tensor=True)

        return embedding

class trained_model_embed():
    def __init__(self, model, layer) -> None:

        self.model, _, _, _ = prediction.load_model(model, device)
        if layer == 'features':
            self.feature_layer = self.model.features
        elif layer == '7':
            self.feature_layer = self.model.features[7]
        elif layer == '6':
            self.feature_layer = self.model.features[6]
        elif layer == '5':
            self.feature_layer = self.model.features[5]
        elif layer == '4':
            self.feature_layer = self.model.features[4]
        elif layer == '3':
            self.feature_layer = self.model.features[3]
        elif layer == '2':
            self.feature_layer = self.model.features[2]
        elif layer == '1':
            self.feature_layer = self.model.features[1]
        elif layer == '0':
            self.feature_layer = self.model.features[0]
        else:
            self.feature_layer = self.model.features
            print('no valid layer, feature layer is default.')
        self.avgpool = self.model.avgpool
        

    @staticmethod
    def transform(image):
        transform = transforms.Compose([ 
            transforms.Lambda(partial(preprocessing.custom_crop, crop_style=crop)),
            transforms.Resize((384, 384)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ]) 
        return transform(image)

    def embed(self, image):
        img = self.transform(image).unsqueeze(0).to(device)
        

        # Move model to GPU
        self.model.to(device)
        self.model.eval()

        # run inference
        with torch.no_grad() and helper.ActivationHook(self.feature_layer) as activation_hook:
            _ = self.model(img)
            embedding = activation_hook.activation.squeeze(0)
        
        embedding = torch.flatten(self.avgpool(embedding))

        return embedding
    
class pretrained_model_embed():
    def __init__(self, model, layer) -> None:
        if model == 'efficientnet_v2_s':
            self.model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')

        if layer == 'features':
            self.feature_layer = self.model.features
        elif layer == '7':
            self.feature_layer = self.model.features[7]
        elif layer == '6':
            self.feature_layer = self.model.features[6]
        elif layer == '5':
            self.feature_layer = self.model.features[5]
        elif layer == '4':
            self.feature_layer = self.model.features[4]
        elif layer == '3':
            self.feature_layer = self.model.features[3]
        elif layer == '2':
            self.feature_layer = self.model.features[2]
        elif layer == '1':
            self.feature_layer = self.model.features[1]
        elif layer == '0':
            self.feature_layer = self.model.features[0]
        else:
            self.feature_layer = self.model.features
            print('no valid layer, feature layer is default.')
        self.avgpool = self.model.avgpool
        

    @staticmethod
    def transform(image):
        transform = transforms.Compose([ 
            transforms.Resize((384, 384)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
        ]) 
        return transform(image)

    def embed(self, image):
        img = self.transform(image).unsqueeze(0).to(device)

        # Move model to GPU
        self.model.to(device)
        self.model.eval()

        # run inference
        with torch.no_grad() and helper.ActivationHook(self.feature_layer) as activation_hook:
            _ = self.model(img)
            embedding = activation_hook.activation.squeeze(0)
        
        embedding = torch.flatten(self.avgpool(embedding))

        return embedding

images = []
for root, _, fnames in sorted(os.walk(images_path, followlinks=True)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        if fname.endswith((".jpg", ".jpeg", ".png")):
            images.append(path)

if embedding_model_name == 'mask2former':
    embedding_model = mask2former_embed()
elif embedding_model_name == 'dino':
    embedding_model = dino_embed()
elif embedding_model_name == 'clip':
    embedding_model = clip_embed()
elif embedding_model_name == 'efficientnet_v2_s':
    embedding_model = pretrained_model_embed(embedding_model_name, config.get('feature_layer'))
elif 'efficientNet' in embedding_model_name:
    model_file = os.path.join(config.get('root_path'), config.get('model_folder'), embedding_model_name)
    embedding_model = trained_model_embed(model_file, config.get('feature_layer'))


embeddings = []
for image_path in images:
    image = Image.open(image_path)
    embedding = embedding_model.embed(image)
    embeddings.append(embedding)

result = torch.stack(embeddings).cpu()

with open(embeddings_file, "wb") as f_out:
    pickle.dump({'images': images, 'embeddings': result}, f_out, protocol=pickle.HIGHEST_PROTOCOL)

print('Done.')