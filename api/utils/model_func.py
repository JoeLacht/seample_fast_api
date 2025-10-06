import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModel

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

class_names = ['dew - роса', 'fogsmog - туман/смог', 'frost - мороз', 'glaze - корка льда', 'hail - град', 'lightning - молния', 'rain - дождь', 'rainbow - радуга', 'rime - иней',
               'sandstorm - песчанная буря', 'snow - снег']

def class_id_to_label(i):
    '''
    Input int: class index
    Returns class name
    '''

    labels = class_names
    return labels[i]

def load_pt_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load('weights/resnet_weights_weather.pth', map_location='cpu', weights_only=False))
    model.eval()
    return model

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose(
        [
            T.Resize((224, 224)), 
            T.ToTensor(),
            T.Normalize(mean, std)
        ]
    )
    print(trnsfrms(img).shape)
    return trnsfrms(img).unsqueeze(0)

class MyRuBert(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        for param in self.bert.parameters():
            param.requires_grad = False
        self.linear = nn.Sequential(
            nn.Linear(312, 512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)
        )

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        out = self.linear(cls_emb)
        return out
    
def load_rubert_model():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

    clf = MyRuBert(bert_model)
    clf.load_state_dict(torch.load("weights/rr_rubert_model.pth", map_location='cpu'))
    clf.eval()
    return tokenizer, clf
