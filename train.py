import gc
import os
import pickle
import glob
import time
import copy
import jsons

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import DataCollatorWithPadding

import numpy as np
import pandas as pd
from scipy.special import softmax

from tqdm import tqdm
from collections import defaultdict


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter


from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import bitsandbytes as bnb


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import warnings
warnings.simplefilter('ignore')

import wandb

# Weights & Biasesのログイン
wandb.login()




"""# Initial configuration"""

class Config():
    # General settings
    competition_name = 'FeedbackPrize2'
    env = 'local'
    seed = 1
    mode = 'train'    # 'train', 'valid'
    debug = False
    model_name = 'v1a'
    processed_data = True    # Set to True if the train_df is processed and added with the target (rank)
    use_tqdm = True
    # For model
    backbone = 'microsoft/deberta-v3-large'
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    config = AutoConfig.from_pretrained(backbone)
    config.output_hidden_states = True
    config.hidden_dropout_prob = 0.
    config.attention_probs_dropout_prob = 0.
    # Add new token
    cls_token = '[FP2]'
    special_tokens_dict = {'additional_special_tokens': [cls_token]}
    tokenizer.add_special_tokens(special_tokens_dict)
    cls_token_id = tokenizer(cls_token)['input_ids'][1]
    # For data
    discourse_type_map = {
        'Lead': 0,
        'Position': 1,
        'Claim': 2,
        'Counterclaim': 3,
        'Rebuttal': 4,
        'Evidence': 5,
        'Concluding Statement': 6,
    }
    label_map = {
        'Ineffective': 0,
        'Adequate': 1,
        'Effective': 2,
        'None': -100,
    }
    training_folds = [0]
    max_len = 1024
    batch_size = 4
    num_workers = os.cpu_count()
    # For training
    apex = True
    gradient_checkpointing = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    nepochs = 5
    start_val_epoch = 2
    val_check_interval = 0.1
    gradient_accumulation_steps = 1.
    max_grad_norm = 1000
    label_smoothing = 0.03
    # For SWA
    use_swa = False
    if use_swa:
        start_swa_epoch = 2
        swa_lr = 5e-6
        swa_anneal_strategy = 'cos'
        swa_anneal_epochs = 1
    else:
        start_swa_epoch = nepochs + 1
    # For AWP
    use_awp = True
    if use_awp:
        start_awp_epoch = 2
        adv_lr = 1e-5
        adv_eps = 1e-3
        adv_step = 1
    else:
        start_awp_epoch = nepochs + 1
    # Optimizer
    lr = 1e-5
    weight_decay = 1e-2
    encoder_lr = 1e-5
    decoder_lr = 1e-5
    min_lr = 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    # Scheduler
    scheduler_type = 'cosine'    # 'linear', 'cosine'
    if scheduler_type == 'cosine':
        num_cycles = 0.5
    num_warmup_steps = 100
    batch_scheduler = True
    # Directories
    if env == 'colab':
        comp_data_dir = f'/content/drive/My Drive/Kaggle competitions/{competition_name}/comp_data'
        extra_data_dir = f'/content/drive/My Drive/Kaggle competitions/{competition_name}/extra_data'
        model_dir = f'/content/drive/My Drive/Kaggle competitions/{competition_name}/model'
        os.makedirs(os.path.join(model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]), exist_ok = True)
        old_comp_data_dir = f'/content/drive/My Drive/Kaggle competitions/{competition_name[:-1]}/data'
    elif env == 'kaggle':
        comp_data_dir = ...
        extra_data_dir = ...
        model_dir = ...
    elif env == 'vastai':
        comp_data_dir = 'data'
        extra_data_dir = 'data'
        model_dir = 'model'
        os.makedirs(os.path.join(model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]), exist_ok = True)
    elif env == 'local':
        comp_data_dir = 'data'
        extra_data_dir = 'ext_data'
        model_dir = 'model'
        os.makedirs(os.path.join(model_dir, model_name.split('_')[0][:-1], model_name.split('_')[0][-1]), exist_ok = True)

cfg = Config()








"""# Random seed"""

def set_random_seed(seed, use_cuda = True):
    np.random.seed(seed) # cpu vars
    torch.manual_seed(seed) # cpu  vars
    random.seed(seed) # Python
    os.environ['PYTHONHASHSEED'] = str(seed) # Python hash building
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
        
set_random_seed(cfg.seed)










"""# Random seed"""

class FB_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.type = df['discourse_type'].values
        self.essay = df['essay_text'].values
        self.targets = df['discourse_effectiveness'].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        discourse = self.discourse[index]
        type = self.type[index]
        essay = self.essay[index]
        text = type + " " + discourse + " " + self.tokenizer.sep_token + " " + essay
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len
                    )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }




"""# Model"""


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, cfg).__init__()
        self.cfg = cfg

        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        self.config = AutoConfig.from_pretrained(cfg.model_name)
        # 新しい単語の追加，https://qiita.com/m__k/items/e620bc14fba7a4e36824
        self.backbone.resize_token_embeddings(len(cfg.tokenizer))
        if cfg.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        

        self.Linear = nn.Linear(self.config.hidden_size, cfg.num_classes)


        # Multidropout
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)

        self._init_weights(self.Linear)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def criterion(self, pred, true):
        loss = nn.CrossEntropyLoss(ignore_index = -100, label_smoothing = cfg.label_smoothing)(pred.permute(0, 2, 1), true)
        return loss

    def forward(self, input_ids, attention_mask, label = None):
        out = self.backbone(input_ids = input_ids, attention_mask = attention_mask,
                         output_hidden_states = False)
        
        # ノートブックを参考に追加
        sequence_output = out[0][:, 0, :]
        logits1 = self.Linear(self.dropout1(sequence_output))
        logits2 = self.Linear(self.dropout2(sequence_output))
        logits3 = self.Linear(self.dropout3(sequence_output))
        logits4 = self.Linear(self.dropout4(sequence_output))
        logits5 = self.Linear(self.dropout5(sequence_output))
        outputs = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        if label is not None:
            loss = self.criterion(output, label)
        else:
            loss = None
        return loss, output
        
        return outputs
