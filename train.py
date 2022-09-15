# 参考サイト https://github.com/minhtriphan/Kaggle-Competition-FeedbackPrize2-5th-solution/blob/main/train.py#L427


import gc
import os
import pickle
import glob
import time
import copy
import jsons
import argparse
import yaml

from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from transformers import DataCollatorWithPadding

import numpy as np
import pandas as pd
from scipy.special import softmax

from tqdm import tqdm
from collections import defaultdict

# For SWA
from torch.optim.swa_utils import AveragedModel, SWALR

# Mixed precision in Pytorch
from torch.cuda.amp import autocast, GradScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.nn import Parameter


from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import bitsandbytes as bnb


import warnings
warnings.simplefilter('ignore')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import warnings
warnings.simplefilter('ignore')

import wandb

# Weights & Biasesのログイン
wandb.login()




""" Initial configuration """

class Config():
    # General settings
    competition_name = 'FeedbackPrize3'
    env = 'local'
    seed = 3655
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








""" Random seed """

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










""" Dataset """

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




""" Model """


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







def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))









""" One-epoch training function """

def train_fn(cfg, model, train_dataloader, optimizer, epoch, num_train_steps, scheduler, 
             valid_dataloader, val_df, valid, swa_model = None, swa_scheduler = None, best_score = np.inf, fold = 0):
    '''
    val_df: the validation dataframe after re-organizing
    valid: the validation dataframe before re-organizing
    '''
    # Set up for training
    scaler = GradScaler(enabled = cfg.apex)   # Enable APEX
    loss = 0
    total_samples = 0
    global_step = 0
    start = end = time.time()

    if cfg.use_awp:
        # Initialize AWP
        awp = AWP(model, optimizer, adv_lr = cfg.adv_lr, adv_eps = cfg.adv_eps, start_step = cfg.start_awp_epoch, scaler = scaler)

    if cfg.use_tqdm:
        tbar = tqdm(train_dataloader)
    else:
        tbar = train_dataloader
        
    val_schedule = [int(i) for i in list(np.linspace(1, len(tbar), num = int(1 / cfg.val_check_interval) + 1, endpoint = True))[1:]]

    for i, item in enumerate(tbar):
        model.train()
        # Set up inputs
        input_ids = item['input_ids'].to(cfg.device)
        attention_mask = item['attention_mask'].to(cfg.device)
        token_type_ids = item['token_type_ids'].to(cfg.device)
        labels = item['labels'].to(cfg.device)
        
        batch_size = input_ids.shape[0]

        # Forward
        with autocast(enabled = cfg.apex):
            batch_loss, _ = model(input_ids, attention_mask, token_type_ids, labels)

        if cfg.gradient_accumulation_steps > 1:
            batch_loss = batch_loss / cfg.gradient_accumulation_steps

        # Backward
        scaler.scale(batch_loss).backward()       
        
        if cfg.use_awp and epoch >= cfg.start_awp_epoch:
            if epoch == cfg.start_awp_epoch and i == 0:
                logging.info(' Start AWP '.center(50, '-'))
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                awp.attack_backward(item, epoch)

        # Update loss
        loss += batch_loss.item() * batch_size
        total_samples += batch_size

        if cfg.use_tqdm:
            tbar.set_description('Batch loss: {:.4f} - Avg loss: {:.4f}'.format(batch_loss, loss / total_samples))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (i + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if swa_model is not None:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                if cfg.batch_scheduler:
                    scheduler.step()
            optimizer.zero_grad()

        # Evaluate
        if epoch >= cfg.start_val_epoch:
            if (i + 1) in val_schedule:
                logging.info('Epoch: [{0}][{1}/{2}] - Start evaluating...'.format(epoch + 1, i + 1, len(tbar)))
                if swa_model is not None:
                    torch.optim.swa_utils.update_bn(list(train_dataloader)[:(i + 1)], swa_model)
                    val_loss, pred = valid_fn(cfg, swa_model, valid_dataloader)
                else:
                    val_loss, pred = valid_fn(cfg, model, valid_dataloader)

                true = np.array([cfg.label_map[item] for sublist in [i.split('|') for i in val_df['label_list'].tolist()] for item in sublist])
                assert pred.shape[0] == len(true)
                valid[list(cfg.label_map.keys())[:3]] = pred

                # Scoring
                score = metric(pred, true)

                end = time.time()
                logging.info('Epoch: [{0}][{1}/{2}] - '
                            'Elapsed {remain:s} - '
                            'Train Loss: {train_loss:.4f} - '
                            'Val Loss: {val_loss:.4f} - '
                            'Score: {score:.4f} - '
                            'LR: {lr:.8f}'
                            .format(epoch + 1, i + 1, len(tbar), 
                                    remain = timeSince(start, float(i + 1) / len(tbar)),
                                    train_loss = loss / total_samples,
                                    val_loss = val_loss,
                                    score = score,
                                    lr = scheduler.get_lr()[0]))
                if score < best_score:
                    best_score = score
                    logging.info(f'Epoch [{epoch + 1}][{i + 1}/{len(tbar)}] - The Best Score Updated to: {best_score:.4f} Model')
                    if swa_model is None:
                        model_state_dict = {
                            'state_dict': model.state_dict(),
                            'pred': pred,
                        }
                    else:
                        model_state_dict = {
                            'state_dict': swa_model.state_dict(),
                            'pred': pred,
                        }
                    ckp = os.path.join(cfg.model_dir, cfg.model_name.split('_')[0][:-1], cfg.model_name.split('_')[0][-1], f'fold_{fold}.pt')
                    torch.save(model_state_dict, ckp)
                else:
                    logging.info(f'Epoch [{epoch + 1}][{i + 1}/{len(tbar)}] - Not The Best Score ({score:.4f}), Current Best Score: {best_score:.4f} Model')
            
    return best_score, valid







""" One-epoch validating function """

def valid_fn(cfg, model, valid_dataloader):
    # Set up for training
    model.eval()

    loss = 0
    total_samples = 0
    start = end = time.time()

    preds = []

    if cfg.use_tqdm:
        tbar = tqdm(valid_dataloader)
    else:
        tbar = valid_dataloader

    for i, item in enumerate(tbar):
        # Set up inputs
        input_ids = item['input_ids'].to(cfg.device)
        attention_mask = item['attention_mask'].to(cfg.device)
        token_type_ids = item['token_type_ids'].to(cfg.device)
        is_tail = item['is_tail']
        labels = item['labels'].to(cfg.device)

        batch_size = input_ids.shape[0]

        # Forward
        with torch.no_grad():
            with autocast(enabled = cfg.apex):
                batch_loss, batch_pred = model(input_ids, attention_mask, token_type_ids, labels)
        
            # Update loss
            loss += batch_loss.item() * batch_size
            total_samples += batch_size

        # Now, locate only the FP2 tokens
        batch_pred = batch_pred[torch.where(input_ids == cfg.cls_token_id)]

        # Store the predictions
        batch_pred = F.softmax(batch_pred, dim = -1)
        preds.append(batch_pred.detach().cpu().numpy())

        # Logging
        end = time.time()
        
    preds = np.concatenate(preds, axis = 0)

    return loss / total_samples, preds








"""* Optimizer and scheduler"""

def get_optimizer(cfg, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': cfg.encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
             'lr': cfg.decoder_lr, 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters, lr = cfg.lr, eps = cfg.eps, betas = cfg.betas)
    return optimizer

def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps = cfg.num_warmup_steps, num_training_steps = num_train_steps
        )
    elif cfg.scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps = cfg.num_warmup_steps, num_training_steps = num_train_steps, num_cycles = cfg.num_cycles
        )
    return scheduler




def training_loop(cfg, fold = 0):
    logging.info(f' Fold {fold} '.center(50, '*'))
    set_random_seed(cfg.seed + fold)
    
    logging.info('Preparing training and validating dataloader...')
    trn = train_df[train_df.fold != fold]
    val = train_df[train_df.fold == fold]
    valid = train[train.fold == fold]

    train_dataset = FeedbackPrize2_Dataset(cfg, trn, mode = 'train')
    valid_dataset = FeedbackPrize2_Dataset(cfg, val, mode = 'valid')

    train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, shuffle = True, collate_fn = collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size = cfg.batch_size * 8, num_workers = cfg.num_workers, shuffle = False, collate_fn = collate_fn)

    logging.info('Preparing model, optimizer, and scheduler...')
    model = FeedbackPrize2_Model(cfg).to(cfg.device)
    optimizer = get_optimizer(cfg, model)
    num_training_steps = int(len(train_dataloader) * cfg.nepochs)
    scheduler = get_scheduler(cfg, optimizer, num_training_steps)

    best_score = np.inf
    if cfg.mode == 'train':
        for epoch in range(cfg.nepochs):
            start_time = time.time()
            
            if epoch == 4:
                break

            if epoch < cfg.start_swa_epoch:
                # Train
                best_score, valid = train_fn(cfg, model, train_dataloader, optimizer, epoch, num_training_steps, scheduler, 
                                             valid_dataloader, val, valid, best_score = best_score, fold = fold)
            else:
                if epoch == cfg.start_swa_epoch:
                    logging.info(' Enable SWA '.center(50, '-'))
                    swa_model = AveragedModel(model)
                    swa_scheduler = SWALR(optimizer, anneal_strategy = cfg.swa_anneal_strategy, anneal_epochs = cfg.swa_anneal_epochs, swa_lr = cfg.swa_lr)
                best_score, valid = train_fn(cfg, model, train_dataloader, optimizer, epoch, num_training_steps, scheduler, 
                                             valid_dataloader, val, valid, swa_model = swa_model, swa_scheduler = swa_scheduler, 
                                             best_score = best_score, fold = fold)
    else:
        ckp = torch.load(os.path.join(cfg.model_dir, cfg.model_name.split('_')[0][:-1], cfg.model_name.split('_')[0][-1], f'fold_{fold}.pt'), map_location = cfg.device)
        if cfg.use_swa:
            swa_model = AveragedModel(model)
            swa_model.load_state_dict(ckp['state_dict'])
            _, pred = valid_fn(cfg, swa_model, valid_dataloader)
        else:
            model.load_state_dict(ckp['state_dict'])
            _, pred = valid_fn(cfg, model, valid_dataloader)
        _, pred = valid_fn(cfg, model, valid_dataloader)
        true = np.array([cfg.label_map[item] for sublist in [i.split('|') for i in val['label_list'].tolist()] for item in sublist])
        assert pred.shape[0] == len(true)
        valid[list(cfg.label_map.keys())[:3]] = pred

        # Scoring
        best_score = metric(pred, true)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_score, valid





""" Main """

def main():
    oofs = []
    for fold in cfg.training_folds:
        fold_score, oof = training_loop(cfg, fold)
        logging.info(f' Score: {fold_score} '.center(50, '*'))
        oofs.append(oof)
    oofs = pd.concat(oofs)
    if set(cfg.training_folds) == {0, 1, 2, 3, 4}:
        oofs = oofs.loc[train.index]
    pred = oofs[list(cfg.label_map.keys())[:3]].values
    true = oofs['discourse_effectiveness'].map(cfg.label_map).values
    assert pred.shape[0] == len(true)
    # Scoring
    score = metric(pred, true)
    logging.info('=' * 50)
    logging.info(f' OOF Score: {score} '.center(50, '*'))
    # Storing OOF file
    oofs.to_pickle(os.path.join(cfg.model_dir, cfg.model_name.split('_')[0][:-1], cfg.model_name.split('_')[0][-1], f"oof_{''.join([str(i) for i in cfg.training_folds])}.pkl"))



""" yamlから読み込むargparseの設定 """
# https://rightcode.co.jp/blog/information-technology/pytorch-yaml-optimizer-parameter-management-simple-method-complete


def get_args():
    # 引数の導入
    parser = argparse.ArgumentParser(description='YAMLありの例')
    parser.add_argument('config_path', type=str, help='設定ファイル(.yaml)')
    args = parser.parse_args()
    return args
 
 
def main(args):
    # 設定ファイル(.yaml)の読み込み
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
 
    # Model作成
    model = nn.Linear(1, 1)
    optimizer = make_optimizer(model.parameters(), **config['optimizer'])
 
    print(optimizer)




if __name__ == '__main__':
    main(get_args())