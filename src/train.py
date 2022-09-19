# 参考リンク(https://qiita.com/haraso_1130/items/20a50b0474c88781dcc1)

import os
import gc
import yaml
import random
import math

import numpy as np
import pandas as pd


from tqdm import tqdm


import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Mixed precision in Pytorch
from torch.cuda.amp import autocast, GradScaler

from transformers import AutoModel, AutoTokenizer, AutoConfig, AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding

# transformersの警告を出力しないようにする
from transformers import logging
logging.set_verbosity_warning()

# 8bit-optimizer
import bitsandbytes as bnb

# cudaのエラーを表示するように設定
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 警告を表示しないようにする
os.environ['TOKENIZERS_PARALLELISM'] = 'true'



""" wandb """
import wandb
# Weights & Biasesのログイン
wandb.login()



""" hydra """
import hydra
from omegaconf import DictConfig



""" seed """
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
        


""" Dataset """
class FB3_Dataset(Dataset):
    def __init__(self,cfg, df):
        self.cfg = cfg
        self.df = df
        self.max_len = cfg.setting.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name,  use_fast=True)
        self.text = df['full_text'].values
        self.targets = df[self.cfg.setting.column].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        inputs = self.tokenizer.encode_plus(
                        text,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=self.max_len
                    )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[idx]
        }




""" Model """
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(self.cfg.model.model_name)
        self.config = AutoConfig.from_pretrained(self.cfg.model.model_name)
        # 新しい単語の追加，https://qiita.com/m__k/items/e620bc14fba7a4e36824
        #self.backbone.resize_token_embeddings(len(cfg.tokenizer))
        if self.cfg.model.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        

        self.Linear = nn.Linear(self.config.hidden_size, self.cfg.setting.num_classes)

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
        MSE_criterion = nn.MSELoss()
        loss = torch.sqrt(MSE_criterion(pred, true.float()))
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
            loss = self.criterion(outputs, label)
        else:
            loss = None
        return loss 
        


""" AWP """
class AWP:
    def __init__(self, cfg, model, optimizer, scaler = None):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.adv_param = cfg.awp.adv_param
        self.adv_lr = cfg.awp.adv_lr
        self.adv_eps = cfg.awp.adv_eps
        #self.start_step = start_step
        self.adv_step = cfg.awp.adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, batch):
        #if (self.adv_lr == 0) or (epoch < self.start_step):
        #    return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step() 
            with autocast(enabled = self.cfg.model.apex):
                input_ids = batch['input_ids'].to(self.cfg.setting.device)
                attention_mask = batch['attention_mask'].to(self.cfg.setting.device)
                #token_type_ids = batch['token_type_ids'].to(cfg.setting.device)
                labels = batch['labels'].to(self.cfg.setting.device)
                adv_loss = self.model(input_ids, attention_mask, labels)
                adv_loss = adv_loss.mean()
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                    
    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}





""" training function """
def train_fn(cfg, model, train_dataloader, optimizer, epoch, scheduler, 
            valid_dataloader, fold, tra_len, loss, total_samples, global_step,
            best_score = np.inf):
    
    # validationをはじめる時の値の設定
    valid_start = cfg.model.start_valid * math.floor(tra_len/cfg.setting.train_batch_size)

    # apexの設定
    if cfg.model.apex:
        scaler = GradScaler(enabled = cfg.model.apex)

    # AWPの設定
    if cfg.model.use_awp:
        start_awp = cfg.model.start_awp * math.floor(tra_len/cfg.setting.train_batch_size)
        awp = AWP(cfg, model, optimizer, scaler = scaler)

    # tqdmの設定
    if cfg.setting.use_tqdm:
        tbar = tqdm(train_dataloader)
    else:
        tbar = train_dataloader
        
    #val_schedule = [int(i) for i in list(np.linspace(1, len(tbar), num = int(1 / cfg.val_check_interval) + 1, endpoint = True))[1:]]

    for i, item in enumerate(tbar):
        model.train()
        global_step += 1

        input_ids = item['input_ids'].to(cfg.setting.device)
        attention_mask = item['attention_mask'].to(cfg.setting.device)
        # token_type_idsは文のペアを入力する時に使う(https://qiita.com/Dash400air/items/a616ef8d088e003dfd4c)
        #token_type_ids = item['token_type_ids'].to(cfg.device)
        target = item['target'].to(cfg.setting.device)
        
        batch_size = input_ids.shape[0]

        optimizer.zero_grad()
        if cfg.model.apex:
            with autocast(enabled = cfg.model.apex):
                batch_loss = model(input_ids, attention_mask, target)

            # Backward
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update() 
        
        else:
            batch_loss = model(input_ids, attention_mask, target)
            # Backward
            batch_loss.backward()
            optimizer.step()
 

        # awpの設定
        if cfg.model.use_awp and global_step >= start_awp:
            print('AWP start')
            awp.attack_backward(item)
        

        # スケジューラーの設定
        if scheduler is not None:
            scheduler.step()

        
        # lossの更新
        loss += batch_loss.item() * batch_size
        total_samples += batch_size
        train_loss = loss / total_samples
        wandb.log({"train_loss": train_loss})

        # 学習率の保存
        L_rate = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": L_rate})

        # epochの保存
        wandb.log({"epoch": epoch})

        if cfg.setting.use_tqdm:
            tbar.set_description('Batch loss: {:.4f} - Avg loss: {:.4f}'.format(batch_loss, train_loss))


        if (global_step + 1) % cfg.model.valid_frequency == 0 and global_step >= valid_start:
            valid_score = valid_fn(cfg, model, valid_dataloader)
            print(f"Validation Loss : {valid_score}")
            wandb.log({"valid_score": valid_score})

            if valid_score <= best_score:
                print(f"Validation Loss Improved ({best_score} ---> {valid_score})")
                best_score = valid_score
                model_path = cfg.setting.model_save_path + f'FB3-{cfg.setting.column}-fold{fold}-{cfg.setting.text}.pth'.replace('/', '-')    # モデルの名前に/が入ることがあるため置き換えてる
                torch.save(model.state_dict(), model_path)
                print("Model Saved")

        torch.cuda.empty_cache()
        gc.collect()
    return 





""" validation function """
def valid_fn(cfg, model, dataloader):
    model.eval()

    valid_size = 0
    validation_loss = 0.0
    
    for i, item in enumerate(dataloader):
        input_ids = item['input_ids'].to(cfg.setting.device)
        attention_mask = item['attention_mask'].to(cfg.setting.device)
        #token_type_ids = item['token_type_ids'].to(cfg.device)
        target = item['target'].to(cfg.setting.device)

        
        batch_size = input_ids.size(0)
        valid_loss = model(input_ids, attention_mask, target)

        validation_loss += (valid_loss.item() * batch_size)
        valid_size += batch_size
        score = validation_loss / valid_size

    torch.cuda.empty_cache()
    gc.collect()
    return score



"""* Optimizer and scheduler"""

def get_optimizer(cfg, model):
    if cfg.optimizer.all_optimize:
        optimizer_parameters = model.parameters()
    else:
        #param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': cfg.encoder_lr, 'weight_decay': cfg.weight_decay},
            {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': cfg.encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
                'lr': cfg.decoder_lr, 'weight_decay': 0.0}
        ]
    if cfg.optimizer.bnb_8bit:
        optimizer = bnb.optim.Adam8bit(optimizer_parameters, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        optimizer = AdamW(optimizer_parameters, lr = cfg.optimizer.lr, eps = cfg.optimizer.eps, betas = cfg.optimizer.betas)
    
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.scheduler.scheduler_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.scheduler.T_max, eta_min=cfg.scheduler.min_lr)

    elif cfg.scheduler.scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.scheduler.T_0, eta_min=cfg.scheduler.min_lr)

    elif cfg.scheduler.scheduler_name == 'Linear':
        # start_factor:最初の学習率にかける値
        # end_factor:最後の到達する学習率にするためにかける値
        # 何エポックで到達するかの数値
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=cfg.scheduler.start_factor, end_factor=cfg.scheduler.end_factor, total_iters=cfg.scheduler.total_iters)

    elif cfg.scheduler.scheduler_name == 'Exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=cfg.scheduler.gamma)

    elif cfg.scheduler.scheduler_name == 'Linear_warmup':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.scheduler.num_warmup_steps, num_training_steps = cfg.scheduler.num_train_steps)

    elif cfg.scheduler.scheduler_name == 'cosine_warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg.scheduler.num_warmup_steps, num_training_steps = cfg.scheduler.um_train_steps)

    else: 
        return None      
    return scheduler

 

""" dataloader function """
def prepare_loaders(cfg, fold, df):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 学習データ数を取得
    train_len = len(df_train)
    
    train_dataset = FB3_Dataset(cfg, df_train)
    valid_dataset = FB3_Dataset(cfg, df_valid)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name,  use_fast=True)
    # ダイナミックパディングの設定
    collate_fn = DataCollatorWithPadding(tokenizer = tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=cfg.setting.train_batch_size, collate_fn=collate_fn, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.setting.valid_batch_size, collate_fn=collate_fn,
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader, train_len




""" training_loop """ 
def training_loop(cfg, fold):
    #logging.info(f' Fold {fold} '.center(50, '*'))
    set_random_seed(cfg.setting.seed)


    # データの読み込み
    train_df = pd.read_csv(cfg.setting.data_path + f"{cfg.setting.column}.csv")
    
    #logging.info('Preparing training and validating dataloader...')
    train_dataloader, valid_dataloader, tra_len= prepare_loaders(cfg, fold, train_df)

    #logging.info('Preparing model, optimizer, and scheduler...')
    model = Model(cfg).to(cfg.setting.device)
    optimizer = get_optimizer(cfg, model)
    # cfgに与えたい
    #num_training_steps = int(len(train_dataloader) * cfg.setting.num_epochs)
    scheduler = get_scheduler(cfg, optimizer)

    # 累積する損失の初期値
    loss = 0
    total_samples = 0
    global_step = 0

    for epoch in range(1, cfg.setting.num_epochs + 1): 
        train_fn(cfg, model, train_dataloader, optimizer, epoch, scheduler, 
                valid_dataloader, fold, tra_len, loss, total_samples, global_step,
                best_score = np.inf)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return 



""" main """
@hydra.main(config_path=".",config_name="config.yaml",version_base=None)
def main(cfg: DictConfig) -> None:
    fold = cfg.setting.fold
    run = wandb.init(project = cfg.wandb.project, 
                     config = cfg,
                     job_type='Train',
                     tags= cfg.wandb.tags,
                     name = f'{cfg.model.model_name}-{cfg.setting.column}-fold{fold}-{cfg.setting.text}',
                     anonymous='allow')
    training_loop(cfg, fold)
    run.finish() 



if __name__ == '__main__':
    main()