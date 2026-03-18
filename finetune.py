import os
import shutil
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import logging
from argparse import ArgumentParser
import yaml
from pydantic import BaseModel
import random
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from functools import partial
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM, 
    AutoModel, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    ProgressCallback
)
from peft import PeftModel
import lmdb
import pickle
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import functools as fn
import warnings

warnings.filterwarnings(
    "ignore", 
    message=".*use_reentrant parameter should be passed explicitly.*", 
    category=UserWarning
)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/finetune.yaml', help='Path to configuration file')
parser.add_argument('--ckpt', type=str, default=None, help='Path to model checkpoint')

args = parser.parse_args()

with open(args.config, 'r') as f:
	cfg_dict = yaml.safe_load(f)


class Config(BaseModel):
    CSV_PATH: str
    MODEL_NAME: str
    BATCH_SIZE: int
    VAL_BATCH_SIZE: int
    LR: float
    EPOCHS: int
    PROT_DIM: int
    LIG_DIM: int 
    HIDDEN_DIM: int
    NUM_HEADS: int
    MAX_DOMAINS: int
    ESM_PATH: str
    NUM_WORKERS: int
    seed: int
    emb_path: str
    esm_name: str
    molformer_name: str
    pretrained_model_path: str
    lora_r: int
    lora_alpha: int = 16
    lora_dropout: float = 0.01
    patience: int = 10
    ACCUM_STEPS: int = 1
    checkpointing: bool = True
    use_amp: str
    peft: bool = True
    wandb_project: str
    use_cached_data: bool = True
    load_cached_data_in_memory: bool = True
    train_cached_mdb_path: str = 'data/pdbbind/cache/train/'
    val_cached_mdb_path: str = 'data/pdbbind/cache/val/'
    test_cached_mdb_path: str = 'data/pdbbind/cache/test/'
    evi_reg: bool = False

cfg = Config(**cfg_dict)

if args.ckpt is not None:
    cfg.pretrained_model_path = args.ckpt

os.environ['WANDB_PROJECT'] = cfg.wandb_project
os.environ['WANDB_NAME'] = cfg.MODEL_NAME
 
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(cfg.seed)

def get_row(row, emb_root_path, prot_tokenizer, mol_tokenizer):
    try:
        emb_path = os.path.join(emb_root_path, f'{row["Assembly_ID"]}_prot_emb.pt')
        domain_mask_path = emb_path.replace('embeddings', 'domain_masks').replace('prot_emb', 'domain_mask')

        embs = torch.load(emb_path, weights_only=True)
        domain_mask = torch.load(domain_mask_path, weights_only=True)
        if len(embs['sequence']) == 0:
            raise ValueError("Empty sequence")
        prot_enc = prot_tokenizer(embs['sequence'], truncation=True, max_length=1024)
        mol_enc = mol_tokenizer(row['smiles'], truncation=True, max_length=512)

        sample = {
            'smiles': row['smiles'],
            'ca_coords': embs['ca_coord'],
            'domain_mask': domain_mask,
            'n_domains': row['total_domains'],
            'sequence': embs['sequence'],
            'length': len(embs['sequence']),
            'input_ids': torch.ones(len(embs['sequence']), dtype=torch.long),
            'prot_input_ids': prot_enc['input_ids'],
            'prot_attention_mask': prot_enc['attention_mask'],
            'mol_input_ids': mol_enc['input_ids'],
            'mol_attention_mask': mol_enc['attention_mask'],
            'labels': torch.tensor(row['affinity'], dtype=torch.float32),
            'type': row['type']
        }
        return sample
    except Exception as e:
        print(f"Error loading {emb_path} or {domain_mask_path}: {e}")
        dummy_seq = "A"*10
        dummy_smiles = "C"
        prot_enc = prot_tokenizer(dummy_seq, truncation=True, max_length=1024)
        mol_enc = mol_tokenizer(dummy_smiles, truncation=True, max_length=512)
        return {
            'smiles': dummy_smiles,
            'ca_coords': torch.zeros((10, 3)),
            'domain_mask': torch.zeros(10),
            'n_domains': 1,
            'sequence': dummy_seq,
            'length': 10,
            'prot_input_ids': prot_enc['input_ids'],
            'prot_attention_mask': prot_enc['attention_mask'],
            'mol_input_ids': mol_enc['input_ids'],
            'mol_attention_mask': mol_enc['attention_mask'],
            'input_ids': torch.ones(10, dtype=torch.long),
            'labels': torch.tensor(0.0, dtype=torch.float32),
            'type': 'X'
        }

def _worker_chunk(df, indices, emb_root_path, esm_name, molformer_name):
    prot_tokenizer = AutoModelForMaskedLM.from_pretrained(cfg.esm_name, trust_remote_code=True).tokenizer
    mol_tokenizer = AutoTokenizer.from_pretrained(cfg.molformer_name, trust_remote_code=True)
    results = []
    for idx in indices:
        row = df.iloc[idx]
        data = get_row(row, emb_root_path, prot_tokenizer, mol_tokenizer)
        data_bytes = pickle.dumps(data)
        results.append((idx, data_bytes))
    return results

def _worker_proxy(args):
    return _worker_chunk(*args)

class BioLiPDataset(Dataset):
    def __init__(self, data_df, cfg, cached_mdb_path, num_csv_processors=32):
        self.data = data_df.copy().reset_index(drop=True)
        self.cfg = cfg
        self.cached_mdb_path = cached_mdb_path
        self.num_processors = num_csv_processors
        
        self.use_cache = cfg.use_cached_data
        self.load_to_memory = cfg.load_cached_data_in_memory

        self.data_memory = None
        self.env = None
        self.prot_tokenizer = None
        self.mol_tokenizer = None
        if self.use_cache:
            self._init_cache()

    def _init_tokenizers(self):
        if self.prot_tokenizer is None:
            self.prot_tokenizer = AutoTokenizer.from_pretrained(self.cfg.esm_name, trust_remote_code=True)
            self.mol_tokenizer = AutoTokenizer.from_pretrained(self.cfg.molformer_name, trust_remote_code=True)

    def _init_cache(self):
        self.env = lmdb.open(
            self.cached_mdb_path,
            map_size=(1024**3) * 60,
            readonly=False,
            lock=False,
        )
        with self.env.begin() as txn:
            is_cached = txn.stat()["entries"] >= len(self.data)
        if not is_cached:
            self._build_cache()
        if self.load_to_memory:
            self._load_cache_into_memory()

    def _build_cache(self):
        print(f"Building cache at {self.cached_mdb_path}...")
        chunk_size = math.ceil(len(self.data) / self.num_processors)
        indices = list(range(len(self.data)))
        chunks = [indices[i: i + chunk_size] for i in range(0, len(indices), chunk_size)]

        with self.env.begin(write=True) as txn:
            with Pool(self.num_processors) as pool:
                tasks = [(self.data, chunk, self.cfg.emb_path, self.cfg.esm_name, self.cfg.molformer_name) for chunk in chunks]
                iterator = pool.imap(_worker_proxy, tasks)
            
                for chunk_results in tqdm(iterator, total=len(tasks)):
                    for idx, data_bytes in chunk_results:
                        txn.put(str(idx).encode(), data_bytes)

    def _load_cache_into_memory(self):
        print("Loading cached data into memory...")
        self.data_memory = [None] * len(self.data)
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    idx = int(key.decode())
                    if idx < len(self.data_memory):
                        self.data_memory[idx] = pickle.loads(value)
                except Exception as e:
                    print(f"Error loading index {key}: {e}")
                    continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx is None:
            return None
        data = None

        if self.load_to_memory and self.data_memory:
            data = self.data_memory[idx]
        elif self.use_cache and self.env:
            with self.env.begin() as txn:
                data_bytes = txn.get(str(idx).encode())
                if data_bytes:
                    data = pickle.loads(data_bytes)
        if data is None:
            self._init_tokenizers()
            data = get_row(self.data.iloc[idx], self.cfg.emb_path, self.prot_tokenizer, self.mol_tokenizer)
            
        return data

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    prot_input_ids = [torch.tensor(b['prot_input_ids'], dtype=torch.long) for b in batch]
    prot_attention_mask = [torch.tensor(b['prot_attention_mask'], dtype=torch.long) for b in batch]
    
    mol_input_ids = [torch.tensor(b['mol_input_ids'], dtype=torch.long) for b in batch]
    mol_attention_mask = [torch.tensor(b['mol_attention_mask'], dtype=torch.long) for b in batch]
    seqs = {
        'input_ids': pad_sequence(prot_input_ids, batch_first=True, padding_value=1),
        'attention_mask': pad_sequence(prot_attention_mask, batch_first=True, padding_value=0)
    }
    
    smiles = {
        'input_ids': pad_sequence(mol_input_ids, batch_first=True, padding_value=2),
        'attention_mask': pad_sequence(mol_attention_mask, batch_first=True, padding_value=0)
    }

    domain_masks = pad_sequence([b['domain_mask'] for b in batch], batch_first=True)
    padding_mask = (domain_masks != 0)
    
    ca_coords = pad_sequence([b['ca_coords'] for b in batch], batch_first=True)
        
    return {
        'seqs': seqs,
        'domain_mask': domain_masks,
        'mask': padding_mask,
        'coords_native': ca_coords,
        'smiles': smiles,
        'types': [b['type'] for b in batch],
        'labels': torch.stack([b['labels'] for b in batch]),
        'return_loss': True
    }
   

class DomainGatedGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, distance_thresh=8.0, cfg=cfg):
        super().__init__()
        self.thresh = distance_thresh
        self.hidden_dim = hidden_dim
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.interface_bias = nn.Parameter(torch.tensor(1.0))

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x, coords, domain_mask, mask):
        h = self.input_proj(x)

        B, L, _ = x.shape

        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

        attn_scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # (B, H, L, L)

        dists = torch.cdist(coords, coords)
        spatial_adj = (dists < self.thresh)
        
        dom_i = domain_mask.unsqueeze(-1)  # (B, L, 1)
        dom_j = domain_mask.unsqueeze(1)  # (B, 1, L)
        diff_domain = (dom_i != dom_j)

        is_structure_i = (dom_i >= 2)
        is_structure_j = (dom_j >= 2)
        valid_structure_pair = is_structure_i & is_structure_j

        interface_adj = (diff_domain & valid_structure_pair).float()

        attn_scores = attn_scores + (interface_adj.unsqueeze(1) * self.interface_bias)

        adj = spatial_adj * mask.unsqueeze(1) * mask.unsqueeze(2)
        attn_scores = attn_scores.masked_fill(~adj.unsqueeze(1), -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, H, L, L)
        attn_probs = F.dropout(attn_probs, p=0.1, training=self.training)
        attn_probs = attn_probs * mask.view(B, 1, L, 1).float()
        out = (attn_probs @ v).transpose(1, 2).contiguous().view(B, L, -1)  # (B, L, H*D)
        out = self.out_proj(out)

        out = self.norm(h + self.act(out))
        return out * mask.unsqueeze(-1).float()

def evidential_loss(mu, v, alpha, beta, targets, lam=1.0, eps=1e-6):
    twoBlambda = 2 * beta * (1 + v)
    nll = 0.5 * torch.log(math.pi / v + eps) \
        - alpha * torch.log(twoBlambda + eps) \
        + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda + eps) \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    error = torch.abs(targets - mu)
    reg = error * (2 * v + alpha)
    return torch.mean(nll + lam * reg)


class HCLBIND_Finetune(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.prot_seq_enc = AutoModelForMaskedLM.from_pretrained(cfg.esm_name, trust_remote_code=True)
        self.hclbind_prot_struct_enc = DomainGatedGAT(cfg.PROT_DIM, cfg.HIDDEN_DIM, cfg=cfg)

        self.lig_smiles_enc = AutoModel.from_pretrained(cfg.molformer_name, deterministic_eval=True, trust_remote_code=True)
        self.hclbind_lig_proj = nn.Sequential(
            nn.Linear(cfg.LIG_DIM, cfg.HIDDEN_DIM),
            nn.LayerNorm(cfg.HIDDEN_DIM),
            nn.SiLU()
        )
        
        self.hclbind_head_fusion = nn.MultiheadAttention(cfg.HIDDEN_DIM, cfg.NUM_HEADS, batch_first=True)
        
        outdim = 4 if cfg.evi_reg else 1
        self.hclbind_regressor = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.HIDDEN_DIM//2, outdim)
        )

    def fuse_and_pool(self, h_prot, h_lig, lig_mask, prot_mask):
        fused, _ = self.hclbind_head_fusion(h_lig, h_prot, h_prot, key_padding_mask=~prot_mask)
        mask_expanded = lig_mask.unsqueeze(-1).float()
        z_complex = (fused * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return z_complex

    def forward(self, seqs, coords_native, domain_mask, mask, smiles, labels=None, return_loss=None, types=None):
        with torch.no_grad():
            h_lig = self.lig_smiles_enc(**smiles).last_hidden_state.clone()
            h_esm = self.prot_seq_enc(**seqs).last_hidden_state[:, 1:-1, :].clone()
        lig_mask = smiles['attention_mask']
        h_lig = self.hclbind_lig_proj(h_lig)

        h_prot_native = self.hclbind_prot_struct_enc(h_esm, coords_native, domain_mask, mask)
        z_native = self.fuse_and_pool(h_prot_native, h_lig, lig_mask, mask)
        if self.cfg.evi_reg:
            outputs = self.hclbind_regressor(z_native)

            gamma = outputs[:, 0]
            v = F.softplus(outputs[:, 1]) + 1e-6
            alpha = F.softplus(outputs[:, 2]) + 1.0 + 1e-6
            beta = F.softplus(outputs[:, 3]) + 1e-6

            preds = gamma

            loss = None
            if labels is not None:
                if labels.dim() > 1: labels = labels.squeeze()
                loss = evidential_loss(gamma, v, alpha, beta, labels)
            uncertainty = beta / (v * (alpha - 1))

            return {
                'loss': loss,
                'preds': preds,
                'labels': labels,
                'uncertainty': uncertainty
            }
        else:
            preds = self.hclbind_regressor(z_native).squeeze(-1)
            loss = None
            if labels is not None:
                loss = nn.MSELoss()(preds, labels)

            return {'loss': loss, 'preds': preds, 'labels': labels}
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.prot_seq_enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.lig_smiles_enc.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

def split_dataset(df):
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    logging.info(f"Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")
    return train_df, val_df, test_df


class EpochProgressBarCallback(TrainerCallback):
    def __init__(self):
        self.training_bar = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            steps_per_epoch = int(state.max_steps / args.num_train_epochs)
            current_epoch = int(state.epoch) + 1

            if self.training_bar is None:
                self.training_bar = tqdm(total=steps_per_epoch, desc=f"Epoch {current_epoch}/{args.num_train_epochs}")
            else:
                self.training_bar.reset(total=steps_per_epoch)
                self.training_bar.set_description(f"Epoch {current_epoch}/{args.num_train_epochs}")
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            self.training_bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None and logs:
            self.training_bar.set_postfix(logs)

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_bar is not None:
            self.training_bar.close()

def get_cindex(Y, P):
	P = P[:,np.newaxis] - P
	P = np.float32(P==0) * 0.5 + np.float32(P>0)

	Y = Y[:,np.newaxis] - Y
	Y = np.tril(np.float32(Y>0), 0)

	P_sum = np.sum(P*Y)
	Y_sum = np.sum(Y)

	if Y_sum==0:
		return 0
	else:
		return P_sum/Y_sum


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        original_state_dict = self.model.state_dict
        def contiguous_state_dict():
            return {k: v.contiguous() for k, v in original_state_dict().items()}
        self.model.state_dict = contiguous_state_dict
        tensors = dict()
        for k, v in self.model.named_parameters():
            if 'hclbind' in k:
                tensors[k] = v.detach().cpu()
        torch.save(tensors, os.path.join(output_dir, 'aux_weights.pt'))
        return super()._save(output_dir)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.flatten()
    labels = labels.flatten()
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    pcc = pearsonr(labels, preds)[0]
    scc = spearmanr(labels, preds)[0]
    cindex = get_cindex(np.array(labels), np.array(preds))

    return {
        'mse': mse,
        'rmse': rmse,
        'pcc': pcc,
        'scc': scc,
        'cindex': cindex
    }

def main():
    out_dir = os.path.join('outputs', cfg.MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)

    current_file = os.path.abspath(__file__)
    shutil.copy2(current_file, os.path.join(out_dir, os.path.basename(current_file)))
    config_src = os.path.abspath(args.config)
    shutil.copy2(config_src, os.path.join(out_dir, os.path.basename(config_src)))
    if args.ckpt is not None:
        os.system(f"cp -r {args.ckpt} {out_dir}/pre_{os.path.basename(args.ckpt)}")
    else:
        os.system(f"cp -r {cfg.pretrained_model_path} {out_dir}/pre_{os.path.basename(cfg.pretrained_model_path)}")
    
    full_df = pd.read_csv(cfg.CSV_PATH)
    train_df, val_df,test_df = split_dataset(full_df)
    train_dataset = BioLiPDataset(train_df, cfg, cfg.train_cached_mdb_path)
    val_dataset = BioLiPDataset(val_df, cfg, cfg.val_cached_mdb_path)
    test_dataset = BioLiPDataset(test_df, cfg, cfg.test_cached_mdb_path)

    model = HCLBIND_Finetune(cfg)
    
    if cfg.pretrained_model_path:
        print(f'Loading pretrained weights from {cfg.pretrained_model_path}...')

        sd = torch.load(os.path.join(cfg.pretrained_model_path, 'aux_weights.pt'), map_location='cpu', weights_only=True)
        if cfg.peft:
            model = PeftModel.from_pretrained(model, cfg.pretrained_model_path)
        else:
            sd = {k.replace('base_model.model.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        [print('Missing ', k) for k in missing if 'hclbind' in k and 'regressor' not in k]
        [print('Unexpected ', k) for k in unexpected if 'hclbind' in k and 'idd_head' not in k and 'lpm_proj' not in k]
    else:
        print("Pretrained weights DOES NOT exist. Train from scratch.")
    for n, p in model.named_parameters():
        if 'hclbind' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}, Trainable parameters: {trainable_params}')
    training_args = TrainingArguments(
        output_dir=os.path.join('outputs', cfg.MODEL_NAME),
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.VAL_BATCH_SIZE,
        gradient_accumulation_steps=cfg.ACCUM_STEPS,
        learning_rate=cfg.LR,
        logging_dir=os.path.join('outputs', cfg.MODEL_NAME, 'logs'),
        logging_steps=20,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_rmse',
        greater_is_better=False,
        seed=cfg.seed,
        fp16=(cfg.use_amp == 'fp16'),
        bf16=(cfg.use_amp == 'bf16'),
        dataloader_num_workers=cfg.NUM_WORKERS,
        dataloader_persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2 if cfg.NUM_WORKERS > 0 else None,
        remove_unused_columns=False,
        group_by_length=False,
        ddp_find_unused_parameters=True,
        gradient_checkpointing=cfg.checkpointing,
        report_to='wandb'
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)]
    )
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(EpochProgressBarCallback())

    print("Starting Training...")
    trainer.train()

    best_ckpt = trainer.state.best_model_checkpoint
    if training_args.load_best_model_at_end and best_ckpt is not None:
        print(f"Loading best model from {best_ckpt}")
        trainer._load_best_model()
        final_path = os.path.join('outputs', cfg.MODEL_NAME, 'best_model')
    else:
        print("No best checkpoint found; saving last model.")
        final_path = os.path.join('outputs', cfg.MODEL_NAME, 'last_model')

    trainer.save_model(final_path)
    print(f'Training Complete. Model saved to {final_path}')


if __name__ == "__main__":
    main()