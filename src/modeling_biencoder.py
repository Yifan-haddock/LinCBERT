from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses, distances
from typing import List, Optional, Tuple, Union
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

class BiEncoder_Normer(PreTrainedModel):
    
    supports_gradient_checkpointing = True
    
    def __init__(self, concept_encoder, context_encoder,alpha = 2.0, beta = 40.0, base = 0.5, projection = 'linear', 
                use_miner = False, miner_margin=0.2, mean_pooling = False, config = None, **kwargs):
        
        super(BiEncoder_Normer, self).__init__(config)
        self.use_miner = use_miner
        self.mean_pooling = mean_pooling
        self.config = config
        
        self.context_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path = context_encoder)
        self.concept_encoder = AutoModel.from_pretrained(pretrained_model_name_or_path = concept_encoder)
        
        if projection == 'linear':
            self.context_pooler = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.context_encoder.config.hidden_size, self.concept_encoder.config.hidden_size)
            )
        elif projection == 'nonlinear':
            self.context_pooler = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.context_encoder.config.hidden_size, self.context_encoder.config.hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(self.context_encoder.config.hidden_size, self.concept_encoder.config.hidden_size),    
            )
        else:
            print(f"{projection}: not a valid projection.")

        if use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets="all")
        else:self.miner = None
        
        self.loss = losses.MultiSimilarityLoss(
                alpha= alpha, beta= beta, base=base)  # 1,2,3; 40,50,60
        # self.acti = nn.Tanh()
        # self.loss = nn.CrossEntropyLoss()
        
    def forward(
        self,
        concept_input_ids: torch.Tensor = None,
        concept_attention_mask : torch.Tensor = None,
        context_input_ids: torch.Tensor = None, 
        context_attention_mask : torch.Tensor = None,
        context_mean_inputs_mask : torch.Tensor = None,
        labels : Optional[torch.Tensor] = None,
        **kwargs  
    ):
        concept_encoder_outputs = self.concept_encoder(concept_input_ids, attention_mask = concept_attention_mask).last_hidden_state[:,0,:]
        if self.mean_pooling:
            context_mean_inputs_mask = context_mean_inputs_mask.unsqueeze(2)
            context_encoder_outputs = self.context_encoder(context_input_ids, attention_mask = context_attention_mask).last_hidden_state
            context_encoder_outputs = (context_encoder_outputs * context_mean_inputs_mask).sum(1) / context_mean_inputs_mask.sum(1)
        else:
            context_encoder_outputs = self.context_encoder(context_input_ids, attention_mask = context_attention_mask).last_hidden_state[:,0,:]
        context_pooler_outputs = self.context_pooler(context_encoder_outputs)
        # context_encoder_outputs = self.acti(self.context_encoder(context_input_ids, attention_mask = context_attention_mask).last_hidden_state[:,0,:])
        # context_pooler_outputs = self.acti(self.context_pooler(context_encoder_outputs))
        # logits = F.normalize(concept_pooler_outputs, p =2 ,dim = 1) @ F.normalize(context_pooler_outputs, p=2, dim = 1).T
        # targets = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        if self.use_miner:
            hard_pairs = self.miner(concept_encoder_outputs, labels, ref_emb = context_pooler_outputs, ref_labels = labels.clone())
            loss = self.loss(concept_encoder_outputs, labels, indices_tuple = hard_pairs, ref_emb = context_pooler_outputs, ref_labels = labels.clone())
        else:
            loss = self.loss(concept_encoder_outputs, labels, ref_emb = context_pooler_outputs, ref_labels = labels.clone())
        return (loss,)
    
    def dictionary_embedding(self, model_inputs, show_progress = True, device = 'cuda:0'):
        self.eval() # prevent dropout
        self.to(device)
        
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        batch_size=1024
        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, input_ids.shape[0], batch_size))
            else:
                iterations = range(0, input_ids.shape[0], batch_size)
                
            for start in iterations:
                end = min(start + batch_size, input_ids.shape[0])
                batch_ids = input_ids[start:end].to(device)
                batch_masks = attention_mask[start:end].to(device)
                ## use saved tokenize.pt
                batch_dense_embeds = self.concept_encoder(batch_ids, batch_masks).last_hidden_state[:,0,:]
                dense_embeds.append(batch_dense_embeds.detach().cpu().numpy().astype(np.float32))
                torch.cuda.empty_cache()
            
        dense_embeds = np.concatenate(dense_embeds, axis = 0)
        
        return dense_embeds
    
    def query_embedding(self, model_inputs, show_progress = True, device = 'cuda:0'):
        self.eval() # prevent dropout
        self.to(device)
        
        input_ids = model_inputs["context_input_ids"]
        attention_mask = model_inputs["context_attention_mask"]
        context_mean_inputs_mask = model_inputs["context_mean_inputs_mask"]

        batch_size=40
        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, input_ids.shape[0], batch_size))
            else:
                iterations = range(0, input_ids.shape[0], batch_size)
                
            for start in iterations:
                end = min(start + batch_size, input_ids.shape[0])
                batch_ids = input_ids[start:end].to(device)
                batch_masks = attention_mask[start:end].to(device)
                if self.mean_pooling:
                    batch_context_mean_inputs_mask = context_mean_inputs_mask[start:end].to(device)
                ## use saved tokenize.pt
                
                if self.mean_pooling:
                    batch_context_mean_inputs_mask = batch_context_mean_inputs_mask.unsqueeze(2)
                    batch_dense_embeds = self.context_encoder(batch_ids, attention_mask = batch_masks).last_hidden_state
                    batch_dense_embeds = (batch_dense_embeds * batch_context_mean_inputs_mask).sum(1) / batch_context_mean_inputs_mask.sum(1)
                else:
                    batch_dense_embeds = self.context_encoder(batch_ids, attention_mask = batch_masks).last_hidden_state[:,0,:]
                
                batch_dense_embeds = self.context_pooler(batch_dense_embeds)
                
                # batch_dense_embeds = batch_dense_embeds.detach().cpu().numpy()
                dense_embeds.append(batch_dense_embeds.detach().cpu().numpy().astype(np.float32))
                torch.cuda.empty_cache()
            
        dense_embeds = np.concatenate(dense_embeds, axis = 0)
        
        return dense_embeds