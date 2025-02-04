import os
os.environ['WANDB_MODE'] = 'offline'
import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
from configuration_xlm_roberta import XLMRobertaConfig
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses, distances
from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from transformers import Trainer
import datasets
import argparse
import wandb
from modeling_biencoder import BiEncoder_Normer
from peft import (get_peft_model_state_dict, get_peft_model, LoraConfig, PeftModel)

from typing import Optional
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging
import safetensors.torch
from safetensors.torch import load_file

logger = logging.get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-cf","--config_file")
package = parser.parse_args().config_file

CONCEPT_ENCODER_PATH = getattr(__import__(package, fromlist=[None]),  "CONCEPT_ENCODER_PATH")
CONTEXT_ENCODER_PATH = getattr(__import__(package, fromlist=[None]),  "CONTEXT_ENCODER_PATH")
MODEL_SAVE_DIR = getattr(__import__(package, fromlist=[None]),  "MODEL_SAVE_DIR")
TRAIN_NAME= getattr(__import__(package, fromlist=[None]),  "TRAIN_NAME")
DATASET= getattr(__import__(package, fromlist=[None]),  "DATASET")

MICRO_BATCH_SIZE = getattr(__import__(package, fromlist=[None]),  "MICRO_BATCH_SIZE")
GRADIENT_ACCUMULATION_STEPS = getattr(__import__(package, fromlist=[None]),  "GRADIENT_ACCUMULATION_STEPS")

LEARNING_RATE = getattr(__import__(package, fromlist=[None]),  "LEARNING_RATE")
EPOCHS = getattr(__import__(package, fromlist=[None]),  "EPOCHS")

ALPHA = getattr(__import__(package, fromlist=[None]),  "ALPHA")
BETA = getattr(__import__(package, fromlist=[None]),  "BETA")
BASE = getattr(__import__(package, fromlist=[None]),  "BASE")
PROJ = getattr(__import__(package, fromlist=[None]),  "PROJ")

USEMINER = getattr(__import__(package, fromlist=[None]),  "USEMINER")
FREEZE_PARAMS = getattr(__import__(package, fromlist=[None]),  "FREEZE_PARAMS")
PEFT_TUNING = getattr(__import__(package, fromlist=[None]),  "PEFT_TUNING")
PEFT_TUNING_MODULE = getattr(__import__(package, fromlist=[None]),  "PEFT_TUNING_MODULE")
ADD_SOFT_TOKEN = getattr(__import__(package, fromlist=[None]),  "ADD_SOFT_TOKEN")
FINETUNING = getattr(__import__(package, fromlist=[None]),  "FINETUNING")
UNIENCODER = getattr(__import__(package, fromlist=[None]),  "UNIENCODER")

ID_PAD = 0
MAX_LENGTH = 512

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"
SAFE_WEIGHTS_NAME = "pytorch_model.bin"

class CustomTrainer(Trainer):
    def __init__(self, *args, save_changed=True, **kwargs):
        self.save_changed = save_changed
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
        if self.save_changed:
            state_dict = self.model.state_dict()
            filtered_state_dict = {}
            for k, v in self.model.named_parameters():
                if 'lora' not in k and v.requires_grad:
                    filtered_state_dict[k] = state_dict[k]
            torch.save(filtered_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

def find_target_modules(model, tuning_module = "attention"):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if tuning_module == 'all':
            if "Linear" in str(type(module)) and "pooler" not in name:
                unique_layers.add(name)
        elif tuning_module == 'attention':
            if "attention" in name and "Linear" in str(type(module)) and "pooler" not in name:
                unique_layers.add(name)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)

def preprocess_function_train(datapoint):
    if ADD_SOFT_TOKEN:
        special_id = context_tokenizer.additional_special_tokens_ids[0]
    
    model_inputs = {
        "concept_input_ids": None,
        "context_input_ids": None,
        "labels": None
    }
    CUI = datapoint['CUI']
    labels = cui2label[CUI]
    concepts = str(datapoint['PN'])
    mentions = str(datapoint['STR'])
    contexts = str(datapoint['CONTEXT'])
    start = datapoint['start']
    end = datapoint['end']
    
    if ADD_SOFT_TOKEN and start is not None:
        start = int(start)
        end = int(end)
        contexts = contexts[:start] + '[SOFT] ' + contexts[start:end] + ' [SOFT]' + contexts[end:]
    concept_input_ids = concept_tokenizer.encode(concepts)
    
    if UNIENCODER:
        context_input_ids = context_tokenizer.encode(contexts, return_tensors= 'pt')[0]
    else:
        context_input_ids = context_tokenizer.encode(mentions,contexts, return_tensors= 'pt')[0]
    
    if ADD_SOFT_TOKEN and start is not None:
        indices = torch.where(context_input_ids == special_id)[0]
        mention_start = indices[0] + 1
        mention_end = indices[1]
    else:
        mention_start = 1
        mention_end = len(context_input_ids) - 1

    model_inputs["concept_input_ids"] = concept_input_ids
    model_inputs["context_input_ids"] = context_input_ids.tolist()
    model_inputs["labels"] =labels
    model_inputs["mention_start"] = mention_start
    model_inputs["mention_end"] = mention_end
    
    return model_inputs


def data_collator(batch):
    len_max_batch_concept = [len(batch[i].get("concept_input_ids"))
                     for i in range(len(batch))]
    len_max_batch_concept = min(MAX_LENGTH, max(len_max_batch_concept))
    len_max_batch_context = [len(batch[i].get("context_input_ids"))
                     for i in range(len(batch))]
    len_max_batch_context = min(MAX_LENGTH, max(len_max_batch_context))
    batch_concept_input_ids = []
    batch_concept_attention_mask = []
    batch_context_input_ids = []
    batch_context_attention_mask = []
    batch_labels = []
    batch_context_mean_inputs_mask = []

    for ba in batch:
        concept_input_ids, context_input_ids,labels, mention_start, mention_end = ba.get("concept_input_ids"), ba.get("context_input_ids") , ba.get("labels"), ba.get("mention_start"), ba.get("mention_end")
        concept_len_padding = len_max_batch_concept - len(concept_input_ids) 
        concept_input_ids = concept_input_ids[:len_max_batch_concept] + [0] * (concept_len_padding)
        concept_attention_mask = torch.ones(len_max_batch_concept,dtype=torch.long)
        
        if concept_len_padding != 0:
            concept_attention_mask[-concept_len_padding:] = 0
        tensor_concept_input_ids = torch.tensor(concept_input_ids, dtype=torch.long)
        batch_concept_input_ids.append(tensor_concept_input_ids)
        batch_concept_attention_mask.append(concept_attention_mask)
        
        context_len_padding = len_max_batch_context - len(context_input_ids) 
        context_input_ids = context_input_ids[:len_max_batch_context] + [0] * (context_len_padding)
        context_attention_mask = torch.ones(len_max_batch_context,dtype=torch.long)
        if context_len_padding != 0:
            context_attention_mask[-context_len_padding:] = 0
        tensor_context_input_ids = torch.tensor(context_input_ids, dtype=torch.long)
        batch_context_input_ids.append(tensor_context_input_ids)
        batch_context_attention_mask.append(context_attention_mask)
        
        context_mean_inputs_mask = torch.zeros(len_max_batch_context,dtype=torch.long)
        context_mean_inputs_mask[mention_start: mention_end] = 1
        batch_context_mean_inputs_mask.append(context_mean_inputs_mask)
        
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_labels.append(tensor_labels)
        
    batch_concept_input_ids = torch.stack(batch_concept_input_ids)
    batch_concept_attention_mask = torch.stack(batch_concept_attention_mask)
    batch_context_input_ids = torch.stack(batch_context_input_ids)
    batch_context_attention_mask = torch.stack(batch_context_attention_mask)
    batch_context_mean_inputs_mask = torch.stack(batch_context_mean_inputs_mask)
    
    batch_labels = torch.stack(batch_labels)
    input_dict = {
                "concept_input_ids": batch_concept_input_ids,
                "concept_attention_mask": batch_concept_attention_mask,
                "context_input_ids": batch_context_input_ids,
                "context_attention_mask":batch_context_attention_mask,
                "context_mean_inputs_mask":batch_context_mean_inputs_mask,
                "labels" : batch_labels
                }
    return input_dict

if __name__ == '__main__' :
    
    context_encoder_path = CONTEXT_ENCODER_PATH
    concept_encoder_path = CONCEPT_ENCODER_PATH
    # tokenizer = AutoTokenizer.from_pretrained(concept_encoder_path)
    concept_tokenizer = AutoTokenizer.from_pretrained(concept_encoder_path)
    context_tokenizer = AutoTokenizer.from_pretrained(context_encoder_path)
    concept_config = XLMRobertaConfig()
    
    if UNIENCODER:
        mean_pooling = True
    else:
        mean_pooling = False
        
    biencoder = BiEncoder_Normer(concept_encoder= concept_encoder_path, context_encoder=context_encoder_path, 
                                 alpha = ALPHA, beta = BETA, base = BASE, projection = PROJ, use_miner= USEMINER, mean_pooling = mean_pooling, config = concept_config)
    
    
    if hasattr(biencoder, "enable_input_require_grads"):
        biencoder.context_encoder.enable_input_require_grads()
        biencoder.concept_encoder.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
             output.requires_grad_(True)
             
    
    if PEFT_TUNING:
        target_modules = find_target_modules(biencoder, tuning_module = PEFT_TUNING_MODULE)
        lora_config = LoraConfig(target_modules=target_modules,
                        lora_dropout=0.05,
                        lora_alpha=16,
                        task_type="CAUSAL_BERT",
                        bias="none",
                        r=8,
                        )
        biencoder = get_peft_model(biencoder, lora_config)
        for param in biencoder.context_pooler.parameters():
            param.requires_grad = True
        # print(biencoder.print_trainable_parameters())
    
    if FREEZE_PARAMS:
        for param in biencoder.concept_encoder.parameters():
            param.requires_grad = False
            
    if ADD_SOFT_TOKEN:
        new_special_tokens = {'additional_special_tokens': ['[SOFT]']}
        context_tokenizer.add_special_tokens(new_special_tokens)
        biencoder.context_encoder.resize_token_embeddings(len(context_tokenizer))

    if FINETUNING:
        SAFE_TENSOR_FILE = f'{MODEL_SAVE_DIR}/model.safetensors'
        state_dict = load_file(SAFE_TENSOR_FILE)
        biencoder.load_state_dict(state_dict)
        MODEL_SAVE_DIR = os.path.join(MODEL_SAVE_DIR, 'finetuned')
        
    dataset = datasets.load_from_disk(DATASET)
    cui2label = {cui:i for i, cui in enumerate(list(dict.fromkeys(list(dataset["CUI"]))))}
    dataset = dataset.map(preprocess_function_train)
            
    print('model cuda start')
    biencoder.cuda()
    if PEFT_TUNING:
        CustomTrainer = CustomTrainer
    else:
        CustomTrainer = Trainer
    
    trainer = CustomTrainer(
        data_collator=data_collator,
            train_dataset=dataset,
            model=biencoder,
            # tokenizer=tokenizer,
            args=transformers.TrainingArguments(
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                per_device_train_batch_size=MICRO_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                # max_steps=MAX_STEPS,
                num_train_epochs=EPOCHS,
                max_grad_norm=1,
                weight_decay= 0.02, 
                logging_steps=5,
                warmup_steps=50,  # 618
                # warmup_ratio=0.01,
                # warmup_steps=16,
                evaluation_strategy="no",
                lr_scheduler_type="constant", #'constant',  # "cosine",
                logging_first_step=False,
                # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
                # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
                save_strategy="epoch",
                save_total_limit=12,
                # save_steps=SAVE_STEPS,
                # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
                # ddp_find_unused_parameters=None,
                gradient_checkpointing=True,
                # group_by_length=True,  # group together samples of roughly the same length in training
                output_dir=MODEL_SAVE_DIR,
                remove_unused_columns=False,
                optim="adamw_torch",  # "adamw_hf",
                # report_to=[], 
                report_to=["wandb"],  # ["tensorboard"],  # [], ["wandb"]
                fp16=True,
        )
    )
    wandb.init(project= "teaBERT-withcontext",mode = 'offline')
    flag_checkpoint = False
    trainer.train(resume_from_checkpoint=flag_checkpoint)
    ## need to save both lora ab and prefix align layer.
    trainer.save_model()
    context_tokenizer.save_pretrained(MODEL_SAVE_DIR)