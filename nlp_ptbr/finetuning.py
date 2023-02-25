from nlp_ptbr.base import *
from nlp_ptbr.data import *
#from nlp_ptbr.BERTweetBRTokenizer import BertweetBRTokenizer

import glob, os, gc, time
import numpy as np
import pandas as pd
from torch import nn
from torchinfo import summary
from transformers import AdamW, AutoModel, get_scheduler, AutoModelForSequenceClassification, AutoTokenizer
from tqdm.auto import tqdm
from datasets import load_metric
from torch.utils.tensorboard import SummaryWriter

set_seed(GLOBAL_SEED)


class CustomFineTuningForSequenceClassification():
    def __init__(self, name='sentiment-analysis-ptbr',
                 model_name='mbert',
                 checkpoint='bert-base-multilingual-cased',
                 tokenizer_checkpoint=None,
                 labels=['NEGATIVO', 'POSITIVO'],
                 output_hidden_states = False,
                 local_files_only=False,
                 normalization=False,
                 use_slow_tokenizer=False):
        
        super().__init__()
        
        self.name = name
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.num_labels=len(labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint, num_labels=self.num_labels, output_hidden_states=output_hidden_states, local_files_only=local_files_only).to(device)
        
        self.normalization = normalization
        self.use_slow_tokenizer = use_slow_tokenizer
        
        tokenizer_checkpoint = self.checkpoint if self.tokenizer_checkpoint is None else self.tokenizer_checkpoint
       
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, local_files_only=local_files_only, normalization=self.normalization, use_fast=not self.use_slow_tokenizer)

        self.model.config.id2label = {i:e.upper() for i, e in enumerate(labels)}
    
    def model_summary(self):
        summary(self.model)
        
    def initialize(self, len_train_data, num_epochs = 3, num_warmup_steps=0):
        optimizer = AdamW(self.model.parameters(), lr=5e-5, eps = 1e-8)
        num_training_steps = num_epochs * len_train_data
        
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return optimizer, lr_scheduler, num_training_steps
    
    def fit(self, train_loader, test_loader=None,
            epochs=3, metric='loss', greater=False, tensorboard=True,
            logging_strategy="steps", logging_steps=50, 
            checkpoints_dir='.', suffix='all', save=False,
            train_steps_metrics_accumulated=False,
            empty_steps=50,
            log_func=print,
            disable_tqdm=False):
        
        def save_history(history, outputs, dataset='train'):
            for m in outputs.score.keys():
                history[dataset][m].append(outputs.score[m])
            
            history[dataset]['time'].append(outputs.time)
            history[dataset]['loss'].append(outputs.loss)
            
        def log_tensorboard(tensorboard_writer, step, train_outputs, val_outputs=None):
            if tensorboard_writer is not None:
                if val_outputs is not None:
                    tensorboard_writer.add_scalars('Loss', {'train':train_outputs.loss, 'test':val_outputs.loss}, step)
                    tensorboard_writer.add_scalars('Accuracy', {'train':train_outputs.score['acc'], 'test':val_outputs.score['acc']}, step)
                    tensorboard_writer.add_scalars('F1', {'train':train_outputs.score['f1'], 'test':val_outputs.score['f1']}, step)
                else:
                    tensorboard_writer.add_scalar('Loss/train', train_outputs.loss, step)
                    tensorboard_writer.add_scalar('Accuracy/train', train_outputs.score['acc'], step)
                    tensorboard_writer.add_scalar('F1/train', train_outputs.score['f1'], step)
        
        def save_best(history, metric, best_metric, step, dataset='test'):
            tmp_best = best_metric
            
            if greater:
                if history[dataset][metric][-1] > best_metric:
                    history['best_metric'] = history[dataset][metric][-1]
                    history['best_step'] = step
                    tmp_best = history[dataset][metric][-1]
            else:
                if history[dataset][metric][-1] < best_metric:
                    history['best_metric'] = history[dataset][metric][-1]
                    history['best_step'] = step
                    tmp_best = history[dataset][metric][-1]
            return tmp_best
        
        if tensorboard:
            # default `log_dir` is "runs" - we'll be more specific here
            tensorboard_dir = os.path.join(Experiment.UFF_SENTIMENT_OUTPUTS_LOGS, "tensorboard", self.model_name, suffix)
            writer = SummaryWriter(tensorboard_dir, flush_secs=5)
            log_func(f"Tensorboard habilitado. Para visualizar, execute 'tensorboard --logdir={tensorboard_dir}'")
            #writer.add_graph(self.model, next(iter(train_loader)))
        
        best_epoch = 0
        best_valid_metric = float('-inf') if greater else float('inf')
        metric_params = {'average':'weighted'}
        
        if test_loader is not None:
            history = {'strategy': logging_strategy, 'best_step': 0,
                       'metric': metric, 'best_metric': best_valid_metric,
                       'train': {m: [] for m in get_metrics_names(self.num_labels)},
                       'test': {m: [] for m in get_metrics_names(self.num_labels)}}
        else:
            history = {'strategy': logging_strategy, 'best_step': 0,
                       'metric': metric, 'best_metric': best_valid_metric,
                       'train': {m: [] for m in get_metrics_names(self.num_labels)}}
        
        os.makedirs(checkpoints_dir, exist_ok=True)
        file_name = checkpoints_dir
        #file_name = os.path.join(checkpoints_dir, self.name)
        optimizer, lr_scheduler, num_training_steps = self.initialize(len(train_loader), num_epochs=epochs)
        
        pbar_epochs = tqdm(range(epochs), total=epochs, desc='Epochs', leave=False, disable=disable_tqdm)
        
        self.model.train()
        
        running_step = 0
        val_outputs = None
        
        for epoch in pbar_epochs:
            pbar_epochs.set_description(f'Epoch ({epoch+1} de {epochs}).')
            log_func(f'Epoch ({epoch+1} de {epochs}).')
            
            pbar_epoch_iterations = tqdm(enumerate(train_loader), total=len(train_loader), desc='Iterations', leave=False, disable=disable_tqdm)
            
            start_time = time.monotonic()
            
            metric_steps = load_metric('nlp_ptbr/metrics.py')
            running_loss = 0.0
            
            for idx, batch in pbar_epoch_iterations:
                pbar_epoch_iterations.set_description(f'Iteration ({idx+1}/{len(train_loader)})')
                #log_func(f'Iteration ({idx+1}/{len(train_loader)})')
                
                batch = {k: v.to(device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                running_loss += loss.item()
                
                predictions_proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(outputs.logits, dim=-1)
                metric_steps.add_batch(predictions=predictions, references=batch["labels"])
                
                if (logging_strategy=="steps") and ((running_step+1) % logging_steps == 0):
                    end_time = time.monotonic()
                    #score_steps = metric_steps.compute(labels=np.unique(batch["labels"].cpu()), compute_loss=False)
                    score_steps = metric_steps.compute(labels=np.array(range(self.num_labels)), compute_loss=False)
                    
                    if train_steps_metrics_accumulated:
                        train_outputs = EpochOutput(score_steps, end_time - start_time, running_loss / running_step)
                    else:
                        train_outputs = EpochOutput(score_steps, end_time - start_time, running_loss / logging_steps)
                    
                    save_history(history, train_outputs, 'train')
                    
                    if test_loader is not None:
                        val_outputs = self.evaluate_epoch(test_loader, disable_tqdm=disable_tqdm)
                        save_history(history, val_outputs, 'test')
                        best_valid_metric = save_best(history, metric, best_valid_metric, dataset='test', step=running_step+1)
                    else:
                        best_valid_metric= save_best(history, metric, best_valid_metric, dataset='train', step=running_step+1)
                    
                    if tensorboard:
                        log_tensorboard(writer, running_step+1, train_outputs, val_outputs)
                    
                    if not train_steps_metrics_accumulated:
                        metric_steps = load_metric('nlp_ptbr/metrics.py')
                        running_loss = 0.0
                        
                    start_time = time.monotonic()
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                running_step += 1
                
                if ((running_step+1) % empty_steps == 0):
                    torch.cuda.empty_cache()
            
            if (logging_strategy=="epoch") and ((epoch+1) % logging_steps == 0):
                end_time = time.monotonic()
                #score_steps = metric_steps.compute(labels=np.unique(batch["labels"].cpu()), compute_loss=False)
                score_steps = metric_steps.compute(labels=np.array(range(self.num_labels)), compute_loss=False)
                train_outputs = EpochOutput(score_steps, end_time - start_time, running_loss / (logging_steps * len(train_loader)))
                save_history(history, train_outputs, 'train')
                
                if test_loader is not None:
                    val_outputs = self.evaluate_epoch(test_loader, disable_tqdm=disable_tqdm)
                    save_history(history, val_outputs, 'test')
                    best_valid_metric = save_best(history, metric, best_valid_metric, dataset='test', step=epoch+1)
                else:
                    best_valid_metric= save_best(history, metric, best_valid_metric, dataset='train', step=epoch+1)    
                
                if tensorboard:
                    log_tensorboard(writer, epoch+1, train_outputs, val_outputs)

                metric_steps = load_metric('nlp_ptbr/metrics.py')
                running_loss = 0.0
                start_time = time.monotonic()
            
            torch.cuda.empty_cache()
            
        if tensorboard:
            writer.close()
        
        if save:
            print(file_name)
            self.model.save_pretrained(file_name)
        
        return history
        
    def evaluate_epoch(self, eval_dataloader, empty_steps=50, disable_tqdm=False):
        start_time = time.monotonic()
        
        pbar_epoch_iterations = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Iterations", leave=False, disable=disable_tqdm)
        
        metric = load_metric('nlp_ptbr/metrics.py')
        
        self.model.eval()
        
        epoch_loss = 0
        
        for idx, batch in pbar_epoch_iterations:
            pbar_epoch_iterations.set_description(f'Evaluating Iteration ({idx+1}/{len(eval_dataloader)})')
            
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            predictions_proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            #metric.add_batch(predictions=predictions, references=batch["labels"])
            metric.add_batch(predictions=predictions, references=batch["labels"])
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            if ((idx+1) % empty_steps == 0):
                torch.cuda.empty_cache()
            
        #score = metric.compute(labels=np.unique(batch["labels"].cpu()), compute_loss=False)
        score = metric.compute(labels=np.array(range(self.num_labels)), compute_loss=False)
        
        end_time = time.monotonic()
        total_time = end_time - start_time
        
        torch.cuda.empty_cache()
        
        return EpochOutput(score, total_time, epoch_loss / len(eval_dataloader))
    

