import os
import sys
import logging
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import models
from data import TextDataSet
from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device(args.device) #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    if args.continue_train > 0:
        pass

    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True).split(train_data['text'], train_data['sentiment']) #Using StratifiedKFold for cross-validation    
    for fold, (train_index, valid_index) in enumerate(skf): #by skf every fold will have similar label distribution
        seed_everything(args.seed) #fix seed
        if args.continue_train > fold+1:
            logger.info(f'skipping {fold+1}-fold')
            continue
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')
        #logger to log current n-fold output

        kfold_train_data = train_data.iloc[train_index]
        kfold_valid_data = train_data.iloc[valid_index]

        model = getattr(models, args.model)(args.input_size, args.hidden_size, args.num_classes).to(device) #make model based on the model name and args

        train_dataset = TextDataSet(text=kfold_train_data['text'].values, label=kfold_train_data['sentiment'].values, tokenizer=model.tokenizer, max_length=args.max_length)
        valid_dataset = TextDataSet(text=kfold_valid_data['text'].values, label=kfold_valid_data['sentiment'].values, tokenizer=model.tokenizer, max_length=args.max_length)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = get_sch(args.scheduler, optimizer, warmup_epochs=args.warmup_epochs)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
        )
        
        trainer = Trainer(
            train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, fold_result_path, fold_logger)
        trainer.train() #start training

        test_dataset = TextDataSet(text=test_data['text'].values, label=test_data['sentiment'].values, tokenizer=model.tokenizer, max_length=args.max_length)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ) #make test data loader
        pred, target = trainer.test(test_loader)

        fold_logger.info(f'Acc: {accuracy_score(target, pred)}')
        fold_logger.info(f'F1: {f1_score(target, pred, average="macro")}')