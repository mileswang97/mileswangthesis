# train.py

import argparse
import json
import numpy as np
import os
import torch

from datetime import datetime
from pathlib import Path
from sklearn import metrics

#from evaluate import run_model
#from loader import load_data
#from model import MRNet

def train(rundir, diagnosis, epochs, learning_rate, use_gpu):
    val_auc_array = list()
    train_auc_array = list()
    train_loader, valid_loader, test_loader = load_data(diagnosis, use_gpu)
    
    model = MRNet()
    
    if use_gpu:
        model = model.cuda()

    # modify the code in this
    # try with RAdam
    # grid search?
    # can try without weight decay
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=.01)

    # patience too low (after 5 epochs, if AUC hasnt improved, slash learning rate .3), which is why high learning rate seems to work better
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=.3, threshold=1e-4)

    best_val_loss = float('inf')

    start_time = datetime.now()

    for epoch in range(epochs):
        change = datetime.now() - start_time
        print('starting epoch {}. time passed: {}'.format(epoch+1, str(change)))
        
        train_loss, train_auc, _, _ = run_model(model, train_loader, train=True, optimizer=optimizer)
        print(f'train loss: {train_loss:0.4f}')
        print(f'train AUC: {train_auc:0.4f}')

        val_loss, val_auc, _, _ = run_model(model, valid_loader)
        print(f'valid loss: {val_loss:0.4f}')
        print(f'valid AUC: {val_auc:0.4f}')
        val_auc_array.append(val_auc)
        train_auc_array.append(train_auc)
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            file_name = f'val{val_loss:0.4f}_train{train_loss:0.4f}_epoch{epoch+1}'
            save_path = Path(rundir) / file_name
            # dont need to save stuff for now, model is too shitty
            #torch.save(model.state_dict(), save_path)
            
    return val_auc_array, train_auc_array

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rundir', type=str, required=True)
    parser.add_argument('--diagnosis', type=int, required=True)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--learning_rate', default=1e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--max_patience', default=5, type=int)
    parser.add_argument('--factor', default=0.3, type=float)
    return parser

#if __name__ == '__main__':
#    args = get_parser().parse_args()
    
#    np.random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    if args.gpu:
#        torch.cuda.manual_seed_all(args.seed)

#    os.makedirs(args.rundir, exist_ok=True)
    
#    with open(Path(args.rundir) / 'args.json', 'w') as out:
#        json.dump(vars(args), out, indent=4)

#    train(args.rundir, args.diagnosis, args.epochs, args.learning_rate, args.gpu)
