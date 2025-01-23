from ia.algo.arrangement.models import ClusterGNN
# from ia.data import ClusterGNNDataLoader, ClusterGNNDataset
from data import ClusterGNNDataLoader, ClusterGNNDataset
import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import json
from functools import partial
from torch_geometric.utils import unbatch
from utils import COLORS

def validate(data_loader, model: ClusterGNN, criterion, save_dir='.', render=False):
    """Used to monitor progress on a validation set & optionally plot solution."""   

    def forward_hook(module, data_input, data_output, fmap_block):
        fmap_block.append(data_output)
        # input_block.append(data_input)
    
    fmblock = []

    hook_fn = partial(forward_hook, fmap_block=fmblock)
    
    model.eval()

    loss = []

    for batch_idx, (bsz_data, target, fields) in enumerate(data_loader):
        bsz_data = bsz_data.to(device)
        target = [t.to(device) for t in target]
        
        if batch_idx == 0:
            hook = model.ff[-1].register_forward_hook(hook_fn)

        # Full forward pass through the dataset
        with torch.no_grad():
            # tic = time.time()
            output = model(bsz_data)
            # print(time.time() - tic)
       
        loss += [criterion(out, tar).item() for out, tar in zip(output, target)]
    
        if batch_idx == 0 and render:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            hook.remove()
            feature_2d = [torch.pca_lowrank(out, 2)[0].cpu().numpy()
                          for out in unbatch(fmblock[0], bsz_data.batch)]
            for idx, (feature, field) in enumerate(zip(feature_2d, fields)):
                name = 'field%d.png'%(idx)
                path = os.path.join(save_dir, name)
                ax = plt.subplot(1, 2, 1)
                field.render(ax, show=False, line_colors = COLORS[:field.num_fields])
                ax = plt.subplot(1, 2, 2)
                ax.plot(feature[0][0], feature[0][1], '*y')
                for line, field_idx in zip(feature[1:], field.line_field_idx):
                    ax.plot(line[0], line[1], 'o', color=COLORS[field_idx])
                plt.savefig(path)
                ax.clear()
                ax = plt.subplot(1, 2, 1)
                ax.clear()
            plt.close('all')
    model.train()
    return np.mean(loss)

def train(model: ClusterGNN, save_dir, optimizer, criterion, train_data, valid_data, 
          batch_size, max_grad_norm, epochs, cur_epoch=0, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = ClusterGNNDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = ClusterGNNDataLoader(dataset=valid_data, batch_size=40, shuffle=False)

    best_loss = np.inf

    while cur_epoch < epochs:

        model.train()

        times, losses = [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, (bsz_data, target, _) in enumerate(train_loader):
            bsz_data = bsz_data.to(device)
            target = [t.to(device) for t in target]
            
            # Full forward pass through the dataset
            output = model(bsz_data)
#           
            loss = torch.zeros(len(target))
            for idx in range(len(target)):
                # assert torch.all(output[idx] >= 0)
                # assert torch.all(output[idx] <= 1)
                loss[idx] = criterion(output[idx], target[idx])
            loss = torch.mean(loss)
            # assert not torch.isnan(loss).any()

            losses.append(loss.detach().item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # assert not torch.isnan(model.parameters()[0]).any()
            optimizer.step()
            # assert not torch.isnan(model.parameters()[0]).any()
            # assert not torch.isnan(model.parameters()[0].grad).any()

            # if (batch_idx + 1) % 50 == 0:
            #     end = time.time()
            #     times.append(end - start)
            #     start = end

            #     mean_loss = np.mean(losses[-50:])

            #     print('Batch %d/%d, loss: %2.4f, took: %2.4fs' %
            #           (batch_idx, len(train_loader), mean_loss, times[-1]))
        
        mean_loss = np.mean(losses)

        # Save the weights      
        save_path = os.path.join(checkpoint_dir, 'checkpoint_Epoch' + str(cur_epoch) + '.pt')
        checkpoint = {
            'epoch': cur_epoch,
            'gnn': model.gnn.state_dict(),
            'ff': model.ff.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, 'valid', '%s' % cur_epoch)

        mean_valid = validate(valid_loader, model, criterion, valid_dir)

        if (cur_epoch+1) % 50 == 0 or cur_epoch == 0:
            valid_loader.dataset.use_field = True
            validate(valid_loader, model, criterion, valid_dir, render=True)
            valid_loader.dataset.use_field = False

        with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
            f.write(str(mean_loss) + ' ' + str(mean_valid) + '\n')

        # Save best model parameters
        if mean_valid < best_loss:
            best_loss = mean_valid
            save_path = os.path.join(save_dir, 'best_model' + str(cur_epoch) + '.pt')
            torch.save(checkpoint, save_path)

        print('[Epoch' + str(cur_epoch) + ']' + 
              'Mean epoch loss: %2.4f, ' \
              'vailid loss: %2.4f, took: %2.4fs ' % \
              (mean_loss, mean_valid, time.time() - epoch_start))
        
        cur_epoch += 1

def main(cfg):
    # Load config, and start training. Do testing after training.
    model_cfg = cfg.pop('model_config')
    if model_cfg is not None:
        if isinstance(model_cfg, str):
            with open(model_cfg, 'r') as f:
                model_cfg = yaml.safe_load(f)
    model = ClusterGNN(**model_cfg, device=device)
    
    save_cfg = cfg
    save_cfg.update(model.cfg)
    print(save_cfg)

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(cfg['log_dir'], now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg['save_dir'] = save_dir
    with open(os.path.join(save_dir, 'config.json'), 'a') as f:
        json.dump(save_cfg, f, indent=1)

    if cfg['checkpoint'] is not None:
        checkpoint = torch.load(cfg['checkpoint'], map_location=device)
        print("Load checkpoint from " + cfg['checkpoint'])
        model.gnn.load_state_dict(checkpoint['gnn'])     
        model.ff.load_state_dict(checkpoint['ff'])     

    cfg['criterion'] = nn.BCELoss()      
            
    if not cfg['test']:
        # Prepare Training
        train_data_list = []
        for data_set in cfg['train_data']:
            train_data_list += [os.path.join(data_set, x) for x in os.listdir(data_set) if x.split('.')[-1] != 'txt']
        valid_data_list = []
        for data_set in cfg['valid_data']:
            valid_data_list += [os.path.join(data_set, x) for x in os.listdir(data_set) if x.split('.')[-1] != 'txt']
        
        train_data = ClusterGNNDataset(train_data_list)
        valid_data = ClusterGNNDataset(valid_data_list)
        cfg['train_data'] = train_data
        cfg['valid_data'] = valid_data
        cfg['optimizer'] = optim.Adam(model.parameters(), lr=cfg.pop('lr'))

        if cfg['checkpoint'] is not None: # Load checkpoint
            cfg['cur_epoch'] = checkpoint['epoch'] + 1
            cfg['optimizer'].load_state_dict(checkpoint['optimizer'])

        print('====================== Start Training ======================')
        train(model, **cfg)
    
    # Testing
    test_dir = os.path.join(cfg['log_dir'], 'test')
    test_data_list = []
    for data_set in cfg['test_data']:
        test_data_list += [os.path.join(data_set, x) for x in os.listdir(data_set) if x.split('.')[-1] != 'txt']
    
    test_data = ClusterGNNDataset(test_data_list)
       
    test_loader = ClusterGNNDataLoader(dataset=test_data, batch_size=24, shuffle=False)
    
    mean_test = validate(test_loader, model, cfg['criterion'], test_dir)

    print('Average BCE loss: ', mean_test)

if __name__ == '__main__':
    from guppy import hpy

    parser = argparse.ArgumentParser(description='GNN Pre-Train')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--train_data', nargs='+')
    parser.add_argument('--valid_data', nargs='+')
    parser.add_argument('--test_data', nargs='*')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=20)
    
    args = parser.parse_args()
    args = vars(args)
    dev = args.pop('device')
    if dev == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(dev if torch.cuda.is_available() else 'cpu')
    
    config = args.pop('config')
    if config is not None:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        config.update(args)
    else:
        config = args
    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)
    main(config)


    hxx = hpy()
    heap = hxx.heap()
    byrcs = hxx.heap().byrcs
    print(heap)
