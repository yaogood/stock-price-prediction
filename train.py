import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt
from args import args
from model import MyTransformer
from load_data import MyDataset, load_data
import utils.utils as utils


def train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0.0

    for step, (source, target) in enumerate(train_loader):
        optimizer.zero_grad()

        source = source.to(args.device)
        target = target.to(args.device)
        source = source.transpose(0,1).contiguous()
        target = target.transpose(0,1).contiguous()
        tgt_mask = model.generate_square_subsequent_mask(target.size(0)).to(args.device)

        logits = model(source, target, tgt_mask)

        loss = criterion(logits.view(-1, 1), target.view(-1, 1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        if step % args.log_interval == 0 and step > 0:
            args.log.info(' {:5d}/{:5d} batches | mean loss {:5.2f} ' 
                .format(step, len(train_loader), total_loss / (step+1) / args.batch_size))

    return total_loss


def evaluate(model, criterion, valid_loader):
    model.eval()
    total_loss = 0.0
    y_pred = []

    with torch.no_grad():
        for step, (source, target) in enumerate(valid_loader):
            source = source.to(args.device)
            target = target.to(args.device)
            source = source.transpose(0,1).contiguous()
            target = target.transpose(0,1).contiguous()
            tgt_mask = model.generate_square_subsequent_mask(target.size(0)).to(args.device)

            logits = model(source, target, tgt_mask) # [5*b, 1, 1]

            loss = criterion(logits.view(-1, 1), target.view(-1, 1))
            total_loss += loss
            y_pred.append(logits.squeeze().numpy())

    return total_loss, y_pred


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x, y, x_val, y_val, x_test, y_test, date, close_price_true = \
        load_data(args.dataset_file, args.sheet_name, args.src_seq_len, args.tgt_seq_len)

    train_set = MyDataset(x, y)
    valid_set = MyDataset(x_val, y_val)
    test_set  = MyDataset(x_test, y_test)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=16)

    # model setup
    model = MyTransformer(x.shape[2], args.encoder_nlayers, args.encoder_nhead, args.d_model, args.nhid,  
                            args.decoder_nlayers, args.decoder_nhead, args.dropout).to(args.device) 
    if torch.cuda.device_count() > 1:
        print("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    criterion = nn.MSELoss(reduction='sum').to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min, last_epoch=-1)
    start_epoch = 0

    # initialization
    # if args.run_in_google_colab:
    #     pass  
    if os.path.exists(args.exp_dir):
        checkpoint_path = os.path.join(args.exp_dir, 'model')
        checkpoint_file = os.path.join(checkpoint_path, 'checkpoint.pt')
        print(f'=> resuming from {checkpoint_file}')
        assert os.path.exists(checkpoint_file), 'Error. No checkpoint file.'
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # best_fid = checkpoint['best_fid']

        log = utils.create_logger(os.path.join(args.exp_dir, 'log'))
        log.info(f'=> checkpoint {checkpoint_file} loaded, (epoch {start_epoch})')

    else:
        print(f'start new experiment')
        log_path, checkpoint_path = utils.create_exp_path('./exp')
        log = utils.create_logger(log_path)
        log.info('root experimental dir created: {}'.format('./exp'))

    args.log = log
    # log.info(model)
    log.info('param size: {:5.4f} MB'.format(sum(np.prod(v.size()) for v in model.parameters())/1e6))
    log.info('use {0} to train'.format(args.device))
    log.info(args)

    best_val_loss = float("inf")
    best_model = model
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, optimizer, criterion, train_loader)
        valid_loss, _ = evaluate(model, criterion, valid_loader)
        train_loss_list.append(train_loss / len(train_set))
        valid_loss_list.append(valid_loss / len(valid_set))

        log.info('-' * 80)
        log.info('| end of epoch {:3d} | time: {:5.2f}s | lr {:1.5f}  | train mean loss {:5.7f} | valid mean loss {:5.7f} | '
                .format(epoch, 
                        (time.time() - epoch_start_time), 
                        scheduler.get_last_lr()[0], 
                        train_loss / len(train_set), 
                        valid_loss / len(valid_set)
                        ))
        log.info('-' * 80)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model
            utils.save(checkpoint_path, args, model, optimizer, epoch,is_best=True)
        else:
            utils.save(checkpoint_path, args, model, optimizer, epoch,is_best=False)

        scheduler.step()

    test_loss, y_pred = evaluate(best_model, criterion, test_loader)

    mape_score, r_score, theil_score = utils.measure_all(y_pred, date, close_price_true)
    
    log.info('-' * 80)
    log.info('| test loss {:5.7f} | mape_score: {:2.5f} | r_score {:2.5f} | theil_score {:2.5f} | '
            .format(test_loss / len(test_set), mape_score, r_score, theil_score))
    log.info('-' * 80)

    return train_loss_list, valid_loss_list


if __name__ == '__main__':
    train_loss_list, valid_loss_list = main()
    fig = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(range(len(train_loss_list)), train_loss_list, 'r--', label='train')
    plt.plot(range(len(valid_loss_list)), valid_loss_list, 'b', label='valid')
    plt.title('Mean losses')
    plt.legend()
    plt.show()

