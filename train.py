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
        nn.utils.clip_grad_norm_(model.parameters(), 15)
        optimizer.step()

        total_loss += loss.item()

        if step % args.log_interval == 0 and step > 0:
            print(' {:5d}/{:5d} batches | mean loss {:5.2f} ' 
                .format(step, len(train_loader), total_loss / step / args.batch_size))

    return total_loss


def evaluate(eval_model, criterion, valid_loader):
    eval_model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for step, (source, target) in enumerate(valid_loader):
            source = source.to(args.device)
            target = target.to(args.device)
            source = source.transpose(0,1).contiguous()
            target = target.transpose(0,1).contiguous()
            tgt_mask = eval_model.generate_square_subsequent_mask(target.size(0)).to(args.device)

            logits = eval_model(source, target, tgt_mask)

            loss = criterion(logits.view(-1, 1), target.view(-1, 1))
            total_loss += loss

    return total_loss


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = IndexDataset(file_name, args.sheet_name)
    train_split = int(args.split*len(dataset))
    train_set = Subset(dataset, list(range(train_split)))
    valid_set = Subset(dataset, list(range(train_split, len(dataset))))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    model = MyTransformer(args.encoder_nlayers, args.encoder_nhead, args.d_model, args.nhid,  
                            args.decoder_nlayers, args.decoder_nhead, args.dropout).to(args.device) 
    if torch.cuda.device_count() > 1:
        print("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)

    print(model)
    print('param size: {:5.4f} MB'.format(sum(np.prod(v.size()) for v in model.parameters())/1e6))
    print(args.device)

    criterion = nn.MSELoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min, last_epoch=-1)
    

    best_val_loss = float("inf")
    best_model = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, optimizer, criterion, train_loader)
        valid_loss = evaluate(model, criterion, valid_loader)

        print('-' * 80)
        print('| end of epoch {:3d} | time: {:5.2f}s | lr {:1.5f}  | valid mean loss {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time), scheduler.get_last_lr()[0], valid_loss / len(valid_set)))
        print('-' * 80)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_model = model

        scheduler.step()


if __name__ == '__main__':
    main()
