import torch
import torchvision
from torch.optim import SGD, Adam
import torch.nn as nn
import time
import numpy as np
import os

def train_batch(x, y, model, opt, loss_fn, Ncrop=False):
    
    
    model.train()
    if Ncrop:
        # fuse crops and batchsize
        bs, ncrops, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        # forward
        outputs = model(x)
        # combine results across the crops
        outputs = outputs.view(bs, ncrops, -1)
        outputs = torch.sum(outputs, dim=1) / ncrops
    else:
        outputs = model(x)
        
        
    batch_loss = loss_fn(outputs, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item()



@torch.no_grad()
def accuracy(x, y, model, Ncrop=False):
    model.eval()
    
    if Ncrop:
        # fuse crops and batchsize
        bs, ncrops, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        # forward
        outputs = model(x)
        # combine results across the crops
        outputs = outputs.view(bs, ncrops, -1)
        outputs = torch.sum(outputs, dim=1) / ncrops
    else:
        outputs = model(x)
        
    predictions = np.argmax(outputs.cpu().detach().numpy(), axis=-1)
    #prediction = model(x)
    is_correct = (predictions == y.cpu().numpy())
    return is_correct.tolist()


@torch.no_grad()
def val_loss(x, y, model, loss_fn, Ncrop=False):
    model.eval()
    
    if Ncrop:
        # fuse crops and batchsize
        bs, ncrops, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        # forward
        outputs = model(x)
        # combine results across the crops
        outputs = outputs.view(bs, ncrops, -1)
        outputs = torch.sum(outputs, dim=1) / ncrops
    else:
        outputs = model(x)
        
    val_loss_ = loss_fn(outputs, y)
    return val_loss_.cpu().item()


def train_model(model, num_epochs, learning_rate, output_path, train_dl, valid_dl, continue_training=False, Ncrop=False):
    
    
    output_dir = output_path.split('/')[:-1]
    output_dir =  '/' + os.path.join(*output_dir)
    os.makedirs(output_dir, exist_ok=True)
    #os.maked
    #print(output_dir)
    #print(output_path.split('/'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    #train_dl, valid_dl = get_AMIEarDataset(batch_size=batch_size,transform=transform)
    #model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    if not continue_training:
        start_epoch = 0
        best_acc = 0.0
    else:
        #print("here")
        try:
            checkpoint = torch.load(continue_training)
        except:
            print("Model Not Found!")
        start_epoch = checkpoint['epoch'] + 1
        #print(checkpoint.keys())
        best_acc = checkpoint['best_accuracy']
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    loss_fn = nn.CrossEntropyLoss()
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=0.5, patience=20, threshold=0.001,
                verbose=True, min_lr=1e-5,
                threshold_mode='abs')


    
    EPOCHS = num_epochs
    for epoch in range(start_epoch, EPOCHS):
        since = time.time()
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_losses, val_epoch_accuracies = [], []
        print('Epoch {}/{}'.format(epoch, EPOCHS-1))
        print('-' * 10)

        for ix, batch in enumerate(iter(train_dl)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn, Ncrop)
            train_epoch_losses.append(batch_loss)

        train_epoch_loss = np.array(train_epoch_losses).mean()
        
        for ix, batch in enumerate(iter(train_dl)):
            x, y = batch
            is_correct = accuracy(x, y, model, Ncrop)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        for ix, batch in enumerate(iter(valid_dl)):
            x, y = batch
            val_is_correct = accuracy(x, y, model, Ncrop)
            val_l = val_loss(x, y, model, loss_fn, Ncrop)
            val_epoch_accuracies.extend(val_is_correct)
            val_epoch_losses.append(val_l)

        val_epoch_accuracy = np.mean(val_epoch_accuracies)
        val_epoch_loss = np.mean(val_epoch_losses)
        schedular.step(val_epoch_loss)

        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        if val_epoch_accuracy >= best_acc:
            best_acc = val_epoch_accuracy
            torch.save({
                        'epoch': epoch,
                        'best_accuracy': best_acc,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        
                       }, os.path.join(output_dir, output_path.split('.')[0]+'_BEST.pt'))
            print("------Saving Best-------")
            
        torch.save({
                    'epoch': epoch,
                    'best_accuracy': best_acc,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                   }, output_path)
        print("------Saving-------")
        
        time_elapsed = time.time()
        phase = 'Train'
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, train_epoch_loss, train_epoch_accuracy))

        phase = 'Test'
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, val_epoch_loss, val_epoch_accuracy))
        print(f"Epoch Time {int(time_elapsed - since)}s")
        
        
    history = {}
    history['train_losses'] = train_losses
    history['train_accuracies'] = train_accuracies
    history['val_losses'] = val_losses
    history['val_accuracies'] = val_accuracies

    return model, history