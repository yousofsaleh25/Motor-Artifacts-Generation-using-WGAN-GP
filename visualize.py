import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
sns.set_theme()


def train_history(history, save=None):
  epochs = len(history['train_accuracies'])
  plt.figure(figsize=(10, 8))
  plt.plot(range(0, epochs),history['train_accuracies'], 'b', linewidth=1)
  plt.plot(range(0, epochs),history['val_accuracies'], 'r', linewidth=1)
  #plt.plot(history.history['val_accuracy'])
  plt.title('Accuracy', fontsize=15)
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='best')
  plt.grid("on")
  plt.xlim([0, epochs])
  if save:
    plt.savefig(os.path.join(save, 'accuracy.png'))
  plt.show()
  # summarize history for loss



  plt.figure(figsize=(10, 8))
  plt.plot(range(0, epochs),history['train_losses'], 'b', linewidth=1)
  plt.plot(range(0, epochs),history['val_losses'], 'r', linewidth=1)
  plt.legend(['train', 'validation'], loc='best')
  #plt.plot(history.history['val_loss'])
  plt.title('Loss', fontsize=15)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.xlim([0, epochs])
  #plt.legend(['train', 'test'], loc='upper left')
  plt.grid("on")

  if save:
    plt.savefig(os.path.join(save, 'loss.png'))

  plt.show()


def visualize_cmc(cmc, save=None):
  plt.figure(figsize=(10, 8))
  plt.plot(range(1, 101), cmc)

  plt.title('Cumulative Matching Characteristic (CMC)', fontsize=15)
  plt.ylabel('Accuracy')
  plt.xlabel('Rank')
  

  if save:
    plt.savefig(os.path.join(save, 'cmc.png'))

  plt.show()




def test_model(model, test_loader, Ncrop=True):
    model.eval()
    device = next(model.parameters()).device
    
    num_correct = 0
    num_incorrect = 0
    correct_images = []
    
    cpreds = []
    inpreds = []
    incorrect_images = []
    saliency_maps = []
    print(device)
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        #output = model(data)
        
        if Ncrop:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = data.shape
            data = data.view(-1, c, h, w)
            # forward
            #print(model.cuda())
            outputs = model(data)
            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops
        else:
            outputs = model(data)

        preds = outputs.argmax(dim=1, keepdim=True)
        corrects = preds.eq(target.view_as(preds))
  
        for i in range(preds.size(0)):
          if num_correct < 5 and corrects[i].item():
              correct_images.append(data[i])
              cpreds.append(target[i])
              #print(data[i].shape)
              num_correct += 1
          elif num_incorrect < 5 and not corrects[i].item():
              incorrect_images.append(data[i])
              inpreds.append(preds[i])
              num_incorrect += 1
          
          if num_correct >= 5 and num_incorrect >= 5:
              break
    
        if num_correct >= 5 and num_incorrect >= 5:
            break
    
    # Generate saliency maps
    for image, label in zip(correct_images + incorrect_images, cpreds + inpreds):
        print(image.shape)
        image.requires_grad = True
        image = image.unsqueeze(0)
        print(label)

        if Ncrop:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = image.shape
            image = image.view(-1, c, h, w)
            # forward
            output = model(image)
            # combine results across the crops
            output = outputs.view(bs, ncrops, -1)
            output = torch.sum(outputs, dim=1) / ncrops
        else:
            output = model(image)

        
        loss = F.cross_entropy(output, label).to(device)
        model.zero_grad()
        loss.backward()
        saliency_map = image.grad.abs().max(dim=1, keepdim=True)[0]
        saliency_map = saliency_map / saliency_map.max()
        saliency_maps.append(saliency_map)

    # Create grid of images and saliency maps
    grid_images = make_grid(correct_images + incorrect_images, nrow=5, padding=10)
    grid_saliency_maps = make_grid(saliency_maps, nrow=5, padding=10)

    # Plot the results
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the corrected and misclassified images
    axs[0].imshow(np.transpose(grid_images.cpu().numpy(), (1, 2, 0)))
    axs[0].set_title("Correct and Incorrect Images")
    axs[0].axis('off')

    # Plot the saliency maps
    axs[1].imshow(np.transpose(grid_saliency_maps.cpu().numpy(), (1, 2, 0)), cmap='hot')
    axs[1].set_title("Saliency Maps")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


