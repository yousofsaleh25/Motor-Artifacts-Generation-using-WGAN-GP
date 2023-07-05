import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np


def generate_images(generator, batch_size=32, num_images=288, save_path="./generated_images", device='cuda'):
    # Create directory to save images
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Generate and save images
    for i in range(num_images // batch_size):
        z = torch.randn(batch_size, 100, 1, 1).to(device)
        fake_images = generator(z)
        for j in range(batch_size):
            image_path = os.path.join(save_path, f"image_{i*batch_size+j:03d}.png")
            save_image(fake_images[j], image_path)


def visualize_metrics(file_path):
  # Epochs	Time	Test MMD	Test FID	Test KNN	Test EMD
  # Epochs 0	Time 1	Train MMD 2	Test MMD 3	Train FID 4	Test FID 5	Train KNN 6	Test KNN 7	Train EMD 8	Test EMD 9
  with open(file_path, 'r') as f:
    content = f.readlines()

  epochs = []
  test_fids = []
  test_emds = []
  test_mmds = []
  test_knns = []

  train_fids = []
  train_emds = []
  train_mmds = []
  train_knns = []

  for line in content:
    #print(line.split('\t')) 
    epoch = line.split('\t')[0]
    if epoch == 'Epochs':
      continue
    test_fid = line.split('\t')[5]
    test_emd = line.split('\t')[9]
    test_mmd = line.split('\t')[3]
    test_knn = line.split('\t')[7]

    train_fid = line.split('\t')[4]
    train_emd = line.split('\t')[8]
    train_mmd = line.split('\t')[2]
    train_knn = line.split('\t')[6]

    epochs.append(int(epoch))
    test_fids.append(round(float(test_fid), 4))
    test_emds.append(round(float(test_emd), 4))
    test_mmds.append(round(float(test_mmd), 4))
    test_knns.append(round(float(test_knn), 4))

    train_fids.append(round(float(train_fid), 4))
    train_emds.append(round(float(train_emd), 4))
    train_mmds.append(round(float(train_mmd), 4))
    train_knns.append(round(float(train_knn), 4))





  fig, axs = plt.subplots(4, 1, figsize=(10, 16))

  axs[0].plot(epochs, test_fids, 'r', label='Test FID')
  axs[0].plot(epochs, train_fids, 'b', label='Train FID')
  axs[0].set_title("Frechet Inception Distance", fontsize=15)
  axs[0].set_xlabel("Epochs")
  axs[0].set_ylabel("FID")
  axs[0].legend()

  axs[1].plot(epochs, test_emds, 'r', label='Test EMD')
  axs[1].plot(epochs, train_emds, 'b', label='Train EMD')
  axs[1].set_title("Earth Mover's Distance", fontsize=15)
  axs[1].set_xlabel("Epochs")
  axs[1].set_ylabel("EMD")
  axs[1].legend()

  axs[2].plot(epochs, test_mmds, 'r', label='Test MMD')
  axs[2].plot(epochs, train_mmds, 'b', label='Train MMD')
  axs[2].set_title("Maximum Mean Discrepancy", fontsize=15)
  axs[2].set_xlabel("Epochs")
  axs[2].set_ylabel("MMD")
  axs[2].legend()

  axs[3].plot(epochs, test_knns, 'r', label='Test KNN')
  axs[3].plot(epochs, train_knns, 'b', label='Train KNN')
  axs[3].set_title("K-Nearest Neighbors Distance", fontsize=15)
  axs[3].set_xlabel("Epochs")
  axs[3].set_ylabel("KNN")
  axs[3].legend()


  plt.tight_layout()
  plt.savefig("metrics.png", dpi=300)
  plt.show()