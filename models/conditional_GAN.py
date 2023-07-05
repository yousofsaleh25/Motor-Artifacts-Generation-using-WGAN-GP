import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from utils.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils
#from utils.inception_score import get_inception_score
from utils.utils_plus import generate_images
from utils.metrics import calculate_fretchet, compute_score
from utils.motor import get_test_images
from torchvision.utils import save_image

SAVE_PER_TIMES = 100


class Generator_large(torch.nn.Module):
    def __init__(self, channels, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100 + self.num_classes, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),

            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            # State (64x64x64)
            # nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(num_features=32),
            # nn.ReLU(True),

            # # State (32x128x128)
            # State (64x64x64)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx128x128)

        self.output = nn.Tanh()

    def forward(self, x, label):
        label = F.one_hot(label, num_classes=self.num_classes).float()
        label = label.view(-1, self.num_classes, 1, 1)
        #print(label.shape)
        x = torch.cat([x, label], dim=1)
        x = self.main_module(x)
        return self.output(x)



class Generator(torch.nn.Module):
    def __init__(self, channels, num_classes=7):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100 + num_classes
        # Output_dim = C (number of channels)
        self.num_classes = num_classes
        self.main_module = nn.Sequential(
            # Z latent vector 100 + num_classes
            nn.ConvTranspose2d(in_channels=100 + num_classes, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x, label):
        label = F.one_hot(label, num_classes=self.num_classes).float()
        label = label.view(-1, self.num_classes, 1, 1)
        #print(label.shape)
        x = torch.cat([x, label], dim=1)
        x = self.main_module(x)
        return self.output(x)



class Discriminator_large(torch.nn.Module):
    def __init__(self, channels, num_classes=7):
        super().__init__()
        self.num_classes = num_classes
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx128x128)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx128x128)
            nn.Conv2d(in_channels=channels + self.num_classes, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x64x64)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x32x32)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x16x16)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x, y):
        y_onehot = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3], requires_grad=True).to(x.device)
        y_onehot[torch.arange(x.shape[0]), y, :, :] = 1
        #y_onehot = y_onehot.unsqueeze(2).unsqueeze(3)
        #print(y_onehot.shape)
        x = torch.cat([x, y_onehot], dim=1)
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x, y):
        # Use discriminator for feature extraction then flatten to vector of 16384
        y_onehot = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3], requires_grad=True).to(x.device)
        y_onehot[torch.arange(x.shape[0]), y, :, :] = 1
        #y_onehot = y_onehot.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, y_onehot], dim=1)
        x = self.main_module(x)
        return x.view(-1, 1024*16*16)



class Discriminator(torch.nn.Module):
    def __init__(self, channels, num_classes=7):
        self.num_classes = num_classes
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels+num_classes, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))


    def forward(self, x, y):
        y_onehot = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3], requires_grad=True).to(x.device)
        y_onehot[torch.arange(x.shape[0]), y, :, :] = 1
        #y_onehot = y_onehot.unsqueeze(2).unsqueeze(3)
        #print(y_onehot.shape)
        x = torch.cat([x, y_onehot], dim=1)
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x, y):
        # Use discriminator for feature extraction then flatten to vector of 16384
        y_onehot = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3], requires_grad=True).to(x.device)
        y_onehot[torch.arange(x.shape[0]), y, :, :] = 1
        #y_onehot = y_onehot.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x, y_onehot], dim=1)
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)



class CGAN(object):
    def __init__(self, args, num_classes=7):
        self.num_classes = 7
        print(args)
        if args.large_size:
          self.size = (128, 128)
          self.G = Generator_large(args.channels)
          self.D = Discriminator_large(args.channels)
        else:
          self.size = (32, 32)
          self.G = Generator(args.channels, self.num_classes)
          self.D = Discriminator(args.channels, self.num_classes)
        
        print(f"CGAN_GradientPenalty init model. SIZE {self.size}")
        self.C = args.channels
        self.test_path = args.test_dataroot
        self.train_path = args.train_dataroot
        self.class_name = args.class_name
        # Check if cuda is available
        self.check_cuda(args.cuda)

        # CGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = args.batch_size 

        # CGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        if args.from_checkpoint:
          self.start = self.load_checkpoint(args.from_checkpoint, self.G, self.D, self.g_optimizer, self.d_optimizer)
        else:
          self.start = 0
        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False


    def train(self, train_loader):
        self.t_begin = t.time()
        

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.start, self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images, labels = self.data.__next__()

                # Changed HERE
                # Check for batch to have full batch_size
                # if (images.size()[0] != self.batch_size):
                #     continue

                z = torch.rand((images.size()[0], 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)
                labels = self.get_torch_variable(labels)
                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images, labels)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(images.size()[0], 100, 1, 1))
                #fake_labels = self.get_torch_variable(torch.randint(low=0, high=self.num_classes+1, size=(images.size()[0],), dtype=torch.long))
                #print(fake_labels.shape)

                fake_images = self.G(z, labels)
                d_loss_fake = self.D(fake_images, labels)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data, labels)
                gradient_penalty.backward()


                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_labels = torch.randint(low=0, high=self.num_classes, size=(self.batch_size,))
            fake_labels = self.get_torch_variable(fake_labels)

            fake_images = self.G(z, fake_labels)
            g_loss = self.D(fake_images, fake_labels)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % SAVE_PER_TIMES == 0:
                self.save_model()
                #self.generate_test_examples(self.G, batch_size=self.batch_size, num_images=10, save_path=f"./generated_images_{g_iter}")
                
                
                test_images, test_labels = get_test_images(self.test_path, class_name=self.class_name, transform='default',  get_labels=True)
                train_images, train_labels = get_test_images(self.train_path, class_name=self.class_name, transform='default', get_labels=True)
                
                #print(test_images.size)
                z = Variable(torch.randn(test_images.size()[0], 100, 1, 1)).cuda(self.cuda_index)
                test_fake_images = self.G(z, test_labels.cuda(self.cuda_index))

                z = Variable(torch.randn(train_images.size()[0], 100, 1, 1)).cuda(self.cuda_index)
                train_fake_images = self.G(z, train_labels.cuda(self.cuda_index))
                #print(torch.max(fake_images))
                #print(fake_images.max())
                #print(fake_images.min())
                #print(train_images.max())
                #print(train_images.min())
                
                
                # Workaround because graphic card memory can't store more than 830 examples in memory for generating image
                # Therefore doing loop and generating 800 examples and stacking into list of samples to get 8000 generated images
                # This way Inception score is more correct since there are different generated examples from every class of Inception model
                # sample_list = []
                # for i in range(125):
                #     samples  = self.data.__next__()
                #     z = Variable(torch.randn(800, 100, 1, 1)).cuda(self.cuda_index)
                #     samples = self.G(z)
                #     sample_list.append(samples.data.cpu().numpy())
                #
                # # Flattening list of list into one list
                # new_sample_list = list(chain.from_iterable(sample_list))
                # print("Calculating Inception Score over 8k generated images")
                # # # Feeding list of numpy arrays
                # inception_score = get_inception_score(new_sample_list, cuda=True, batch_size=32,
                #                                       resize=True, splits=10)

                if not os.path.exists('training_result_images/'):
                    os.makedirs('training_result_images/')

                # Denormalize images and save them in grid 8x8
                # Denormalize images and save them in a grid with num_classes*num_classes rows
                z = self.get_torch_variable(torch.randn(self.num_classes*self.num_classes, 100, 1, 1))
                fake_labels = []
                for i in range(self.num_classes):
                    fake_labels.extend([i]*self.num_classes)
                fake_labels = self.get_torch_variable(torch.LongTensor(fake_labels))
                samples = self.G(z, fake_labels)
                samples = samples.mul(0.5).add(0.5)
                samples = samples.data.cpu()

                rows = []
                for i in range(self.num_classes):
                    row = torch.cat([samples[j] for j in range(i*self.num_classes, (i+1)*self.num_classes)], dim=2)
                    rows.append(row)
                grid = torch.cat(rows, dim=1)
                utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))


                # Testing
                train_fid = calculate_fretchet(train_images, train_fake_images)
                test_fid = calculate_fretchet(test_images, test_fake_images)
                #print(test_images.size())
                train_s = compute_score(train_images, train_fake_images, k=1, sigma=1, sqrt=True)
                test_s = compute_score(test_images, test_fake_images, k=1, sigma=1, sqrt=True)
                #print(test_s.mmd)
                #print(test_s.emd)
                #print(test_s.knn.acc)

                import datetime

                time_diff = t.time() - self.t_begin
                time_str = str(datetime.timedelta(seconds=time_diff))

                #print("Elapsed time: " + time_str)
                print("\n\nTest FID: {}".format(test_fid))
                print("Test EMD: {}".format(test_s.emd))
                print("Test MMD: {}".format(test_s.mmd))
                print("Test KNN: {}".format(test_s.knn.acc))
                print("-"*55)
                print("Train FID: {}".format(train_fid))
                print("Train EMD: {}".format(train_s.emd))
                print("Train MMD: {}".format(train_s.mmd))
                print("Train KNN: {}\n\n".format(train_s.knn.acc))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time_str))


                # create header if g_iter is 1
                if g_iter == 0:
                    header = "Epochs\tTime\tTrain MMD\tTest MMD\tTrain FID\tTest FID\tTrain KNN\tTest KNN\tTrain EMD\tTest EMD\n"
                    with open("test_metrics.txt", "w") as f:
                        f.write(header)

                # Write to file inception_score, gen_iters, time
                # format output string
                output = f"{g_iter}\t{time_str}\t{train_s.mmd:.2f}\t{test_s.mmd:.2f}\t{train_fid:.2f}\t{test_fid:.2f}\t{train_s.knn.acc:.2f}\t{test_s.knn.acc:.2f}\t{train_s.emd:.2f}\t{test_s.emd:.2f}\n"

                # write to file
                with open("test_metrics.txt", "a") as f:
                    f.write(output)

                
                # ============ Saving Check points ============#
                os.makedirs('checkpoints', exist_ok=True)
                checkpoint_path = f"./checkpoints/checkpoint_{g_iter}.pth"
                self.save_checkpoint(g_iter, self.G, self.D, self.g_optimizer, self.d_optimizer, checkpoint_path)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.cpu(), g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)



        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()


    def evaluate(self, test_loader, D_model_path, G_model_path, classes=None):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.num_classes*self.num_classes, 100, 1, 1))
        fake_labels = []
        for i in range(self.num_classes):
            fake_labels.extend([i]*self.num_classes)
        fake_labels = self.get_torch_variable(torch.LongTensor(fake_labels))
        samples = self.G(z, fake_labels)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()

        rows = []
        for i in range(self.num_classes):
            row = torch.cat([samples[j] for j in range(i*self.num_classes, (i+1)*self.num_classes)], dim=2)
            rows.append(row)
        grid = torch.cat(rows, dim=1)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')
        
        if classes:
          for c in classes:

              test_images, test_labels = get_test_images(self.test_path, class_name=c, transform='default',  get_labels=True)
              train_images, train_labels = get_test_images(self.train_path, class_name=c, transform='default', get_labels=True)

              z = Variable(torch.randn(test_images.size()[0], 100, 1, 1)).cuda(self.cuda_index)
              test_fake_images = self.G(z, test_labels.cuda(self.cuda_index))

              z = Variable(torch.randn(train_images.size()[0], 100, 1, 1)).cuda(self.cuda_index)
              train_fake_images = self.G(z, train_labels.cuda(self.cuda_index))

              # Testing
              train_fid = calculate_fretchet(train_images, train_fake_images)
              test_fid = calculate_fretchet(test_images, test_fake_images)
              #print(test_images.size())
              train_s = compute_score(train_images, train_fake_images, k=1, sigma=1, sqrt=True)
              test_s = compute_score(test_images, test_fake_images, k=1, sigma=1, sqrt=True)
              #print(test_s.mmd)
              #print(test_s.emd)
              #print(test_s.knn.acc)

              print(25*"-" + f" {c} " + 25*"-")
              #print("Elapsed time: " + time_str)
              print("\n\nTest FID: {}".format(test_fid))
              print("Test EMD: {}".format(test_s.emd))
              print("Test MMD: {}".format(test_s.mmd))
              print("Test KNN: {}".format(test_s.knn.acc))
              print("-"*55)
              print("Train FID: {}".format(train_fid))
              print("Train EMD: {}".format(train_s.emd))
              print("Train MMD: {}".format(train_s.mmd))
              print("Train KNN: {}\n\n".format(train_s.knn.acc))

        # self.load_model(D_model_path, G_model_path)
        # z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        # samples = self.G(z)
        # samples = samples.mul(0.5).add(0.5)
        # samples = samples.data.cpu()
        # grid = utils.make_grid(samples)
        # print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        # utils.save_image(grid, 'dgan_model_image.png')




    def generate_images_per_class(self, base, num_images_per_class, 
                                  batch_size, G_model_path, D_model_path, encoder):
        classes = list(encoder.keys())                          
        self.load_model(D_model_path, G_model_path)
        for c in classes:
            # Create directory for class
            class_dir = os.path.join(base, c)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate images for class
            num_batches = num_images_per_class // batch_size
            for i in range(num_batches):
                z = torch.randn(batch_size, 100, 1, 1).cuda(self.cuda_index)
                labels = torch.LongTensor([encoder[c]] * batch_size).cuda(self.cuda_index)
                images = self.G(z, labels)
                images = images.mul(0.5).add(0.5)  # Convert images to 0-1 range
                
                # Save images to class directory
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                for j in range(batch_size):
                    save_image(images[j], os.path.join(class_dir, f"{start_idx+j}.png"))


    def calculate_gradient_penalty(self, real_images, fake_images, labels):
        #print(real_images.size())
        #print(fake_images.size())
        eta = torch.FloatTensor(real_images.size()[0], 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(real_images.size()[0], real_images.size(1), real_images.size(2), real_images.size(3))
        #print(eta.size())
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, labels)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, self.size[0], self.size[1])[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, self.size[0], self.size[1])[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        fake_labels = torch.randint(low=0, high=self.num_classes, size=(z.size()[0],))
        fake_labels = self.get_torch_variable(fake_labels)
        samples = self.G(z, fake_labels).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, self.size[0], self.size[1]))
            else:
                generated_images.append(sample.reshape(self.size[0], self.size[1]))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def generate_test_examples(self, generator, batch_size=32, num_images=77, save_path="./generated_images"):
        generate_images(generator, batch_size, num_images, save_path)

    # new
    def save_checkpoint(self, epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_path):
        state_dict = {
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'g_optimizer': g_optimizer.state_dict(),
            'd_optimizer': d_optimizer.state_dict()
        }
        torch.save(state_dict, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch}")

    # new
    def load_checkpoint(self, checkpoint_path, generator, discriminator, g_optimizer, d_optimizer):
        
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        print(f"Loaded checkpoint at epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, labels) in enumerate(data_loader):
                yield images, labels

    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        fake_label = torch.randint(low=0, high=self.num_classes, size=(1,))
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()
            fake_label = fake_label.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp, fake_label)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,self.size[0],self.size[1]).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
