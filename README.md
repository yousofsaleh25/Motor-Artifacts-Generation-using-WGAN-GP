# Motor Artifacts Generation using Generative Adversarial Network
This project focuses on generating high-quality and diverse motor artifact images using two generative adversarial network (GAN) approaches: Wasserstein GAN with Gradient Penalty (WGAN-GP) and Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP). The purpose of this ReadMe is to provide an overview of the project, explain the frameworks used, showcase generated images, and present the results obtained.

## Overview
The project aims to generate motor artifact images that closely resemble real motor artifacts. Two GAN approaches, WGAN-GP and cWGAN-GP, were employed for this purpose. The generated images were created at two different resolutions: 32x32 for easy comparisons between GAN architectures and 128x128 for higher resolution outputs. Various evaluation metrics, including FID, EMD, and MMD, were utilized to assess the performance of the models.

## WGAN-GP Framework
The WGAN-GP algorithm is a generative adversarial network used for image generation. It employs a Wasserstein distance metric to measure the difference between generated and real images. The framework consists of a discriminator (D) and a generator (G) trained to generate realistic images. The discriminator distinguishes between real and fake images, while the generator aims to deceive the discriminator. A gradient penalty is added to ensure the discriminator's smoothness. The WGAN-GP framework utilizes a neural network-based generator model and a three-layered discriminator neural network.

## cWGAN-GP Framework
To further enhance the generation of motor artifact images, the project implemented a conditional version of the WGAN-GP algorithm called cWGAN-GP. The cWGAN-GP approach incorporates class information into the generation process, allowing for the generation of images belonging to specific classes. The framework modifies the WGAN-GP framework by introducing a condition vector to both the generator and discriminator input. The generator takes a concatenated input of the latent vector and the condition vector, while the discriminator takes a concatenated input of the image and the condition vector.

## Results
The generated images using both the WGAN-GP and cWGAN-GP approaches demonstrate a high degree of variability and resemble real motor artifacts. These images showcase the effectiveness of both approaches in generating diverse and high-quality images. Evaluation metrics, such as FID, EMD, and MMD, indicate that both approaches successfully generate realistic motor artifact images with low divergence from real images. The cWGAN-GP approach, which incorporates class information, shows superior performance in terms of FID, EMD, and classification accuracy, allowing for more control over the generated images.

## Conclusion and Future Work
In conclusion, the project successfully generates high-quality and diverse motor artifact images using the WGAN-GP and cWGAN-GP approaches. The cWGAN-GP approach, in particular, demonstrates superior performance by incorporating class information and generating images closer to real motor artifact images. Future work could explore other GAN architectures or incorporate additional types of information, such as temporal information, to further improve the generation of motor artifact images.

![WGAN](https://github.com/yousofsaleh25/Motor-Artifacts-Generation-using-WGAN-GP/assets/43546116/a7aaa762-d7f1-4a54-8a8b-5d7f322735bf)
To run the WGAN-GP code to generate 128x128 images use --> 

      !python main.py --model WGAN-GP \
                      --is_train True \
                      --download True \
                      --train_dataroot MotorImage_train \
                      --test_dataroot MotorImage_test \
                      --class_name all \
                      --large_size True \
                      --dataset motor \
                      --generator_iters 5001 \
                      --cuda True \
                      --batch_size 64 

if --large_size is set to False the generated images will be of size 32x32

Conditional WGAN-GP
![cGAN](https://github.com/yousofsaleh25/Motor-Artifacts-Generation-using-WGAN-GP/assets/43546116/206b9e34-caa1-4a6f-857d-fe05ec2465bc)
To run the Conditional WGAN-GP to generate 128x128 images for the selected class use -->

      !python main.py --model CGAN \
                      --is_train True \
                      --download True \
                      --train_dataroot MotorImage_train \
                      --test_dataroot MotorImage_test \
                      --class_name all \
                      --large_size True \
                      --dataset motor \
                      --generator_iters 5001 \
                      --cuda True \
                      --batch_size 64 
