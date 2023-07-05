# Motor-Artifacts-Generation-using-WGAN-GP
Motor Artifacts Generation using WGAN-GP
WGAN-GP
![WGAN](https://github.com/yousofsaleh25/Motor-Artifacts-Generation-using-WGAN-GP/assets/43546116/a7aaa762-d7f1-4a54-8a8b-5d7f322735bf)
To run the WGAN-GP code to generate 128x128 images use --> 

!python main.py --model WGAN-GP \
 \t               --is_train True \
                --download True \
                --train_dataroot MotorImage_train \
                --test_dataroot MotorImage_test \
                --class_name all \
                --large_size True \
                --dataset motor \
                --generator_iters 5001 \
                --cuda True \
                --batch_size 64 

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
