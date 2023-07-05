from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP
from models.conditional_GAN import CGAN


def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = WGAN_GP(args)
    elif args.model == 'CGAN':
        model = CGAN(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
      if args.model == 'CGAN':

        if args.generate:

          #classes = ['8bars', 'inner', 'outer', 'ball', '4bars', 'healthy', '1bar', 'all']
          encoder = {'8bars':0, 'inner':1, 'outer':2, 'ball':3, '4bars':4, 'healthy':5, '1bar':6}
          model.generate_images_per_class(args.generate, 
                                          288, 32, args.load_G, args.load_D, encoder)


        
        else:

          classes = ['8bars', 'inner', 'outer', 'ball', '4bars', 'healthy', '1bar', 'all']
          model.evaluate(test_loader, args.load_D, args.load_G, classes)
          for i in range(50):
            model.generate_latent_walk(i)

        
     
      else:

        model.evaluate(test_loader, args.load_D, args.load_G)
        for i in range(50):
           model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    print(args.cuda)
    main(args)
