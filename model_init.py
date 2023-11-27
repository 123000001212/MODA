import models.model_mnist
import models.model_SVHN
import models.model_FashionMNIST
import models.model_GTSRB
import models.model_cifar10
import models.model_vgg19

def model_init(dataset, device):
    if dataset == 'FashionMNIST' or dataset == 'MNIST':
        latent_size = 128
        nb_classes = 10 
        D = models.model_FashionMNIST.Discriminator(x_dim=3, c_dim=10).to(device)
        G = models.model_FashionMNIST.Generator(latent_size, 128, nb_classes).to(device)
        return D, G

    elif dataset == 'SVHN':
        G = models.model_SVHN.OldGenerator(128, 128, 10).to(device)
        D = models.model_SVHN.OldDiscriminator(128, 10).to(device)
        return D, G
    


    elif dataset == 'GTSRB':
        latent_size = 128
        nb_classes = 43
        D = models.model_GTSRB.Discriminator().to(device)
        G = models.model_GTSRB.Generator(latent_size, 128, nb_classes).to(device)
        return D, G

    elif dataset == 'CIFAR10':
        latent_size = 128
        nb_classes = 10
        G = models.model_SVHN.OldGenerator(128, 128, 10).to(device)
        D = models.model_cifar10.AlexNet(3, nb_classes).to(device)
        return D, G