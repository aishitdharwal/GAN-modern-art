from colab_128_gan_color_images import *
import matplotlib.pyplot as plt

def im(x):
    plt.imshow(x)
    plt.savefig('im.png')


latent_dim = 100
g_model = define_generator(latent_dim)
g_model.load_weights('generator_model_300.h5')

x, _ = generate_fake_samples(g_model, latent_dim, 1000)

x=(x+1)/2.0

im(x[800])