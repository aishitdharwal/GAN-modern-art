from colab_128_gan_color_images import *


latent_dim = 100
g_model = define_generator(latent_dim)
g_model.load_weights('generator_model_500.h5')

x, _ = generate_fake_samples(g_model, latent_dim, 1000)

x=(x+1)/2.0

import matplotlib.pyplot as plt
def im(x, i):
    plt.imshow(x)
    plt.savefig('images/500_'+'im'+str(i)+'.png')

for i in range(200,300):
    im(x[i], i)

# def show_images(images: list) -> None:
#     n: int = len(images)
#     f = plt.figure(figsize = (50,50))
#     for i in range(n):
#         # Debug, plot figure
#         f.add_subplot(1, n, i + 1)
#         plt.imshow(images[i], interpolation='nearest')

#     plt.show(block=True)


# show_images(x[950:955])