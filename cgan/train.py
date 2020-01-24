from data import MNISTBatchGenerator, NoiseGenerator
from gan import GAN
from models import cgan_disc, cgan_gen
import plots


def build_gan():
    noise_dim = 64
    gen = cgan_gen(noise_dim=noise_dim)
    disc = cgan_disc()
    gan = GAN(gen, disc)
    batch_gen = MNISTBatchGenerator()
    noise_gen = NoiseGenerator([(noise_dim,)])

    return (gan, batch_gen, noise_gen)


def train_gan(gan, batch_gen, noise_gen, num_epochs=1, steps_per_epoch=1, plot_fn=None):
    gan.fit_generator(batch_gen, noise_gen, num_epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch)
    plots.plot_samples(gan.gen, batch_gen, noise_gen, out_fn=plot_fn)


if __name__ == "__main__":
    (gan, batch_gen, noise_gen) = build_gan()
    for i in range(200):
        train_gan(gan, batch_gen, noise_gen, steps_per_epoch=20,
            plot_fn="../figures/cgan_samples_{:03d}.png".format(i))
