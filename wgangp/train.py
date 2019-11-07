from data import MNISTBatchGenerator, NoiseGenerator
from gan import WGANGP
from models import dcgan_disc, dcgan_gen
import plots


def build_gan():
    noise_dim = 64
    gen = dcgan_gen(noise_dim=noise_dim)
    disc = dcgan_disc()
    wgan = WGANGP(gen, disc)
    batch_gen = MNISTBatchGenerator()
    noise_gen = NoiseGenerator([(noise_dim,)])

    return (wgan, batch_gen, noise_gen)


def train_gan(wgan, batch_gen, noise_gen, num_epochs=1, steps_per_epoch=1, plot_fn=None):
    wgan.fit_generator(batch_gen, noise_gen, num_epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch)
    plots.plot_samples(wgan.gen, batch_gen, noise_gen, out_fn=plot_fn)


if __name__ == "__main__":
    (wgan, batch_gen, noise_gen) = build_gan()
    for i in range(200):
        train_gan(wgan, batch_gen, noise_gen, steps_per_epoch=20,
            plot_fn="../figures/wgan_samples_{:03d}.png".format(i))
