import matplotlib
matplotlib.use("Agg")
from matplotlib import colors, gridspec, pyplot as plt
import numpy as np


def plot_img(img):
    norm = colors.Normalize(-1,1)
    plt.imshow(img, norm=norm, interpolation='nearest',
        cmap='gray')
    plt.gca().tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_samples(gen, batch_gen, noise_gen,
    num_samples=16, samples_per_row=8, out_fn=None):
    real_images = next(batch_gen)
    noise = next(noise_gen)[0]
    num_samples = min(num_samples, real_images.shape[0])
    generated_images = gen.predict(noise[:num_samples,...])

    num_rows = int(np.ceil(num_samples/samples_per_row))
    num_cols = samples_per_row
    plt.figure(figsize=(1.5*num_cols,2*1.5*num_rows))
    gs = gridspec.GridSpec(2, 1, hspace=0.1)
    gs_real = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols,
        hspace=0.02, wspace=0.02, subplot_spec=gs[0,0])
    gs_gen = gridspec.GridSpecFromSubplotSpec(num_rows, num_cols,
        hspace=0.02, wspace=0.02, subplot_spec=gs[1,0])

    for k in range(num_samples):
        i = k//samples_per_row
        j = k%samples_per_row

        plt.subplot(gs_real[i,j])
        plot_img(real_images[k,:,:,0])

        plt.subplot(gs_gen[i,j])
        plot_img(generated_images[k,:,:,0])

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()
