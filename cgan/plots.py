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
    num_labels=10, samples_per_label=5, out_fn=None):

    labels = np.concatenate(
        [np.array([l]*samples_per_label) for l in range(num_labels)]
    )
    cond = np.zeros((len(labels),num_labels), dtype=np.float32)
    cond[np.arange(len(labels)),labels] = 1

    try:
        old_batch_size = noise_gen.batch_size
        noise_gen.batch_size = num_labels*samples_per_label
        noise = next(noise_gen)[0]
    finally:
        noise_gen.batch_size = old_batch_size
    
    generated_images = gen.predict([cond,noise])

    plt.figure(figsize=(1.5*num_labels,1.5*samples_per_label))

    gs = gridspec.GridSpec(samples_per_label, num_labels,
        hspace=0.02, wspace=0.02)

    for k in range(samples_per_label*num_labels):
        j = k//samples_per_label
        i = k%samples_per_label

        plt.subplot(gs[i,j])
        plot_img(generated_images[k,:,:,0])

    if out_fn is not None:
        plt.savefig(out_fn, bbox_inches='tight')
        plt.close()
