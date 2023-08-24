import os
import numpy as np
import tensorflow as tf
import sklearn.metrics
import matplotlib.pyplot as plt
from scipy import linalg
import sys
from tensorflow.python.training.moving_averages import assign_moving_average

def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=12)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k, class_name=None, fig_path=None,
                 percentile=100, filter_real=None):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
        class_name: str, name of the class we are computing the score
        fig_path: str, path to save the figure
        percentile: float, threshold to filter the real distances
        filter_real: mode of filtering, either 'clip' for clipping to percentile, or 'ignore' to only consider distances
        below the percentile threshold
    Returns:
        dict of precision, recall, density, and coverage.
    """

    print('Num real: {} Num fake: {}'
          .format(real_features.shape[0], fake_features.shape[0]))

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    real_nn_percentile = np.percentile(real_nearest_neighbour_distances, percentile)

    if class_name:
        if class_name == 'EBSD/D':
            class_name = 'EBDS_S'

        cp = sns.color_palette()
        with sns.plotting_context('poster'):
            fig, ax = plt.subplots(1, 1, figsize=[12, 9])
            ax.hist(real_nearest_neighbour_distances, bins=10, rwidth=.85)
            ax.axvline(real_nn_percentile, linestyle='--', color=cp[0], label='P{}'.format(percentile))

            ax.set_title('Class {}: {}'.format(class_name, real_features.shape[0]))
            ax.legend()
            fig.savefig(os.path.join(fig_path, 'real_distances_{}.png'.format(class_name)))
            # plt.show()

    # Filter the tails of the real NN_k distribution
    if filter_real:
        out_mask = real_nearest_neighbour_distances > real_nn_percentile
        in_mask = real_nearest_neighbour_distances < real_nn_percentile
        if filter_real == 'clip':
            real_nearest_neighbour_distances[out_mask] = real_nn_percentile
        elif filter_real == 'ignore':
            real_nearest_neighbour_distances = real_nearest_neighbour_distances[in_mask]
            distance_real_fake = distance_real_fake[in_mask, :]

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


def compute_realism(real_features, fake_features, nearest_k):
    """
    Computes realism of the fake samples given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([M, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        r_score: numpy.ndarray([M,], dtype=np.float32)
    """

    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)  # shape ([N,]
    medians = np.median(real_nearest_neighbour_distances, keepdims=True)
    medians_mask = real_nearest_neighbour_distances < medians
    # The following has shape [M,N]
    distance_fake_real = compute_pairwise_distance(fake_features, real_features)
    # We use the broadcast division matching the shapes
    frac = real_nearest_neighbour_distances / distance_fake_real
    frac = frac[:, medians_mask]
    realism = np.max(frac, axis=-1)  # shape [M,]
    return realism


def realism_filter(real_features, fake_features, nearest_k, top_fraction, axis=-1):
    """
    Given a set of fake features, returns the indices of the top percentile
     realistic samples according to the realism score
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([M, feature_dim], dtype=np.float32)
        nearest_k: int.
        top_fraction: int or float

    Returns:
        top_indices: numpy.ndarray[int(percentile*M)], dtype=np.int64
    """
    if top_fraction > 1:
        top_fraction = top_fraction / 100
    m = fake_features.shape[0]
    top_n = int(m * top_fraction)
    realism_vector = compute_realism(real_features, fake_features, nearest_k)
    top_indices = np.argpartition(realism_vector, top_n, axis=axis)[..., -top_n:]
    return top_indices


@tf.function
def kl_step(batch):
    kl = batch * (tf.math.log(batch) - tf.math.log(tf.reduce_mean(batch, axis=0, keepdims=True)))
    kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
    score = tf.exp(kl)
    return score


def compute_kl(predictions, n_splits=10):
    predictions = np.concatenate(predictions, axis=0)
    bs = int(tf.math.ceil(predictions.shape[0] / n_splits))
    ds = tf.data.Dataset.from_tensor_slices(predictions)
    ds = ds.batch(bs).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    scores = []
    for batch in ds:
        scores.append(kl_step(batch))
    return np.mean(scores), np.std(scores)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError("Imaginary component {}".format(m))
            print('There is an Imaginary component {} that is being ignored'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):
    """Calculation of the statistics used by the FID.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def compute_fid(gen_activations, mu_real, sigma_real):
    mu_gen, sigma_gen = calculate_activation_statistics(gen_activations) 
    fid = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    return fid

class Ema:

    def __init__(self, beta):
        self.beta = tf.cast(beta, dtype=tf.float32)
        self.shadow = {}
        self.backup = {}

    # @tf.function
    def register(self, model, var_dict):
        for var in model.trainable_variables:
            # if ops.executing_eagerly_outside_functions():
            init_value = var.read_value()
            # else:
            # init_value = var.initialized_value()
            name = var.name
            # Copying variable from the original model
            var_dict[name] = tf.Variable(init_value, name=name, trainable=False)

    @tf.function
    def __call__(self, model):
        for var in model.trainable_variables:
            name = var.name
            assert name in self.shadow
            assign_moving_average(self.shadow[name], var, self.beta, False)

    @tf.function
    def dict_to_model(self, model, dict):
        for var in model.trainable_variables:
            name = var.name
            assert name in dict
            var.assign(dict[name])


@tf.function
def load_ema_weights(model):
    for var in model.trainable_variables:
        var.assign(model.ema.average(var))


def backup_weights(model, backup_dict):
    for var in model.trainable_variables:
        value = var.read_value()
        name = var.name
        backup_dict[name] = tf.Variable(value, name=name, trainable=False)


@tf.function
def restore_non_ema_weights(model, backup):
    for var in model.trainable_variables:
        name = var.name
        assert name in backup
        var.assign(backup[name])
        

if __name__ == "__main__":
    # u = np.ones([10, 10])
    # v = np.zeros([8, 10])
    d = np.random.normal(2, 0.1, size=[100, 10])
    t = np.random.normal(1.9, 0.2, size=[1000, 10])

    # r = compute_realism(d, t, nearest_k=2)
    # f = realism_filter(d, t, 2, 50)
    # m1 = knn_precision_recall_features(u, v)
    m = compute_prdc(d, t, nearest_k=2)
