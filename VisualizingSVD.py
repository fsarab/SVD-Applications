import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'RdBu_r'

PRECISION = 3


def svd(M):
    """Returns the Singular Value Decomposition of M (via Numpy), with all
    components returned in matrix format
    """
    U, s, Vt = np.linalg.svd(M)

    # Put the vector singular values into a padded matrix
    S = np.zeros(M.shape)
    np.fill_diagonal(S, s)

    # Rounding for display
    return np.round(U, PRECISION), np.round(S, PRECISION), np.round(Vt.T, PRECISION)


def visualize_svd(m, n, fig_height=5):
    """Show the Singular Value Decomposition of a random matrix of size `m x n`

    Parameters
    ----------
    m : int
        The number of rows in the random matrix
    n : int
        The number of columns
    fig_height : float
        Fiddle parameter to make figures render better (because I'm lazy and
        don't want to work out the scaling arithmetic).
    """
    # Repeatability
    np.random.seed(123)

    # Generate random matrix
    M = np.random.randn(m, n)

    # Run SVD, as defined above
    U, S, V = svd(M)

    # Visualization
    fig, axs = plt.subplots(1, 7, figsize=(12, fig_height))

    plt.sca(axs[0])
    plt.imshow(M)
    plt.title(f'$M \\in \\mathbb^{m} \\times {n}$', fontsize=14)

    plt.sca(axs[1])
    plt.text(.25, .25, '=', fontsize=48)
    plt.axis('off')

    plt.sca(axs[2])
    plt.imshow(U)
    plt.title(f'$U \\in \\mathbb{R}^{m} \\times {m}$', fontsize=14)

    plt.sca(axs[3])
    plt.text(.25, .25, '$\\times$', fontsize=48)
    plt.axis('off')

    plt.sca(axs[4])
    plt.imshow(S)
    plt.title(f'$S \\in \\mathbb{R}^{m} \\times {n}$')

    plt.sca(axs[5])
    plt.text(0.25, .25, '$\\times$', fontsize=48)
    plt.axis('off')

    plt.sca(axs[6])
    cmap = plt.imshow(V.T)
    plt.colorbar(cmap, ax=axs, orientation='horizontal', aspect=50)
    plt.title(f'$V^T \\in \\mathbb{R}^{n} \\times {n}$', fontsize=14)

    plt.suptitle(f'SVD Components $m={m}, n={n}$', fontsize=18)

    fname = f'/tmp/svd-{m}x{n}.png'
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    print(fname)

visualize_svd(4, 4, fig_height=3)