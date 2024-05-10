import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

pca = LinearDiscriminantAnalysis(n_components=3)
#pca = PCA(n_components=3)
logits = torch.load('D:\\activations\\big\\metric\\penis_logits')
#lb_em = [(a[1],a[5]) for a in logits[0]]
lb_em = [(a[1],a[4]) for a in logits[1]]
lb = torch.concatenate([k[0].flatten(end_dim=-1) for k in lb_em])
em = torch.concatenate([k[1].flatten(end_dim=-2) for k in lb_em])
em  =em[torch.nonzero(torch.logical_and(torch.logical_not(lb == 0),torch.logical_not(lb ==1))).squeeze()].cpu().numpy()
lb = lb[torch.nonzero(torch.logical_and(torch.logical_not(lb == 0),torch.logical_not(lb ==1))).squeeze()].cpu().numpy()


empca = pca.fit_transform(em,lb)
print(davies_bouldin_score(empca,lb))
#empca = pca.fit_transform(em)
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def plot(embeds, labels, fig_path='./cock_metric.png'):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)
print
#plot(normalized(empca),lb)