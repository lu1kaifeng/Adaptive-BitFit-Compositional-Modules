import numpy as np
import os
import torch
from torch import functional as F
for n in ['bn','cts','nw','un','wl']:
    adapter_list = np.load(os.path.join('D:\\activations\\hybrid\\'+n, "vis.npy"), allow_pickle=True)
    k = adapter_list.item()['encoder.layer.6.attention.output.dense.parametrizations.bias.original1']
    p = torch.softmax(k[:,1:]*100 +torch.tile(k[:,0],(1024,1)).permute(1,0),dim=0).numpy().transpose()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    vmin, vmax = 0.0, 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)
    plot = plt.pcolormesh(p, norm=norm, cmap='binary')
    plt.colorbar(plot, norm=norm)
    plt.savefig("hybrid-"+n+'.png')


for n in ['bn','cts','nw','un','wl']:
    adapter_list = np.load(os.path.join('D:\\activations\\linear\\'+n, "vis.npy"), allow_pickle=True)
    k = adapter_list.item()['encoder.layer.6.attention.output.dense.parametrizations.bias.original1']
    p = torch.softmax(k[:,:],dim=0).numpy().transpose()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    vmin, vmax = 0.0, 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(10)
    plot = plt.pcolormesh(p, norm=norm, cmap='binary')
    plt.colorbar(plot, norm=norm)
    plt.savefig("linear-"+n+'.png')