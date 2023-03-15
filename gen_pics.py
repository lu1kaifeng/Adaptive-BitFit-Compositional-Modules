import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.pyplot import pcolormesh

coe_list = []
for n in ['bn','cts','nw','un','wl']:
    adapter_list = np.load(os.path.join('model_bitfit//bert//bert//lll//1_bc_bn_cts_nw_un_wl_0.05_42//'+n, "adapter_list.npy"), allow_pickle=True)
    adapter_list = adapter_list.item()['mix_coe']
    adapter_list = list(map(lambda x:(x[0],F.softmax(x[1]).detach().cpu()),list(adapter_list.items())))


    title = list(map(lambda x:x[0],adapter_list))

    mat = dict()
    for i in range(24):
        r = re.compile(".*layer\."+str(i)+"\.")
        newlist = list(filter(r.match, title))  # Read Note below
        mat[i] = newlist
    mat_tensor = np.zeros([24,8,6])
    adapter_dict = {k:v for k,v in adapter_list}
    for i in range(24):
        for j in range(8):
            ten = adapter_dict[mat[i][j]]
            mat_tensor[i][j] = np.pad(ten.numpy(),(0,mat_tensor[i][j].shape[0] - ten.shape[0]),'constant')
    coe_list.append(mat_tensor)

import matplotlib as mpl
from matplotlib import pyplot
cl_list = []
titles = ['bn','cts','nw','un','wl']
i=0
for i,c in enumerate(coe_list):
    cl = np.squeeze(coe_list[i][:,:,i+1])
    cl_list.append(cl)
    from matplotlib import colors

    divnorm = colors.TwoSlopeNorm(vmin=0.8/(i+2), vcenter=1/(i+2), vmax = 1.5/(i+2))
    pcolormesh(cl, cmap="coolwarm", norm=divnorm)

    # tell imshow about color map so that only set colors are used
    #img = pyplot.imshow(cl, interpolation='nearest',origin='lower')

    # make a color bar
    #pyplot.clim(0, 1/(i+2))
    pyplot.title(titles[i])
    i=i+1
    pyplot.colorbar()
    pyplot.xticks(np.arange(8),['query', 'key', 'value', 'output.dense', 'output.LayerNorm', 'intermediate.dense', 'output.dense', 'output.LayerNorm']
                  ,rotation=30)
    pyplot.subplots_adjust(bottom=0.25)
    pyplot.show()
