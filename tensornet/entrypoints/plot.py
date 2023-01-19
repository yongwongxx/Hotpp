import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_prop(prop):
    if "per_" in prop:
        x = np.load("target_{}.npy".format(prop[4:]))
        y = np.load("output_{}.npy".format(prop[4:]))
        n = np.load("n_atoms.npy")
        while len(n.shape) < len(x.shape):
            n = n[..., None]
        x = x / n
        y = y / n
    else:
        x = np.load("target_{}.npy".format(prop))
        y = np.load("output_{}.npy".format(prop))
    x = x.reshape(-1)
    y = y.reshape(-1)
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect(1)
    # titile
    plt.title("Miao {0} vs DFT {0}".format(prop), fontsize=16)
    # axis
    ymajorFormatter = ticker.FormatStrFormatter('%.1f') 
    xmajorFormatter = ticker.FormatStrFormatter('%.1f') 
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.set_xlabel('DFT  {}'.format(prop), fontsize=14)
    ax.set_ylabel('Miao {}'.format(prop), fontsize=14)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)    
    ax.tick_params(labelsize=16)
    # scatter
    ax.scatter(x, y)
    # diagonal line
    s = min(np.min(x), np.min(y))
    e = max(np.max(x), np.max(y))
    ax.plot([s, e], [s, e], color='black',linewidth=3,linestyle='--',)
    # rmse
    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    plt.text(0.85 * s + 0.15 * e,
             0.15 * s + 0.85 * e,
             "RMSE: {:.3f}\nMAE: {:.3f}".format(rmse, mae), fontsize=14)
    plt.savefig('{}.png'.format(prop))
    return None


def main(*args, properties=["per_energy", "forces"], **kwargs):
    for prop in properties:
        try:
            plot_prop(prop)
        except:
            print("Fail in plot {}".format(prop))


if __name__ == "__main__":
    main()
