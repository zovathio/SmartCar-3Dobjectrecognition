import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("log.csv",delimiter =';')
print(df.head())
losses = ["val_loss","loss",]
accs = ["acc","val_acc"]
header = list(df)
REAL_EPOCH = len(df)
print(REAL_EPOCH)

plt.style.use("ggplot")
fig = plt.figure(figsize=(16, 8))
for (i, l) in enumerate(losses):
    #     plt.figs
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(np.arange(0, REAL_EPOCH), df[l], label=l)
    fig.legend()
    #     ax[1].set_ylim([-0.01,0.01])

    #     ax[0].set_ylim([0,1])
    plt.tight_layout()
    # save as figures
plt.savefig("images/loss.png")

fig = plt.figure(figsize=(16, 8))
for (i, l) in enumerate(accs):
    #     plt.figs
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.plot(np.arange(0, REAL_EPOCH), df[l], label=l)
    fig.legend()
    #     ax[1].set_ylim([-0.01,0.01])

    #     ax[0].set_ylim([0,1])
    plt.tight_layout()
    # save as figures
plt.savefig("images/acc.png")