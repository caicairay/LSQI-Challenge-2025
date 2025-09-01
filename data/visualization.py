import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
# Load data
data = pd.read_csv("./EstData.csv")
fig, axs = plt.subplots(6, 8, figsize= (20, 15), sharex = True, sharey=True)
for id, ax in enumerate(axs.flatten()):
    data_plot = data[data["ID"] == (id+1)]
    # Set style
    sns.lineplot(
        data=data_plot,
        x="TIME",
        y="DV",
        style="EVID",  # distinguish by DOSE
        hue="DVID",        # distinguish by ID
        markers=True,
        dashes=False,
        palette="tab10",
        ax=ax,
        legend='brief' if (id == 47 or id == 11) else False
    )

#plt.title("TIME vs DV by DOSE and ID")
plt.xlabel("TIME")
plt.ylabel("DV")
# plt.legend(title="DOSE / ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
#plt.show()
plt.savefig('test.png')