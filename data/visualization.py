import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
# Load data
data = pd.read_csv("./EstData.csv")

# CMT and DVID are redundant
print((data[data["CMT"]==3]["DVID"] ==2).all())
print((data[data["CMT"]==2]["DVID"] ==1).all())
print((data[data["CMT"]==1]["DVID"] ==0).all())
# MDV and EVID are redundant
print((data[data["MDV"]==1]["EVID"] ==1).all())
print((data[data["MDV"]==0]["EVID"] ==0).all())

# Plot
bw_list = []
for id in range(48):
    data_plot = data[data["ID"] == (id+1)]
    bw_list.append(data_plot["BW"].iloc[0])
bw_list = np.array(bw_list).reshape(4, 12)
sort_idx = np.argsort(bw_list, axis=1)
for i in range(4):
    sort_idx[i] += i * 12

id_list = sort_idx.flatten()
bw_list=bw_list.flatten()

fig, axs = plt.subplots(4, 12, figsize= (24, 8), sharex = True, sharey=True)
for id, ax in zip(id_list,axs.flatten()):
    # ax.set_ylim([0, 16.5])
    ax.text(0.1, 20.5, f"BW={bw_list[id]}")
    data_plot = data[data["ID"] == (id+1)]
    data_observe = data_plot[data_plot["EVID"] == 0]
    sns.lineplot(
        data=data_observe,
        x="TIME",
        y="DV",
        hue="DVID",        # distinguish by ID
        style="DVID",
        markers=True,
        dashes=False,
        palette="tab10",
        ax=ax,
        # legend='brief' if (id == 47 or id == 11) else False
        legend=False,
    )
    data_dose = data_plot[data_plot["EVID"] == 1]
    sns.lineplot(
        data=data_dose,
        x="TIME",
        y="AMT",
        markers=True,
        dashes=False,
        # palette="tab10",
        ax=ax,
        # legend='brief' if (id == 47 or id == 11) else False
        legend=False
    )
    if data_plot["COMED"].all() == 1:
        ax.tick_params(color='green', labelcolor='green')
        for spine in ax.spines.values():
            spine.set_edgecolor('green')

# Sorting
# fig2, axs2 = plt.subplots(4, 12, figsize= (24, 8), sharex = True, sharey=True)
# bw_list = np.array(bw_list).reshape(4, 12)
# for i in range(4):
#    axs_sub = axs[i]
#    sub_bw_list = bw_list[i]
#    sorted_indices = np.argsort(sub_bw_list)
#    print(sorted_indices)
#    axs_sub = axs_sub[sorted_indices]
#    axs2[i] = axs_sub

plt.xlabel("TIME")
plt.ylabel("DV")
plt.tight_layout()
plt.savefig('test.png')