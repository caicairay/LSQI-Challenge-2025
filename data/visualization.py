import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("./EstData.csv")

# Filter for DOSE
# data = data[data["DOSE"] == 1]
# Filter for DVID
data = data[data["DVID"] == 1]
data = data[data["CMT"] == 2]

# Set style
sns.set(style="whitegrid")

# Create plot
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=data,
    x="TIME",
    y="DV",
    hue="DOSE",  # distinguish by DOSE
    # hue="ID",        # distinguish by ID
    markers=True,
    dashes=False,
    palette="tab10",
)

plt.title("TIME vs DV by DOSE and ID")
plt.xlabel("TIME")
plt.ylabel("DV")
plt.legend(title="DOSE / ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
