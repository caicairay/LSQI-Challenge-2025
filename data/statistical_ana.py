import pandas as pd
import matplotlib.pyplot as plt


def extract_observed_data(data):
    observed_data = data[data["EVID"] == 0]
    return observed_data


def extract_dosing_data(data):
    dosing_data = data[data["EVID"] == 1]
    return dosing_data


def extract_biomarker_data(data):
    biomarker_data = data[data["DVID"] == 2]
    return biomarker_data


def extract_compound_data(data):
    compound_data = data[data["DVID"] == 1]
    return compound_data


def print_statistical_analysis(data):
    # Statistical analysis (all dataset)
    for col in data.columns:
        print(f"Column: {col}")
        print(data[col].value_counts(dropna=False))
        print("-" * 40)


if __name__ == "__main__":

    data = pd.read_csv(r"./EstData.csv")

    dosing_data = extract_dosing_data(data)
    observed_data = extract_observed_data(data)

    num_subjects = data["ID"].nunique()
    num_dosing = dosing_data["ID"].nunique()
    num_ref = num_subjects - num_dosing

    observed_compounds = extract_compound_data(observed_data)
    observed_biomarker = extract_biomarker_data(observed_data)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)
    plot_dv_offset = 1

    for subject_id in range(1, num_subjects + 1):

        plot_biomarker = observed_biomarker.loc[
            observed_biomarker["ID"] == subject_id, :
        ]
        plot_compound = observed_compounds.loc[
            observed_compounds["ID"] == subject_id, :
        ]
        ax1.semilogx(
            plot_biomarker["TIME"],
            plot_biomarker["DV"],  # + plot_dv_offset * subject_id,
            color=str(subject_id / num_subjects * 0.5),
            label=f"Subject {subject_id}",
            marker="o",
        )

        if plot_compound.shape[0] == 0:
            continue

        ax2.semilogx(
            plot_compound["TIME"],
            plot_compound["DV"],  # + plot_dv_offset * subject_id,
            color=str(subject_id / num_subjects * 0.5),
            label=f"Subject {subject_id}",
            marker="o",
        )
    ax1.set_title("Observed Biomarker Data")
    ax2.set_title("Observed Compound Data")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("DV")
    ax2.set_xlabel("Time")

    fig.savefig("observed_data_plots.png")
    plt.show()
