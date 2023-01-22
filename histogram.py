#################################################################
# FILE : histogram.py
# WRITERS :  Linoy Bushari, Yuval Gabbay, Noa Goldenberg, Bar Melinarskiy, and Yuval Roditi
# EXERCISE : Algorithms in Computational Biology - 76558 - Hackathon
# DESCRIPTION: histogram
#################################################################

import pandas as pd
from seaborn import histplot
import matplotlib.pyplot as plt


def histogram(df_c, df_h):
    for col_name in [2, 51, 23245, 13495, 14176, 8694, 14274]:
        cancer = (df_c.iloc[col_name][1:].astype(float) * 100 / 999).reset_index(drop=True).to_frame()
        cancer["subjects"] = "diagnosed"
        healthy = (df_h.iloc[col_name][1:].astype(float) * 100 / 999).reset_index(drop=True).to_frame()
        healthy["subjects"] = "healthy"
        new_df = pd.concat([cancer, healthy])
        p1 = histplot(new_df,
                      kde=True,
                      x=col_name,
                      hue="subjects",
                      common_norm=False,
                      stat="percent",
                      palette={'diagnosed': 'tab:red', 'healthy': 'tab:green'})
        p1.set(xlabel="methylation percent", ylabel="percent of subjects")
        plt.title('Histograms of ' + df_h.iloc[col_name, 0])
        print('saves fig ' + str(col_name))
        plt.savefig('histogram_plots/histogram_' + str(col_name))
        plt.clf()

if __name__ == '__main__':
    c_df = pd.read_table('DATA/BRCA_Primary_Tumor.chr19.tsv')
    h_df = pd.read_table('DATA/BRCA_Solid_Tissue_Normal.chr19.tsv')
    histogram(c_df, h_df)
