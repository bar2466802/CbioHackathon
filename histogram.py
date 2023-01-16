import pandas as pd
import seaborn


def histogram(df_c, df_h):
    cancer = (df_c.iloc[2][1:].astype(float) * 100 / 999).reset_index(drop=True).to_frame()
    cancer["label"] = "cancer"
    healthy = (df_h.iloc[2][1:].astype(float) * 100 / 999).reset_index(drop=True).to_frame()
    healthy["label"] = "healthy"
    new_df = pd.concat([cancer, healthy])
    p = histplot(new_df, kde=True, hue="label", x=2, stat="count")
    p.set(xlabel="percent", ylabel="count")
    plt.show()