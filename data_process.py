import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import preprocessing

def status(x, length, nominal=True):
    if nominal:
        return x.value_counts()
    else:
        return pd.Series([x.max(), x.min(), x.mean(), x.median(),
                    x.quantile(.25), x.quantile(.75), length-x.count()],
                    index = ['Max', 'Min', 'Mean', 'Median', 'Q1', 'Q3', 'Miss'])

def visualize(df, rows, cols):
    # 绘制直方图
    df.hist(figsize=(16, 8), layout=(rows,cols))
    # 绘制盒图
    df.plot(kind='box', subplots=True, layout=(rows,cols), sharex=False, sharey=False,
                    figsize=(16, 8))
    tPlots, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, 8))
    n = 0
    for col in df:
        stats.probplot(df[col], plot=axes[n//cols, n%cols])
        axes[n // cols, n % cols].set_title(col)
        n += 1
    plt.show()


def compare(df1, df2, bins=50):
    '直方图比较'
    for col in df1:
        mean1 = df1[col].mean()
        mean2 = df2[col].mean()

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        df1[col].hist(ax=ax1, grid=False, figsize=(15, 5), bins=bins)
        ax1.axvline(mean1, color='r')
        plt.title('origin\n{}\nmean={}'.format(col, str(mean1)))
        ax2 = fig.add_subplot(122)
        df2[col].hist(ax=ax2, grid=False, figsize=(15, 5), bins=bins)
        ax2.axvline(mean2, color='b')
        plt.title('filled\n{}\nmean={}'.format(col, str(mean2)))
        plt.subplots_adjust(wspace=0.3, hspace=10)
        plt.show()