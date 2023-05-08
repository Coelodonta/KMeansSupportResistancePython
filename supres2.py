import numpy as np
from scipy.signal import argrelextrema
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('TkAgg')
import warnings
warnings.filterwarnings("ignore")

def find_optimal_clusters_silhouette(data, min_clusters, max_clusters, method):
    cluster_range = range(min_clusters, max_clusters)

    if method == "agglomerative":
        clustering_function = AgglomerativeClustering
    elif method == "kmeans":
        clustering_function = KMeans
    else:
        raise ValueError("Invalid clustering method. Choose either 'agglomerative' or 'kmeans'.")

    silhouette_scores = [silhouette_score(data, clustering_function(n_clusters=i).fit(data).labels_) for i in cluster_range]

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    return optimal_clusters

def find_peaks_troughs(data):
    highs = data["High"].values
    lows = data["Low"].values

    peak_indices = argrelextrema(highs, np.greater)
    trough_indices = argrelextrema(lows, np.less)

    peaks = highs[peak_indices]
    troughs = lows[trough_indices]

    return peaks, troughs, peak_indices, trough_indices

def find_peaks_higher_than_price(price,peak_labels,peaks):
    unique_labels=set(peak_labels)
    labels=[]
    for label in unique_labels:
        if np.min(peaks[peak_labels == label]) >= price:
            labels.append(label)
    return labels

def find_troughs_lower_than_price(price,trough_labels,troughs):
    unique_labels=set(trough_labels)
    labels=[]
    for label in unique_labels:
        if np.max(troughs[trough_labels == label]) <= price:
            labels.append(label)
    return labels


def plot_supres(df, title, filename, min_clusters=5, show_all=False,show_latest=True,show_closest=True):
    """
    Draw Support/Resistance levels and zones
    :param df: Pandas data frame with labels High, Low, Close, Volume and indexed on Date
    :param title: Title of graph
    :param filename: Optional file name for saving the images. If it's None the images are displayed instead
    :param min_clusters: Minimum number of clusters.
    :param show_all: If set to True all historical S/R levels will be plotted. If False only the S/R zones closest to
                     current price and/or the most current S/R zones will be plotted.
    :param show_latest: If True (and show_all is False) plot the most recent S/R zones.
    :param show_closest: If True (and show_all is False) plot the S/R zones closely surrounding the current price.
    :return: Nothing
    """
    # Find peaks and troughs in data
    peaks, troughs, peak_indices, trough_indices = find_peaks_troughs(df)
    peaks, troughs = np.reshape(peaks, (-1, 1)), np.reshape(troughs, (-1, 1))

    # Determine the number of clusters from the peaks and throughs
    max_clusters_peaks = len(peaks)
    max_clusters_troughs = len(troughs)

    # Find the optimal number of clusters for peaks and troughs
    optimal_clusters_peaks = find_optimal_clusters_silhouette(peaks, min_clusters,
                                max_clusters=max_clusters_peaks, method='kmeans')

    optimal_clusters_troughs = find_optimal_clusters_silhouette(troughs, min_clusters=min_clusters,
                                max_clusters=max_clusters_troughs, method='kmeans')

    ac_peaks = KMeans(n_clusters=optimal_clusters_peaks)
    ac_troughs = KMeans(n_clusters=optimal_clusters_troughs)

    peak_labels = ac_peaks.fit_predict(peaks)
    trough_labels = ac_troughs.fit_predict(troughs)

    gap_multiplier = 0.020  # Adjust this value to control the gap size
    price_range = df["High"].max() - df["Low"].min()

    # Plot the stock data with support and resistance levels
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1]})

    # Plot the price data with support and resistance levels
    ax1.plot(df.index, df["High"], label="High")
    ax1.plot(df.index, df["Low"], label="Low")
    ax1.plot(df.index, df["Close"], label="Close")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    if not show_all:
        # Show only most recent
        if show_latest:
            graph_type = "Latest zones"

            if len(peak_labels) > 0:
                # Determine which cluster the last peak belongs to
                peak_cluster_label = peak_labels[-1]
                # Find the max high of the last peak cluster group
                max_high_peak_cluster = np.max(peaks[peak_labels == peak_cluster_label])
                min_high_peak_cluster = np.min(peaks[peak_labels == peak_cluster_label])
                # Plot the support and resistance levels
                ax1.hlines(max_high_peak_cluster, df.index.min(), df.index.max(), colors="red",linestyles="dashed",linewidth=0.5, )
                ax1.hlines(min_high_peak_cluster, df.index.min(), df.index.max(), colors="red",linestyles="dashed",linewidth=0.5, )
                ax1.fill_between(df.index, min_high_peak_cluster, max_high_peak_cluster, color="red", alpha=0.25)
                ax1.text(df.index[0],min_high_peak_cluster - (1.0 * gap_multiplier) * price_range,
                    f"Latest Resistance: {min_high_peak_cluster:.4f} - {max_high_peak_cluster:.4f}",color="black",fontsize=10,)

            if len(trough_labels) > 0:
                # Determine which cluster the last trough belongs to
                trough_cluster_label = trough_labels[-1]
                # Find the min low of the last trough cluster group
                min_low_trough_cluster = np.min(troughs[trough_labels == trough_cluster_label])
                max_low_trough_cluster = np.max(troughs[trough_labels == trough_cluster_label])

                # Plot the support and resistance levels
                ax1.hlines(min_low_trough_cluster, df.index.min(), df.index.max(),colors="green",linestyles="dashed",linewidth=0.5,)
                ax1.hlines(max_low_trough_cluster, df.index.min(), df.index.max(),colors="green",linestyles="dashed",linewidth=0.5,)
                ax1.fill_between(df.index, min_low_trough_cluster, max_low_trough_cluster, color="green", alpha=0.25)
                ax1.text(df.index[0],max_low_trough_cluster + (1.5 * gap_multiplier) * price_range,
                         f"Latest support: {max_low_trough_cluster:.4f} - {min_low_trough_cluster:.4f}",color="black",fontsize=10)

        if show_closest:
            graph_type = "Zones surrounding last price"
            # Determine the clusters closest to teh current price
            price = df.iloc[-1]['Close']

            u_peak_labels = find_peaks_higher_than_price(price=price,peak_labels=peak_labels,peaks=peaks)
            if len(u_peak_labels) >0:
                closest_label = u_peak_labels[0]
                val = np.min(peaks[peak_labels == closest_label])
                for label in u_peak_labels:
                    if np.min(peaks[peak_labels == label]) < val:
                        closest_label = label
                        val = np.min(peaks[peak_labels == closest_label])
                max_high_peak_cluster = np.max(peaks[peak_labels == closest_label])
                min_high_peak_cluster = np.min(peaks[peak_labels == closest_label])
                # Plot the support and resistance levels
                ax1.hlines(max_high_peak_cluster, df.index.min(), df.index.max(), colors="red",linestyles="dashed",linewidth=0.5, )
                ax1.hlines(min_high_peak_cluster, df.index.min(), df.index.max(), colors="red",linestyles="dashed",linewidth=0.5, )
                ax1.fill_between(df.index, min_high_peak_cluster, max_high_peak_cluster, color="red", alpha=0.25)
                ax1.text(df.index[0],min_high_peak_cluster - (1.0 * gap_multiplier) * price_range,
                    f"Resistance closest to price: {min_high_peak_cluster:.4f} - {max_high_peak_cluster:.4f}",color="black",fontsize=10,)


            u_trough_labels = find_troughs_lower_than_price(price=price,trough_labels=trough_labels, troughs=troughs)
            if len(u_trough_labels) > 0:
                closest_label = u_trough_labels[0]
                val = np.max(troughs[trough_labels == closest_label])
                for label in u_trough_labels:
                    if np.max(troughs[trough_labels == label]) > val:
                        closest_label = label
                        val = np.max(troughs[trough_labels == closest_label])

                min_low_trough_cluster = np.min(troughs[trough_labels == closest_label])
                max_low_trough_cluster = np.max(troughs[trough_labels == closest_label])
                ax1.hlines(min_low_trough_cluster, df.index.min(), df.index.max(),colors="green",linestyles="dashed",linewidth=0.5,)
                ax1.hlines(max_low_trough_cluster, df.index.min(), df.index.max(),colors="green",linestyles="dashed",linewidth=0.5,)
                ax1.fill_between(df.index, min_low_trough_cluster, max_low_trough_cluster, color="green", alpha=0.25)
                ax1.text(df.index[0],max_low_trough_cluster + (1.5 * gap_multiplier) * price_range,
                         f"Support closest to price: {max_low_trough_cluster:.4f} - {min_low_trough_cluster:.4f}",color="black",fontsize=10)

    else:
        graph_type = "Historic levels"
        # Show a dotted line for all the support levels lower than price and resistance levels higher than price
        price = df.iloc[-1]['Close']

        u_peak_labels = find_peaks_higher_than_price(price=price,peak_labels=peak_labels,peaks=peaks)
        u_trough_labels = find_troughs_lower_than_price(price=price,trough_labels=trough_labels, troughs=troughs)

        for label in u_peak_labels:
            min_high_peak_cluster = np.min(peaks[peak_labels == label])
            ax1.hlines(min_high_peak_cluster, df.index.min(), df.index.max(), colors="red", linestyles="dashed",
                       linewidth=1, )
            ax1.text(df.index[0],min_high_peak_cluster - (1.5 * gap_multiplier) * price_range,
                     f"Resistance: {min_high_peak_cluster:.4f}",color="black",fontsize=10)

        for label in u_trough_labels:
            max_low_trough_cluster = np.max(troughs[trough_labels == label])
            ax1.hlines(max_low_trough_cluster, df.index.min(), df.index.max(), colors="green", linestyles="dashed",
                       linewidth=1, )
            ax1.text(df.index[0],max_low_trough_cluster + (1.0 * gap_multiplier) * price_range,
                     f"Support: {max_low_trough_cluster:.4f}",color="black",fontsize=10)


    ax1.set_title(f"{title} - {graph_type}")

    # Plot the volume data
    volume = df["Volume"]
    volume_norm = volume / volume.max()
    ax2.vlines(df.index, 0, volume_norm, color="black", alpha=0.5)
    ax2.fill_between(df.index, volume_norm, 0, color="blue", alpha=0.25)
    ax2.set_ylim([0, 1])
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.subplots_adjust(hspace=0.25)
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.close()