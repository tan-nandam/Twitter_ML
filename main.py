import random as rnd
import re
import math
import pandas as pd
import matplotlib.pyplot as plt


def load_tweets(file_path):
    file = open(file_path, "r", encoding="utf8")
    data = list(file)
    tweets = []

    for i in range(len(data)):
        data[i] = data[i].strip('\n')
        data[i] = data[i][50:]
        data[i] = " ".join(filter(lambda x: x[0] != '@', data[i].split()))
        data[i] = re.sub(r'http\S+|www\S+|https\S+', '', data[i], flags=re.MULTILINE)
        data[i] = data[i].replace('#', '')
        data[i] = data[i].lower()
        data[i] = re.sub(r'[^\w\s]', '', data[i])
        data[i] = " ".join(data[i].split())
        tweets.append(data[i].split(' '))

    file.close()
    return tweets


def initialize_random_centroids(tweets, k):
    centroids = []
    random_indices = set()
    for i in range(k):
        idx = rnd.randint(0, len(tweets) - 1)
        if idx not in random_indices:
            random_indices.add(idx)
            centroids.append(tweets[idx])
    return centroids


def calculate_jaccard_distance(data, centroid):
    numerator = list(set(data) & set(centroid))
    denominator = list(set(data) | set(centroid))
    distance = 1 - (len(numerator) / len(denominator))
    return distance


def assign_tweet_labels(tweets, centroids, k):
    labels = [[] for _ in range(k)]

    for tweet in tweets:
        distances = []
        for centroid in centroids:
            distances.append(calculate_jaccard_distance(tweet, centroid))
        if min(distances) == 1:
            label_idx = rnd.randint(0, len(centroids) - 1)
        else:
            label_idx = distances.index(min(distances))
        labels[label_idx].append([tweet, min(distances)])

    return labels


def new_centroids(labels):
    updated_centroids = []
    idx=-1
    for i in range(len(labels)):
        distance_matrix = []
        for m in range(len(labels[i])):
            distance_matrix.append([])
            for n in range(len(labels[i])):
                if m == n:
                    distance_matrix[m].append(0)
                else:
                    if m <= n:
                        distance = calculate_jaccard_distance(labels[i][m][0], labels[i][n][0])
                    else:
                        distance = distance_matrix[n][m]
                    distance_matrix[m].append(distance)
        if (distance_matrix):
            idx = distance_matrix.index(min(distance_matrix))
            updated_centroids.append(labels[i][idx][0])
        else:
            updated_centroids.append(labels[i][m][0])

    return updated_centroids


def perform_k_means(tweets, k):
    centroids = initialize_random_centroids(tweets, k)
    labels = assign_tweet_labels(tweets, centroids, k)
    old_centroids = []

    for iteration in range(k):
        labels = assign_tweet_labels(tweets, centroids, k)
        old_centroids = centroids
        centroids = new_centroids(labels)

    return labels, centroids


def elbow_plot(k_values, error_val):
    # Elbow Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_val, marker='o')
    plt.title('Elbow Plot for K-Means Clustering')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.show()


def distribution_plot(k_values, labels_sizes_all):
    for i, k in enumerate(k_values):
        plt.figure(figsize=(20, 20))
        plt.bar(range(1, k + 1), labels_sizes_all[i])
        plt.title(f'Cluster Size Distribution for K = {k}')
        plt.xlabel('Cluster Number')
        plt.ylabel('Number of Tweets in Cluster')
        plt.xticks(range(1, k + 1))
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    data_url = 'https://github.com/tan-nandam/Twitter_ML/blob/main/foxnewshealth.txt'
    tweets_data = load_tweets(data_url)

    k_values = [5, 10, 15, 20, 25, 30]
    output = []
    error_val = []
    labels_sizes_all = []
    for k in k_values:
        tweet_clusters, centroids = perform_k_means(tweets_data, k)
        sse = 0
        for i in range(len(tweet_clusters)):
            for j in range(len(tweet_clusters[i])):
                sse += pow(tweet_clusters[i][j][1], 2)
        error_val.append(sse)
        label_size = [len(cluster) for cluster in tweet_clusters]
        labels_sizes_all.append(label_size)
        output.append({'Value of K': k, 'Size of each cluster': ', '.join(
            f'{i + 1}: {size} tweets' for i, size in enumerate(label_size)), 'SSE': sse})
        print("k value:", k)

    df = pd.DataFrame(output)
    print(df.to_string())
    df.to_csv("output.csv")
    print("results written to the output.csv file")

    # plots
    elbow_plot(k_values, error_val)
    # distribution plot for each k value of a cluster
    distribution_plot(k_values, labels_sizes_all)
