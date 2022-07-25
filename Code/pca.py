import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt



def apply_pca(pca, data, correlation):

    # Automatic selection of the number of principal components based on the elbow rule
    if pca == "elbow":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        pca = PCA()
        pca.fit_transform(data)
        list_pca = np.array(pca.explained_variance_ratio_)

        scaler = MinMaxScaler(feature_range=(0, len(list_pca)))
        a_scaled = scaler.fit_transform(np.array(list_pca).reshape(-1, 1))
        a_scaled = np.array(a_scaled).reshape(len(a_scaled), )

        c = np.diff(a_scaled)
        c = abs(np.array(c))

        # PCA explained variance plot
        # plt.plot(range(len(a_scaled)), a_scaled)
        # plt.show()

        for n, value in enumerate(c):
            if value <= 1:
                pca_index = n
                break

        print(f"Optimal no. of principal components: {pca_index}")
        print()
        pca = PCA(pca_index)
        data_pca = pca.fit_transform(data)

        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))


        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        input_data = data_pca



    # No pca
    elif pca == "none":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        kk_max = []
        kk_min = []
        for kk in data:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        data = np.array(data).reshape((len(correlation), len(correlation[0]) - 1, data.shape[1]))

        input_data = data



    # PCA using default PCA parameters (All components are kept)
    elif pca == "auto":

        data = np.array(data).reshape(data.shape[0] * data.shape[1], data.shape[2])

        pca = PCA(2)
        data_pca = pca.fit_transform(data)
        data_pca = np.array(data_pca).reshape((len(correlation), len(correlation[0]) - 1, data_pca.shape[1]))

        kk_max = []
        kk_min = []
        for kk in data_pca:

            kk_min.append(np.min(kk))
            kk_max.append(np.max(kk))

        data_min = np.min(kk_min)
        data_max = np.min(kk_max)

        input_data = data_pca


    return input_data, data_max, data_min
