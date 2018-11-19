import numpy as np
from io_data import read_data, write_data
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import pandas as pd


def maximization(image, size, mean_sum0, mean_sum1, size0, size1, respons0_list, respons1_list):
    mu_new0 = mean_sum0 / size0
    mu_new1 = mean_sum1 / size1

    var_0 = np.zeros((3, 3))
    var_1 = np.zeros((3, 3))

    i = 0
    for y in image:
        for x in y:
            var_0 += respons0_list[i] * np.outer((x - mu_new0), (x - mu_new0))
            var_1 += respons1_list[i] * np.outer((x - mu_new1), (x - mu_new1))
            i += 1

    var_new0 = var_0 / size0
    var_new1 = var_1 / size1

    mix_new0 = size0 / size
    mix_new1 = size1 / size

    return mu_new0, var_new0, mix_new0, mu_new1, var_new1, mix_new1


def expectation(image, centers, cov0, cov1, mix0, eps=0.5):
    mean0, mean1 = centers[:2]  # Mean 1

    mix_coef0, mix_coef1 = mix0, 1 - mix0

    # Total number of samples
    size = image.shape[0] * image.shape[1]

    log_likelihoods = []
    iteration_number = 0

    while True:

        iteration_number += 1
        print("iteration_" + str(iteration_number) + ":")

        size0, size1 = 0, 0
        respons0_list, respons1_list = [], []
        mean_sum0, mean_sum1 = [0, 0, 0], [0, 0, 0]

        for rows in image:
            for x in rows:
                # Get gaussian density
                prob0 = multivariate_normal.pdf(x, mean=mean0, cov=cov0, allow_singular=True)
                prob1 = multivariate_normal.pdf(x, mean=mean1, cov=cov1, allow_singular=True)

                Numerator0 = mix_coef0 * prob0
                Numerator1 = mix_coef1 * prob1

                denom = Numerator0 + Numerator1

                resp0 = Numerator0 / denom
                resp1 = Numerator1 / denom

                respons0_list.append(resp0)
                respons1_list.append(resp1)

                mean_sum0 += resp0 * x
                mean_sum1 += resp1 * x

                size0 += resp0
                size1 += resp1

        # maximization
        # update means, covariances and co-efficients
        mu_new0, var_new0, mix_new0, mu_new1, var_new1, mix_new1 = maximization(
            image, size, mean_sum0, mean_sum1, size0, size1, respons0_list, respons1_list)

        mean0, mean1 = mu_new0, mu_new1
        cov0, cov1 = var_new0, var_new1
        mix_coef0, mix_coef1 = mix_new0, mix_new1

        # Calculate Log Likelihood
        ll = 0
        sumList = []
        for rows in image:
            for x in rows:
                prob0 = multivariate_normal.pdf(x, mu_new0, var_new0, allow_singular=True)
                prob1 = multivariate_normal.pdf(x, mu_new1, var_new1, allow_singular=True)

                sum = (mix_new0 * prob0) + (mix_new1 * prob1)
                sumList.append(np.log(sum))

            ll = np.sum(np.asarray(sumList))

        log_likelihoods.append(ll)

        print("Log Likelihood: " + str(ll))

        if len(log_likelihoods) < 2: continue
        # exit loop if log likelihoods dont change much for twice
        if np.abs(ll - log_likelihoods[-2]) < eps: break

    return mix_coef0, mix_coef1, mean0, mean1, cov0, cov1


def fit_EM(image, data, centers, cov0, cov1, mix0, eps=0.5):
    mix_coef0, mix_coef1, mean0, mean1, cov0, cov1 = expectation(image, centers, cov0, cov1, mix0, eps)

    back_data = data.copy()
    front_data = data.copy()
    mask_data = data.copy()

    for i in range(0, len(data) - 1):
        cell = data[i]
        point = [cell[2], cell[3], cell[4]]
        prob0 = multivariate_normal.pdf(point, mean=mean0, cov=cov0, allow_singular=True)

        resp0 = mix_coef0 * prob0
        prob1 = multivariate_normal.pdf(point, mean=mean1, cov=cov1, allow_singular=True)
        resp1 = mix_coef1 * prob1

        resp0 = resp0 / (resp0 + resp1)
        resp1 = resp1 / (resp0 + resp1)

        if resp0 < resp1:
            back_data[i][2] = back_data[i][3] = back_data[i][4] = 0
            mask_data[i][2] = mask_data[i][3] = mask_data[i][4] = 0

        else:
            front_data[i][2] = front_data[i][3] = front_data[i][4] = 0
            mask_data[i][2] = 100
            mask_data[i][3] = mask_data[i][4] = 0

    return back_data, front_data, mask_data


def output(data_dict, filename, prefix='outputs/a2/'):
    for type, data in data_dict.items():
        write_data(data, prefix + type + '_' + filename)
        read_data(prefix + type + '_' + filename, False, save=True,
                  save_name=prefix + type + '_' + filename.replace('.txt', '.jpg'))


def main():
    prefix = 'a2/'
    for filename in ['cow.txt','fox.txt','owl.txt','zebra.txt']:
    # for filename in ['owl.txt']:
        print('Solving ' + filename + ':...')
        data, image = read_data(prefix + filename, True)
        eps=0.5

        if(filename == 'owl.txt'):
            mix0 = 0.4
            means = np.array([[55, 0, 20], [80, 0, 0]])
            cov0 = np.array([[20, 0, 0], [0, 2, 0], [0, 0,10]])
            cov1 = np.array([[42, 0, 0], [0, 1, 0], [0, 0, 10]])
        else:
            X = np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

            # set covs, coefficients
            cluster_map = pd.DataFrame()
            cluster_map['data'] = X.tolist()
            cluster_map['cluster'] = kmeans.labels_
            cluster0 = cluster_map[cluster_map.cluster == 0]['data']
            cluster1 = cluster_map[cluster_map.cluster == 1]['data']
            cov0 = np.cov(np.transpose(cluster0.tolist()))
            cov1 = np.cov(np.transpose(cluster1.tolist()))
            mix0 = float(len(cluster0)) / len(cluster_map)
            means = kmeans.cluster_centers_

        back_data, front_data, mask_data = fit_EM(image, data, means, cov0, cov1, mix0, eps=eps)
        data_dict = {'back': back_data, 'front': front_data, 'mask': mask_data}
        output(data_dict, filename)


main()


