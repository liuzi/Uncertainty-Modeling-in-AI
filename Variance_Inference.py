import numpy as np
import cv2
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal


## sigma: noise, J:coupling strength
def variance_inference(noise_binary, sigma=1, J=1.2, smooth_rate=0.6, T=20):
    height, width = noise_binary.shape[:2]

    # y_i ~ N(x_i; sigma^2)
    y = noise_binary + sigma * np.random.randn(height, width)

    p_r = multivariate_normal.logpdf(y.flatten(), mean=+1, cov=sigma ** 2)
    p_l = multivariate_normal.logpdf(y.flatten(), mean=-1, cov=sigma ** 2)
    logodds = np.reshape(p_r - p_l, (height, width))

    mean = 2 * sigmoid(logodds) - 1

    log_r = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=+1, cov=sigma ** 2), (height, width))
    log_l = np.reshape(multivariate_normal.logpdf(y.flatten(), mean=-1, cov=sigma ** 2), (height, width))

    evidence_LB = np.zeros(T)

    for i in range(T):
        meanNew = mean
        for xi in range(width):
            for yi in range(height):
                pos = yi + height * xi
                neighborhood = pos + np.array([-1, 1, -height, height])
                boundary_idx = [yi != 0, yi != height - 1, xi != 0, xi != width - 1]
                neighborhood = neighborhood[np.where(boundary_idx)[0]]
                xx, yy = np.unravel_index(pos, (height, width), order='F')
                nx, ny = np.unravel_index(neighborhood, (height, width), order='F')

                Sbar = J * np.sum(mean[nx, ny])
                meanNew[xx, yy] = (1 - smooth_rate) * meanNew[xx, yy] + smooth_rate * np.tanh(
                    Sbar + 0.5 * logodds[xx, yy])
                evidence_LB[i] = evidence_LB[i] + 0.5 * (Sbar * meanNew[xx, yy])

        mean = meanNew

        update_mean = mean + 0.5 * logodds
        q_r, q_l = sigmoid(+2 * update_mean), sigmoid(-2 * update_mean)

        evidence_LB[i] = evidence_LB[i] + np.sum(q_r * log_r + q_l * log_l) + \
                         np.sum(-q_l * np.log(q_l + 1e-10) - q_r * np.log(q_r + 1e-10))  ## Entropy

    return mean


def main():

    filename = "a1/%d_noise.png"
    outputname = "outputs/a1/VI_%d_denoise.png"

    for i in range(1,5):
        noise = cv2.imread(filename % i, 0)
        noise_binary = +1 * (noise == 255) + -1 * (noise == 0)
        mean_img = variance_inference(noise_binary)
        denoise_pic = +255 * (mean_img > 0)
        cv2.imwrite(outputname % i, denoise_pic)


main()

