import numpy as np
import cv2 # opencv: https://pypi.python.org/pypi/opencv-python


class Ising_model:

    def __init__(self, image, external, B):
        self.dict = {0: -1, 255: 1, -1: 0, 1: 255}
        self.image = np.vectorize(lambda x: self.dict[x])(image)
        self.width, self.height = self.image.shape[:2]
        self.ext_vector = external * self.image
        self.B = B

    def get_neighbors(self, x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        else:
            neighbors.append((self.width - 1, y))
        if x < self.width - 1:
            neighbors.append((x + 1, y))
        else:
            neighbors.append((0, y))
        if y > 0:
            neighbors.append((x, y - 1))
        else:
            neighbors.append((x, self.height - 1))
        if y < self.height - 1:
            neighbors.append((x, y + 1))
        else:
            neighbors.append((x, 0))
        return neighbors

    def localE(self, x, y):
        return self.ext_vector[x, y] + sum((self.image[sx, sy]) for (sx, sy) in self.get_neighbors(x, y))

    def gibbs_sampling(self, x, y):

        p = 1 / (1 + np.exp(-2 * self.B * self.localE(x, y)))
        if np.random.uniform(0, 1) <= p:
            self.image[x, y] = 1
        else:
            self.image[x, y] = -1


def ising_denosing(image, q=0.6, B=7, burn_in=8, T=18, p=0.7):
    external = 0.5 * np.log(q / (1 - q))
    ismodel = Ising_model(image, external, B)

    mean_img = np.zeros_like(ismodel.image).astype(np.float64)
    for i in range(burn_in + T):
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if (np.random.random() <= p):
                    ismodel.gibbs_sampling(x, y)
            if i > burn_in:
                mean_img += ismodel.image
    mean_img = mean_img/T
    mean_img[mean_img >= 0] = 255
    mean_img[mean_img < 0] = 0
    return mean_img


def main():
    params = {1:{'q': 0.65, 'B':4, 'burn_in':10, 'T':18, 'p':0.65},
              2:{'q': 0.6, 'B':7, 'burn_in':8, 'T':18, 'p':0.7},
              3:{'q': 0.65, 'B':4, 'burn_in':10, 'T':18, 'p':0.65},
              4:{'q': 0.65, 'B':4, 'burn_in':10, 'T':18, 'p':0.65}}

    filename = "a1/%d_noise.png"
    outputname = "outputs/a1/giibs_%d_denoise.png"

    for i in range(1,5):
        noise = cv2.imread(filename % i, 0)
        mean_img = ising_denosing(noise, q=params[i]['q'],
                                  B=params[i]['B'], burn_in=params[i]['burn_in'],
                                  T=params[i]['T'],p=params[i]['p'])
        cv2.imwrite(outputname % i, mean_img)

main()
