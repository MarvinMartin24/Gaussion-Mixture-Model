# Machine Learning 2
# Homework 2 - Gr01A

# MARVIN MARTIN
# MICHEL OMAR AFLAK

import numpy as np
import glob, os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Gaussian Mixture Model Class
class GMM:
    def __init__(self, X, J):
        self.X = X
        self.I, self.N = np.shape(X)[:2]
        self.mu = None
        self.sigma = None
        self.phi = None
        self.J = J

    def pdf(self, x, mu, sigma):
        num = np.exp(-0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu))
        den = np.sqrt(np.power(2 * np.pi, self.N) * np.linalg.det(sigma))
        val = num / den
        return val[0][0]

    def fit(self, epochs = 100, verbose=True):
        self.phi = np.ones(self.J) / self.J
        self.mu = np.random.rand(self.J, self.N, 1)
        self.sigma = np.array([np.identity(self.N) * np.var(self.X, axis=0) for j in range(self.J)])
        likelihood = np.zeros((self.I, self.J))

        for e in range(epochs):
            if verbose:
                print('Epoch %d/%d' %(e + 1, epochs), end="\r")

            # e-step
            for i in range(self.I):
                s = sum(self.phi[j] * self.pdf(self.X[i],  self.mu[j], self.sigma[j]) for j in range(self.J))
                for j in range(self.J):
                    likelihood[i, j] = self.phi[j] * self.pdf(self.X[i], self.mu[j], self.sigma[j]) / s

            # m-step
            for j in range(self.J):
                s = sum(likelihood[i, j] for i in range(self.I))
                self.mu[j] = sum(likelihood[i, j] * self.X[i] for i in range(self.I)) / s
                self.sigma[j] = sum(likelihood[i, j] * np.dot(self.X[i] - self.mu[j], (self.X[i] - self.mu[j]).T) for i in range(self.I)) / s
                self.phi[j] = s / I

            if gif == "y":
                self.save_plot(e)

    def predict_proba(self, x):
        return [
            self.pdf(x, self.mu[j], self.sigma[j]) * self.phi[j]
            for j in range(self.J)
        ]

    def predict_class(self, x):
        probs = self.predict_proba(x)
        idx = np.argmax(probs)
        return idx, probs[idx]

    def save_plot(self, e):
        if not os.path.exists(cwd + '/plots'):
            os.makedirs(cwd + '/plots')

        plt.ioff()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        t = np.linspace(-30, 20, 50)
        points = np.array([
            [x, y, max(gmm.pdf(np.reshape([x, y], (N, 1)), gmm.mu[j], gmm.sigma[j]) for j in range(J))]
            for x in t for y in t
        ])
        colors = points[:,2] / max(points[:,2])

        ax.scatter(X[:,0], X[:,1], c='black', marker='.')
        ax.scatter(points[:,0], points[:,1], points[:,2], c=colors, cmap='hsv', marker='.')
        plt.title(f'Distribution plot using GMM in 2D iteration {e}')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('pdf')
        ax.view_init(30, e + 3)
        plt.savefig(f'plots/{e}.png')
        plt.close(fig)

#### ---------------------------------------------------------------------- ####
# Create an animation
cwd = os.getcwd()
def make_gif():
    fp_in = cwd + "/plots/*"
    fp_out = cwd + "/animation.gif"
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime)]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True)
    print(fp_out)

print("Homework n°2 on GMM - M.MARTIN and M.AFLAK")
print('Do you want to generate an animation (might take few minutes)? (y/n):')
gif = input()

#### ---------------------------------------------------------------------- ####

# Variables and Object creations
I = 300     # samples per class
N = 2       # features -> could be 3 but for ploting purposes, we used 2
J = 2       # clusters
epoch = 75 # Fitting iterations

mu_true = np.random.randint(-20, 20, (J, N)) # Target mu
sigma_true = np.array([np.identity(N) * np.random.randint(1, 20, N) for j in range(J)]) # Target sigma

# Data generated automatically (Might have to run several time if clusters are to close)
X = np.vstack([
    np.reshape(np.random.multivariate_normal(m, s, I), (I, N, 1))
    for m, s in zip(mu_true, sigma_true)
])
np.random.shuffle(X)


# Initialize Gaussian Mixture Model
gmm = GMM(X, J)
gmm.fit(epoch)

# ---------------------- Results after fitting ---------------------------------
print('\nReal (mu, sigma)')
for j in range(J):
    print(j, '-', mu_true[j].reshape(N), sigma_true[j].diagonal())

print('\nFound (mu, sigma)')
for j in range(J):
    print(j, '-', gmm.mu[j].reshape(N), gmm.sigma[j].diagonal())

print('\nPredictions on random samples (5) from true distributions:')
for j in range(J):
    samples = np.reshape(np.random.multivariate_normal(mu_true[j], sigma_true[j], 5), (5, N, 1))
    predictions = [gmm.predict_class(sample)[0] for sample in samples]
    print(j, '-', predictions)

# Animation creation
if gif == "y":
    print("Animation created !")
    make_gif()
else:
    print("No animation created !")

# ---------------------- Plots -------------------------------------------------
# Last state
fig_final = plt.figure()
ax_final = fig_final.add_subplot(111, projection='3d')

t = np.linspace(-30, 20, 50)
points = np.array([
    [x, y, max(gmm.pdf(np.reshape([x, y], (N, 1)), gmm.mu[j], gmm.sigma[j]) for j in range(J))]
    for x in t for y in t
])
colors = points[:,2] / max(points[:,2])

ax_final.scatter(X[:,0], X[:,1], c='black', marker='.')
ax_final.scatter(points[:,0], points[:,1], points[:,2], c=colors, cmap='hsv', marker='.')
ax_final.set_xlabel('X1')
ax_final.set_ylabel('X2')
ax_final.set_zlabel('pdf')
plt.title('Distribution plot using GMM in 2D')
plt.show()
