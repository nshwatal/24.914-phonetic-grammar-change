from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


gen = np.random.default_rng()
VOWEL_LENGTH = 200
VOWEL_VOICE_DURATION = 175
DESIRED_DISTANCE = 80
p = 4
F0_VS_ASP = 15
MALE_LEN_VOT = 50
MALE_LEN_F0 = 4
NUM_EPOCHS = 100
TRIALS_PER_EPOCH = 50
VOT_BIAS = -20
VOT_STD = 10
F0_STD = 0.5
dx = 0.00001
DISTINCTIVENESS_WEIGHT = 10
LEARNING_RATE = 0.05
MLE_ITERATIONS = 100


def loss(vot, f0, w_vot, w_f0):
    return (
        w_f0 * (f0 - MALE_LEN_F0) ** 2
        + DISTINCTIVENESS_WEIGHT
        * (
            ((vot - MALE_LEN_VOT) ** 2 + p * (f0 - MALE_LEN_F0)) ** 0.5
            - DESIRED_DISTANCE
        )
        ** 2
        + w_vot * ((VOWEL_LENGTH - vot) - VOWEL_VOICE_DURATION) ** 2
    )


def loss_vec(x, w_vot, w_f0):
    return loss(x[0], x[1], w_vot, w_f0)

def gaussian(x, mean, std):
        return 1/(std*np.sqrt(2*np.pi)) * np.exp((-1/2)*((x-mean)/std)**2)

def make_likelihood_func(vot_0, f0_0):

    def likelihood_func(vot, f0, w_vot, w_f0):

        x0 = [vot_0, f0_0]
        weights = (w_vot, w_f0)

        ideal_vot, ideal_f0 = minimize(
        loss_vec,
        x0,
        method="nelder-mead",
        args=weights,
        options={"xatol": 1e-8, "disp": False},
        ).x

        return gaussian(vot, ideal_vot, VOT_STD) * gaussian(f0, ideal_f0, F0_STD)

    return likelihood_func

def make_w_vot_deriv(likelihood_func, w_vot, w_f0):

    def func(vot, f0):
        return (likelihood_func(vot, f0, w_vot+dx, w_f0) - likelihood_func(vot, f0, w_vot, w_f0))/dx

    return func

def make_w_f0_deriv(likelihood_func, w_vot, w_f0):

    def func(vot, f0):
        return (likelihood_func(vot, f0, w_vot, w_f0+dx) - likelihood_func(vot, f0, w_vot, w_f0))/dx

    return func

vots = []
f0s = []
w_vots = [1.0]
w_f0s = [5.0]



weights = np.asarray([1.0, 5.0])
x0 = [120, 4.54]
for epoch in range(NUM_EPOCHS):
    res = minimize(
        loss_vec,
        x0,
        method="nelder-mead",
        args=tuple(weights),
        options={"xatol": 1e-8, "disp": False},
    ).x

    current_vot, current_f0 = res
    vots.append(current_vot)
    f0s.append(current_f0)
    print(f"{current_vot= }, {current_f0= }")
    current_w_vot, current_w_f0 = weights

    likelihood_func = make_likelihood_func(current_vot, current_f0)

    for iteration in range(MLE_ITERATIONS):
        vot_deriv = make_w_vot_deriv(likelihood_func, current_w_vot, current_w_f0)
        f0_deriv = make_w_f0_deriv(likelihood_func, current_w_f0, current_w_f0)

        vot_list = gen.normal(current_vot, VOT_STD, TRIALS_PER_EPOCH)
        f0_list = gen.normal(current_f0, F0_STD, TRIALS_PER_EPOCH)
        biased_vot = vot_list + VOT_BIAS

        vot_update_list = vot_deriv(biased_vot, f0_list)
        f0_update_list = f0_deriv(biased_vot, f0_list)

        vot_update_mean = np.mean(vot_update_list)
        f0_update_mean = np.mean(f0_update_list)

        current_w_vot += LEARNING_RATE * vot_update_mean
        current_w_f0 += LEARNING_RATE * f0_update_mean

    print(current_w_vot, current_w_f0)
    w_vots.append(current_w_vot)
    w_f0s.append(current_w_f0)
    weights = [current_w_vot, current_w_f0]

vots_d = pd.DataFrame(vots)
vots_d.to_excel("vot_data.xlsx")
f0s_d = pd.DataFrame(f0s)
f0s_d.to_excel("f0_data.xlsx")
plt.plot(vots)
plt.show()
plt.plot(f0s)
plt.show()
plt.plot(w_vots)
plt.show()
plt.plot(f0s)
plt.show()


# x0 = np.array([105, 5])
# min_dist = None
# min_vec = None
# for w_f0 in range(1, 10):
#     for w_d in range(10, 11):
#         for w_f0_asp in range(1, 10):
#                 res = minimize(loss_vec, x0, method='nelder-mead',
#                     args=(w_f0, w_d, w_f0_asp), options={'xatol': 1e-8, 'disp': False})

#                 if min_dist is None or np.linalg.norm(res.x-x0) < min_dist:
#                     min_vec = res.x
#                     min_dist = np.linalg.norm(res.x-min_vec)
#                     print(res.x)
#                     print(w_f0, w_d, w_f0_asp)

# print(min_vec)

