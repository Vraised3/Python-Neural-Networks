from numpy import array, random, dot, exp
# i or


def sig(Train_set_input, Train_set_output):
    return 1 / (1 + exp(-dot(Train_set_input, Train_set_output)))


def dif_sig(val):
    return (1 - val) * val


ip = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
op = array([[0, 1, 1, 0]]).T

random.seed(1)
Train_Weights = 2 * random.random((3, 1)) - 1

for i in range(10000):
    Output = sig(ip, Train_Weights)
    Err = op - Output
    Train_Weights += dot(ip.T, Err * dif_sig(Output))
print(sig(array([1, 0, 0]), Train_Weights))
