import cantera as ct
import numpy as np
import json
import random
from rich.progress import track

m = 1000 # number of reactions
s_uniform = 100 # number of uniform sample points
s_norm = 0 # number of normal distribution points

T_input = []
P_input = []
Y_input = np.zeros((10000000,20)) # make it large enough
Y_label = np.zeros((10000000,20)) # make it large enough
Y = np.zeros(20)

y_index = 0
for i in track(range(m), description="gen data..."):
    # sample point distribution
    sample_uniform = random.sample(range(1,1000), s_uniform)
    sample_norm = np.random.normal(530, 20, s_norm)
    sample_norm = sample_norm.astype(int)
    sample_norm = sample_norm[sample_norm > 1]
    sample_norm = sample_norm[sample_norm < 1000]
    sample = np.sort(np.append(sample_norm, sample_uniform))

    # Data range
    T = np.random.uniform(1300, 1800)
    P = np.random.uniform(0.8*ct.one_atm, 1.1*ct.one_atm)
    Y[19] = np.random.uniform(0.713257, 0.73275983) # N2
    Y[3] = 1  # CH4
    Y[10] = np.random.uniform(0.2, 0.3) # CH4
    Y[0:19] *= (1-Y[19])/np.sum(Y[0:19])

    initial_TPY = T, P, Y

    gas = ct.Solution('drm19.yaml')
    gas.TPY = initial_TPY

    r = ct.IdealGasConstPressureReactor(gas, name='R1')
    sim = ct.ReactorNet([r])

    for tt in range(1, 1000):
        if tt in sample:
            T_input.append(np.float64(gas.T))
            P_input.append(np.float64(gas.P))
            Y_input[y_index] = np.float64(gas.Y)

        sim.advance(tt*1e-6)

        if tt in sample:
            Y_label[y_index] = np.float64(gas.Y)
            y_index += 1

# Additional data
T_input = np.array(T_input)
P_input = np.array(P_input)

new_len = T_input.shape[0]

input = np.zeros((new_len, 21))
label = np.zeros((new_len, 19))

input[:, 0] = T_input
input[:, 1] = P_input
input[:, 2:21] = np.maximum(Y_input[:new_len, :-1], 1e-40)

label[:, :19] = np.maximum(Y_label[:new_len, :-1], 1e-40)

# Original data
# data = np.load("./dataset.npy")

# part of original data
# num = 2**19
# rand_row = np.random.choice(data.shape[0], size=num,replace=False)
# data = data[rand_row]

# orig_input = np.zeros((data.shape[0],21))
# orig_label = np.zeros((data.shape[0],19))

# orig_input[:,:2]=data[:,:2]
# orig_input[:,2:21] = data[:,2:21]
# orig_label = data[:,44:63]

# Concatenated data
n = input.shape[0] #+ orig_input.shape[0]
print("data size = ", n)

# Transformation
lamda = 0.2
dt = 1e-6

bct_input = np.zeros((n,21))
bct_label = np.zeros((n,19))

bct_input[:, 0:2] = input[:, 0:2] # np.vstack((input[:, 0:2], orig_input[:, 0:2]))
bct_input[:,2:21] = (input[:,2:21]**lamda - 1)/lamda # (np.vstack((input[:,2:21], orig_input[:, 2:21]))**lamda - 1) / lamda 

bct_label_old = (label[:, :19]**lamda - 1)/lamda # (np.vstack((label[:, :19], orig_label[:, :19]))**lamda - 1) / lamda  
bct_label[:, :19] =  (bct_label_old - bct_input[:,2:21]) / dt

inputs_mean = np.mean(bct_input,axis=0, dtype=np.float64)
inputs_std = np.std(bct_input,axis=0, dtype=np.float64)

labels_mean = np.mean(bct_label,axis=0, dtype=np.float64)
labels_std = np.std(bct_label,axis=0, dtype=np.float64)

norm_inputs = (bct_input - inputs_mean) / inputs_std
norm_labels = (bct_label - labels_mean) / labels_std

data_path = "./norm-new.json"
normdata = {
    "dt": dt,
    "lambda": lamda,
    "inputs_mean": inputs_mean.tolist(),
    "inputs_std": inputs_std.tolist(),
    "labels_mean": labels_mean.tolist(),
    "labels_std": labels_std.tolist(),
}

with open(data_path, 'w') as json_file:
        json.dump(normdata,json_file,indent=4)

np.save('norm_inputs-new.npy',norm_inputs)
np.save('norm_labels-new.npy',norm_labels)

np.savetxt("input.txt", norm_inputs)
np.savetxt("label.txt", norm_labels)