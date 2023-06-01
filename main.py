import pandas as pd
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt; plt.rcParams['font.size'] = 16
import sys
from GA.GA import *
import pickle
import torch
from model.nn import NN
from model.physics import IDM
import os

# Model hyper
ALPHA = 0.1

# Training parameter
N_ITER = 500
Eval_num = 10
NUM_TRAIN = 1000

INPUT_DIM = 3
N_HIDDEN = 3
HIDDEN_DIM = 60
OUTPUT_DIM = 1

#CUDA support
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')

USE_GA = False

nn_args = (INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM)
nn_kwargs = {"activation_type": "sigmoid",
             "last_activation_type": "none",
             "device": DEVICE}

params_trainable =  {
    "v0": False,
    "T": False,
    "s0": False,
    "a": False,
    "b": False
    }

optimizer_kwargs = {
    "lr":0.001
}

optimizer_physics_kwargs = {
    "lr":0.1
}


if 1:
    np.random.seed(1234)
    torch.manual_seed(1234)
    RANDOM_SEED = 1234
else:
    np.random.seed(4321)
    torch.manual_seed(4321)
    RANDOM_SEED = 4321

def xvfl_to_feature(arr):
    # xvfl means "position (x) and velocity (v) of the follower (f) and the leader (l)"
    # that is, the columns stand for: 1. position of the follower; 2. velocity of the follower
    #                                 3. position of the leader;   4. velocity of the leader


    # dx
    dx = arr[:, 2] - arr[:, 0]
    dx = dx.reshape(-1, 1)
    dx = dx[:-1, :]

    # dv
    dv = arr[:, 3] - arr[:, 1]
    dv = dv.reshape(-1, 1)
    dv = dv[:-1, :]

    # vf and a
    vf = arr[:, 1]
    a = np.diff(vf) * 10
    vf = vf.reshape(-1, 1)
    vf = vf[:-1, :]
    a = a.reshape(-1, 1)
    return np.hstack([dx, dv, vf, a])


def data_split(feature_a_in_one, DataSize, seed=RANDOM_SEED):
    # "feature_a_in_one" means all feature (i.e., dx, dv, v) and acceleration pair.

    feature_a_in_one = feature_a_in_one[:DataSize["total"]]  # in case the total number is not consistent
    train_val_ext, test = train_test_split(feature_a_in_one, test_size=DataSize['test'], random_state=seed)
    train_val, ext = train_test_split(train_val_ext, test_size=DataSize['ext'], random_state=seed)
    train, val = train_test_split(train_val, test_size=DataSize['val'], random_state=seed)
    X_train = train[:, :3];
    a_train = train[:, 3].reshape(-1, 1)
    X_val = val[:, :3];
    a_val = val[:, 3].reshape(-1, 1)
    X_test = test[:, :3];
    a_test = test[:, 3].reshape(-1, 1)

    # collocation data
    X_ext = ext[:, :3]
    X_aux = np.concatenate([X_train, X_ext])
    return X_train, a_train, X_val, a_val, X_test, a_test, X_aux


def simulate(init, XV_L, nn):
    # long-term simulation: given the initial state, simulation the whole trajectory.
    # note that the leader's trajectory is given

    xl_0 = init[2]
    vl_0 = init[3]
    xf_0 = init[0]
    vf_0 = init[1]
    assert xl_0 == XV_L[0, 0]
    assert vl_0 == XV_L[0, 1]
    XF = [xf_0]
    VF = [vf_0]

    for i in range(0, len(XV_L) - 1):
        one_state = np.array([XV_L[i, 0], XV_L[i, 1], XF[-1], VF[-1]])
        feature = np.array([one_state[0] - one_state[2], one_state[1] - one_state[3], one_state[-1]])
        a = nn(torch.Tensor(feature.reshape(-1, 3))).cpu().detach().numpy()  # xv_lf
        v_next = VF[-1] + 0.1 * a
        x_next = (v_next + VF[-1]) / 2 * 0.1 + XF[-1]
        XF.append(x_next.flatten()[0])
        VF.append(v_next.flatten()[0])
    return XF, VF


# relative error function
def get_XV_error(xvfl_test, nn):
    XFs = []
    VFs = []
    XF_tests = []
    VF_tests = []
    for test_idx in range(len(xvfl_test)):
        XF, VF = simulate(xvfl_test[test_idx][0, :], xvfl_test[test_idx][:, 2:], nn)
        XFs.append(np.array(XF))
        VFs.append(np.array(VF))
        XF_tests.append(xvfl_test[test_idx][:, 0])
        VF_tests.append(xvfl_test[test_idx][:, 1])

    XFs = np.concatenate(XFs)
    VFs = np.concatenate(VFs)
    XF_tests = np.concatenate(XF_tests)
    VF_tests = np.concatenate(VF_tests)

    X_error = np.sqrt(sum(np.square(XFs - XF_tests)) / sum(np.square(XF_tests)))
    V_error = np.sqrt(sum(np.square(VFs - VF_tests)) / sum(np.square(VF_tests)))

    return X_error, V_error


def data_revise_nn(num_train, feature_a, seed=1234):
    np.random.seed(seed)
    DataSize = {"train": num_train,
                "ext": 300,
                "val": int(0.4 * num_train),
                "test": 300}
    DataSize["total"] = sum(DataSize.values())

    idx = np.random.choice(len(feature_a_in_one), DataSize["total"], replace=False)  # feature_a_in_one is global
    feature_a_in_one_dowsample = feature_a[idx]
    X_train, a_train, X_val, a_val, X_test, a_test, X_aux = data_split(feature_a_in_one_dowsample, DataSize, seed=seed)

    # model
    X_train = torch.Tensor(X_train).to(DEVICE)
    a_train = torch.Tensor(a_train).to(DEVICE)
    X_val = torch.Tensor(X_val).to(DEVICE)
    a_val = torch.Tensor(a_val).to(DEVICE)
    X_test = torch.Tensor(X_test).to(DEVICE)
    a_test = torch.Tensor(a_test).to(DEVICE)
    X_aux = torch.Tensor(X_aux).to(DEVICE)

    return X_train, a_train, X_val, a_val, X_test, a_test, X_aux


args_GA = {
    'sol_per_pop': 10,
    'num_parents_mating': 5,
    'num_mutations': 1,  # set 1 to mutate all the parameters
    'mutations_extend': 0.1,
    'num_generations': 10,

    'delta_t': 0.1,
    'mse': 'position',
    'RMSPE_alpha_X': 0.5,
    'RMSPE_alpha_V': 0.5,
    'lb': [10, 0, 0, 0, 0],
    'ub': [40, 10, 10, 5, 5]
}



# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    with open(os.path.join('data', 'idm_data_no_restriction.pickle'), 'rb') as f:
        data_pickle = pickle.load(f)

    # xvfl means "position (x) and velocity (v) of the follower (f) and the leader (l)"
    # that is, the columns stand for: 1. position of the follower; 2. velocity of the follower
    #                                 3. position of the leader;   4. velocity of the leader

    xvfl = data_pickle['idm_data']  # x, v of leading and following

    # test data for final evaluation
    xvfl_test = xvfl[-1 * Eval_num:]

    # state to feature
    feature_a = list(map(xvfl_to_feature, xvfl[:-1 * Eval_num]))
    feature_a_in_one = np.vstack(feature_a)

    X_train, a_train, X_val, a_val, X_test, a_test, X_aux = data_revise_nn(NUM_TRAIN, feature_a_in_one)


    if USE_GA is True:
        ga = GA(args_GA)
        #para, mse, duration = ga.executeGA(xvfl)
        para = np.array([30.51004085,  1.08624567,  7.24089548,  1.80457837,  3.37961181])
        para = {"v0": para[0],
                "T": para[1],
                "s0": para[2],
                "a": para[3],
                "b": para[4]}
    else:
        para = data_pickle['para']

    nn_kwargs["mean"] = X_train.mean(0)
    nn_kwargs["std"] = X_train.std(0)

    net = NN(nn_args, nn_kwargs)
    physics = IDM(para, params_trainable, device = DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad is True]
        , **optimizer_kwargs)

    if sum(list(params_trainable.values())) > 0:
        optimizer_physics = torch.optim.Adam(
            [p for p in physics.torch_params.values() if p.requires_grad is True]
            , **optimizer_physics_kwargs)
    else:
        optimizer_physics = None

    loss_fun = torch.nn.MSELoss()

    # train model
    best_mse = 100000
    best_it = 0
    for it in range(N_ITER):
        a_pred = net(X_train)
        a_pred_aux_nn = net(X_aux)
        a_pred_aux_phy = physics(X_aux)

        loss_obs = loss_fun(a_pred, a_train)
        loss_aux = loss_fun(torch.flatten(a_pred_aux_nn), a_pred_aux_phy)

        loss = ALPHA*loss_obs + (1-ALPHA)*loss_aux

        optimizer.zero_grad()
        if sum(list(params_trainable.values())) > 0:
            optimizer_physics.zero_grad()

        loss.backward()

        optimizer.step()
        if sum(list(params_trainable.values())) > 0:
            optimizer_physics.step()


        a_pred_val = net(X_val)
        loss_val = loss_fun(a_val, a_pred_val).cpu().detach().numpy()

        print('it=', it,  'loss_train=', loss_obs.cpu().detach().numpy(), " loss_val=", loss_val)

        # early stop

        if loss_val < best_mse:
            best_mse = loss_val
            best_it = it

        if it-best_it > 50:
            break

    # test: long-term simulation

    X_error, V_error = get_XV_error(xvfl_test, net)
    print('position error: ', X_error, '     velocity error: ', V_error)














