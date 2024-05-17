import gymnasium as gym
import onnxruntime
import numpy as np
from .data.fusion_lstm.fusion_utils import bpw_nn,kstar_nn,kstar_lstm
from .data.fusion_lstm.fusion_utils import i2f, f2i
# import random
import os
import time 
import json, zipfile
from .reward.fusion_reward import get_reward
from .base import Env

class FusionEnv(Env):
    low_state   = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18, 1.1, 3.8, 0.84, 1.1, 3.8, 0.84, 1.15, 1.15, 0.45]

    input_params = ['Ip [MA]','Bt [T]','GW.frac. [-]',\
                    'Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]',\
                    'Pec2 [MW]','Pec3 [MW]','Zec2 [cm]','Zec3 [cm]',\
                    'In.Mid. [m]','Out.Mid. [m]','Elon. [-]','Up.Tri. [-]','Lo.Tri. [-]'] #15dims
                    
    input_init = [0.5,1.8,0.275, 1.5, 1.5, 0.5, 0.0,0.0,0.0,0.0, 1.32, 2.22,1.7,0.3,0.75]
    input_mins = [0.3,1.5,0.2, 0.0, 0.0, 0.0, 0.0,0.0,-10,-10, 1.265,2.18,1.6,0.1,0.5 ]
    input_maxs = [0.8,2.7,0.6, 1.75,1.75,1.5, 0.8,0.8, 10, 10, 1.36, 2.29,2.0,0.5,0.9 ]

    output_params0 = ['βn','q95','q0','li']
    output_params1 = ['βp','wmhd']
    output_params2 = ['βn','βp','h89','h98','q95','q0','li','wmhd']

    target_params = ['βp','q95','li']
    target_init   = [ 1.6,  5.0,0.95]
    target_mins   = [ 1.1,  3.8,0.84]
    target_maxs   = [ 2.1,  6.2,1.06]
    
    dummy_params = ['Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 'In.Mid. [m]', 'Out.Mid. [m]']

    decimal = np.log10(1000)
    max_models = 10
    dir, _ = os.path.split(os.path.abspath(__file__))
    nn_model_path = os.path.join(dir, 'data/fusion_lstm/weights/nn') 
    lstm_model_path = os.path.join(dir, 'data/fusion_lstm/weights/lstm')
    bpw_model_path = os.path.join(dir, 'data/fusion_lstm/weights/bpw')
    plot_length = 165
    interval = 20
    year_in = 2021

    def __init__(self,
                random_target: bool=True,
                max_episode_steps: int=100,
                seed: float=None,
                mode: str='train'):
        
        self._init_inputSliderDict()
        self._init_outputs()
        self._init_targetSliderDict()
        self._init_targets()

        self.random_target = random_target

        # self.action_space = gym.spaces.Box(low=np.array([0.0]*6), high=np.array([1.0]*6), shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*6), high=np.array([1.0]*6), shape=(6,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=np.array([0.3, 1.6, 0.1, 0.5, 1.265,2.18, 
                                                              1.1,  3.8,0.84,
                                                              1.1,  3.8,0.84, 
                                                              0.0, 0.0, 0.0,]), 
                                                high=np.array([0.8, 2.0, 0.5, 0.9, 1.36, 2.29,
                                                               2.1,  6.2, 1.06,
                                                               2.1,  6.2, 1.06,
                                                               1.75,1.75, 1.5,]), 
                                                shape=(15,), dtype=np.float32)
        self._max_episode_steps = max_episode_steps
        self.seed(seed)
        self._mode = mode

    def seed(self,seed):
        if seed is None:
            self._seed = np.random.randint(1,100)
            self.np_random = np.random.RandomState(seed=self._seed)
            return
        self._seed = seed
        self.np_random = np.random.RandomState(seed=self._seed)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def reset(self, seed=None, **kwargs):
        self.seed(seed)

        self.kstar_nn = kstar_nn(model_path=self.nn_model_path, n_models=1)
        self.kstar_lstm = kstar_lstm(model_path=self.lstm_model_path, n_models=self.max_models)
        self.bpw_nn = bpw_nn(model_path=self.bpw_model_path, n_models=self.max_models)

        self.t = 0
        self.x = np.zeros([10,21], dtype=np.float32)
        """
        'βn','q95','q0','li' 0-3
        'Ip','Bt' 'Pnb1a','Pnb1b','Pnb1c', 4-8
        'Pec2','Pec3','Zec2','Zec3 ', 9-12
        'R_in','R_out','κ','Sig_u','Sig_l', 13-17
        'R_in' 'GW.frac. [-]' 18-19
        'year' 20
        """
        self.first = True
        self._predict0d(steady=True)
        self.first = False

        idx_convert = [0, 12, 13, 14, 10, 11]
        observation = np.zeros_like(self.low_state)

        observation[:6] = [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in idx_convert]
        observation[6:9] = [self.outputs[self.output_params2[i]][-1] for i in [1, 4, 6]]
        observation[9:12] = [i2f(self.targetSliderDict[self.target_params[i]], self.decimal) for i in [0, 1, 2]]
        observation[12:] =  [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in [3, 4, 5]]
        observation = observation.astype(np.float32)
        self.obs = observation
        return observation, {}

    def step(self, action):
        # breakpoint()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = (action+1)/2
        self.t += 1
        # _pre_target = [i2f(self.targetSliderDict[self.target_params[i]], self.decimal) for i in [0, 1, 2]]
        
        # idx_convert = [0, 12, 13, 14, 10, 11]
        # pre_observation = np.zeros_like(self.low_state)
        # # pre_observation[:6] = [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in idx_convert]
        # # pre_observation[6:9] = [self.outputs[self.output_params2[i]][-1] for i in [1, 4, 6]]
        # pre_observation[9:12] = [i2f(self.targetSliderDict[self.target_params[i]], self.decimal) for i in [0, 1, 2]]
        # # pre_observation[12:] =  [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in [3, 4, 5]]


        ########################
        idx_convert = [0, 12, 13, 14, 10, 11]
        # 返归一化action
        action_max_min = [self.input_maxs[i] - self.input_mins[i] for i in idx_convert]
        action_min = [self.input_mins[i] for i in idx_convert]
        _temp_action = np.array(action_max_min) * action + np.array(action_min)

        new_action = _temp_action
        current_action = [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in idx_convert]
        daction = (new_action - current_action) / self.interval
        for i in range(self.interval): # Control phase
            current_action += daction
            for i, idx in enumerate(idx_convert):
                self.inputSliderDict[self.input_params[idx]] = f2i(current_action[i], self.decimal)        

        for i in range(self.interval): # Relaxation phase
            if i < self.interval - 1:
                self._predict0d(steady=False)

        
        #一轮结束后 对 target 和['Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]'] 进行重新采样 随机数据
        if self.random_target:
            for target_param in self.target_params:
                idx = self.target_params.index(target_param)
                _temp_value = self.np_random.uniform(self.target_mins[idx], self.target_maxs[idx])
                self.targetSliderDict[target_param]= f2i(_temp_value, self.decimal)

            for input_param in ['Pnb1a [MW]','Pnb1b [MW]','Pnb1c [MW]']:
                idx = self.input_params.index(input_param)
                _temp_value = self.np_random.uniform(self.input_mins[idx], self.input_maxs[idx])
                self.inputSliderDict[input_param] = f2i(_temp_value, self.decimal)

        #得到next obs
        idx_convert = [0, 12, 13, 14, 10, 11]
        observation = np.zeros_like(self.low_state)
        observation[:6] = [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in idx_convert]
        observation[6:9] = [self.outputs[self.output_params2[i]][-1] for i in [1, 4, 6]]
        observation[9:12] = [i2f(self.targetSliderDict[self.target_params[i]], self.decimal) for i in [0, 1, 2]]
        observation[12:] =  [i2f(self.inputSliderDict[self.input_params[i]], self.decimal) for i in [3, 4, 5]]
        observation = observation.astype(np.float32)

        # rew = np.sum(-np.log(np.abs(np.array(_pre_target)-observation[6:9])))
        _data = {
            "obs": self.obs,
            "action": action,
            "next_obs" : observation,
                }
        rew = get_reward(_data)
        # error_scale = (observation[6:9] - np.array(_pre_target))/np.array([0.2, 0.5, 0.05])
        # rms = np.sqrt(np.mean(error_scale**2))
        # rew = -np.log(rms)

        if self.t >= self.max_episode_steps:
            terminate = True
        else:
            terminate = False

        self.obs = observation

        return observation, rew, False, False, {'target':observation[9:12], 'PNB': observation[12:]} #False

    def _predict0d(self,steady=True):
        # Predict output_params0 (βn, q95, q0, li)
        if steady:
            x = np.zeros(17,dtype=np.float32)
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(x)-1):
                x[i] = self.inputSliderDict[self.input_params[idx_convert[i]]]/10**self.decimal
            x[9],x[10] = 0.5*(x[9]+x[10]),0.5*(x[10]-x[9])
            x[14] = 1 if x[14]>1.265+1.e-4 else 0
            x[-1] = self.year_in
            y = self.kstar_nn.predict(x)
            if y.shape[0] == 1:
                y = y.reshape((y.shape[-1],))
            for i in range(len(self.output_params0)):
                if len(self.outputs[self.output_params0[i]]) >= self.plot_length:
                    del self.outputs[self.output_params0[i]][0]
                elif len(self.outputs[self.output_params0[i]]) == 1:
                    self.outputs[self.output_params0[i]][0] = y[i]
                self.outputs[self.output_params0[i]].append(y[i])

            self.x[:,:len(self.output_params0)] = y
            self.x[:,len(self.output_params0):] = x
        else:
            self.x[:-1,len(self.output_params0):] = self.x[1:,len(self.output_params0):]
            
            idx_convert = [0,1,3,4,5,6,7,8,9,10,11,12,13,14,10,2]
            for i in range(len(self.x[0])-1-4):
                self.x[-1,i+4] = self.inputSliderDict[self.input_params[idx_convert[i]]]/10**self.decimal  
            self.x[-1,13],self.x[-1,14] = 0.5*(self.x[-1,13]+self.x[-1,14]),0.5*(self.x[-1,14]-self.x[-1,13])
            self.x[-1,18] = 1 if self.x[-1,18]>1.265+1.e-4 else 0

            y = self.kstar_lstm.predict(self.x)
            if y.shape[0] == 1:
                y = y.reshape((y.shape[-1],))
            self.x[:-1,:len(self.output_params0)] = self.x[1:,:len(self.output_params0)]
            self.x[-1, :len(self.output_params0)] = y

            for i in range(len(self.output_params0)): #['βn','q95','q0','li']
                if len(self.outputs[self.output_params0[i]]) >= self.plot_length:
                    del self.outputs[self.output_params0[i]][0]
                elif len(self.outputs[self.output_params0[i]]) == 1:
                    self.outputs[self.output_params0[i]][0] = y[i]
                self.outputs[self.output_params0[i]].append(y[i])

        # Update output targets (βp, q95, li)
        if not self.first:
            for i, target_param in enumerate(self.target_params): #['βp','q95','li']
                if len(self.targets[target_param]) >= self.plot_length:
                    del self.targets[target_param][0]
                elif len(self.targets[target_param]) == 1:
                    self.targets[target_param][0] = i2f(self.targetSliderDict[target_param],self.decimal)
                self.targets[target_param].append(i2f(self.targetSliderDict[target_param],self.decimal) )

        # Predict output_params1 (βp, wmhd)
        x = np.zeros(8,dtype=np.float32)
        idx_convert = [0,0,1,10,11,12,13,14]
        x[0] = self.outputs['βn'][-1]
        for i in range(1,len(x)):
            x[i] = self.inputSliderDict[self.input_params[idx_convert[i]]]/10**self.decimal
        x[3],x[4] = 0.5*(x[3]+x[4]),0.5*(x[4]-x[3])
        y = self.bpw_nn.predict(x)
        if y.shape[0] == 1:
            y = y.reshape((y.shape[-1],))
        for i in range(len(self.output_params1)):
            if len(self.outputs[self.output_params1[i]]) >= self.plot_length:
                del self.outputs[self.output_params1[i]][0]
            elif len(self.outputs[self.output_params1[i]]) == 1:
                self.outputs[self.output_params1[i]][0] = y[i]
            self.outputs[self.output_params1[i]].append(y[i])

        # Store dummy parameters (Ip)
        for p in self.dummy_params:
            if len(self.dummy[p]) >= self.plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = i2f(self.inputSliderDict[p],self.decimal)
            self.dummy[p].append(i2f(self.inputSliderDict[p],self.decimal))

        # Estimate H factors (h89, h98)
        ip = self.inputSliderDict['Ip [MA]']/10**self.decimal
        bt = self.inputSliderDict['Bt [T]']/10**self.decimal
        fgw = self.inputSliderDict['GW.frac. [-]']/10**self.decimal
        ptot = max(self.inputSliderDict['Pnb1a [MW]']/10**self.decimal \
               + self.inputSliderDict['Pnb1b [MW]']/10**self.decimal \
               + self.inputSliderDict['Pnb1c [MW]']/10**self.decimal \
               + self.inputSliderDict['Pec2 [MW]']/10**self.decimal \
               + self.inputSliderDict['Pec3 [MW]']/10**self.decimal \
               , 1.e-1) # Not to diverge
        rin = self.inputSliderDict['In.Mid. [m]']/10**self.decimal
        rout = self.inputSliderDict['Out.Mid. [m]']/10**self.decimal
        k = self.inputSliderDict['Elon. [-]']/10**self.decimal

        rgeo,amin = 0.5*(rin+rout),0.5*(rout-rin)
        ne = fgw*10*(ip/(np.pi*amin**2))
        m = 2.0 # Mass number

        tau89 = 0.038*ip**0.85*bt**0.2*ne**0.1*ptot**-0.5*rgeo**1.5*k**0.5*(amin/rgeo)**0.3*m**0.5
        tau98 = 0.0562*ip**0.93*bt**0.15*ne**0.41*ptot**-0.69*rgeo**1.97*k**0.78*(amin/rgeo)**0.58*m**0.19
        h89 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau89
        h98 = 1.e-6*self.outputs['wmhd'][-1]/ptot/tau98

        if len(self.outputs['h89']) >= self.plot_length:
            del self.outputs['h89'][0], self.outputs['h98'][0]
        elif len(self.outputs['h89']) == 1:
            self.outputs['h89'][0], self.outputs['h98'][0] = h89, h98

        self.outputs['h89'].append(h89)
        self.outputs['h98'].append(h98)

    def _init_inputSliderDict(self,):
        self.inputSliderDict = {}
        for input_param in self.input_params:
            idx = self.input_params.index(input_param)
            self.inputSliderDict[input_param]= f2i(self.input_init[idx], self.decimal)

    def _init_outputs(self,):
        self.outputs, self.dummy = {}, {}
        for p in self.output_params2:
            self.outputs[p] = [0.]
        for p in self.dummy_params:
            self.dummy[p] = [0.]

    def _init_targetSliderDict(self):
        self.targetSliderDict = {}
        for target_param in self.target_params:
            idx = self.target_params.index(target_param)
            self.targetSliderDict[target_param]= f2i(self.target_init[idx], self.decimal)
            
    def _init_targets(self):
        self.targets = {}
        for i,target_param in enumerate(self.target_params):
            self.targets[target_param] = [self.target_init[i], self.target_init[i]]

def actv(x, method):
    if method == 'relu':
        return np.max([np.zeros_like(x), x], axis=0)
    elif method == 'tanh':
        return np.tanh(x)
    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif method == 'linear':
        return x

class SB2_model():
    def __init__(self, model_path, low_state, high_state, low_action, high_action, activation='relu', last_actv='tanh', norm=True, bavg=0.):
        zf = zipfile.ZipFile(model_path)
        data = json.loads(zf.read('data').decode("utf-8"))
        self.parameter_list = json.loads(zf.read('parameter_list').decode("utf-8"))
        self.parameters = np.load(zf.open('parameters'))
        self.layers = data['policy_kwargs']['layers'] if 'layers' in data['policy_kwargs'].keys() else [64, 64]
        self.low_state, self.high_state = low_state, high_state
        self.low_action, self.high_action = low_action, high_action
        self.activation, self.last_actv = activation, last_actv
        self.norm = norm
        self.bavg = bavg

    def predict(self, x, yold=None):
        xnorm = 2 * (x - self.low_state) / np.subtract(self.high_state, self.low_state) - 1 if self.norm else x
        ynorm = xnorm
        for i, layer in enumerate(self.layers):
            w, b = self.parameters[f'model/pi/fc{i}/kernel:0'], self.parameters[f'model/pi/fc{i}/bias:0']
            ynorm = actv(np.matmul(ynorm, w) + b, self.activation)
        w, b = self.parameters[f'model/pi/dense/kernel:0'], self.parameters[f'model/pi/dense/bias:0']
        ynorm = actv(np.matmul(ynorm, w) + b, self.last_actv)

        y = 0.5 * np.subtract(self.high_action, self.low_action) * (ynorm + 1) + self.low_action if self.norm else ynorm
        if yold is None:
            yold = x[:len(y)]
        y =  self.bavg * yold + (1 - self.bavg) * y
        return y


if __name__ =='__main__':
    import time
    rl_designer_model_path =  './data/fusion_lstm/weights/best_model.zip'
    low_state   = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18, 1.1, 3.8, 0.84, 1.1, 3.8, 0.84, 1.15, 1.15, 0.45]
    high_state  = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29, 2.1, 6.2, 1.06, 2.1, 6.2, 1.06, 1.75, 1.75, 0.6]
    low_action  = [0.35, 1.68, 0.2, 0.5, 1.265, 2.18]
    high_action = [0.75, 1.9, 0.5, 0.8, 1.34, 2.29]
    designer = SB2_model(
                model_path = rl_designer_model_path, 
                low_state = low_state, 
                high_state = high_state, 
                low_action = low_action, 
                high_action = high_action,
            )


    env = FusionEnv()

    all_reward = []
    for s in range(100):        
        obs,_ = env.reset()
        next_obs_list = []
        rew_list = []
        done = False
        while not done:
            action = designer.predict(obs)
            input_mins = np.array([0.3, 1.6,0.1,0.5, 1.265,2.18 ])
            input_maxs = np.array([0.8, 2.0,0.5,0.9, 1.36, 2.29 ])
            action = (action-input_mins)/(input_maxs-input_mins)
            action = np.random.normal(action, 0.05)
            # action = np.array([random.uniform(env.input_mins[i], env.input_maxs[i]) for i in [0, 12, 13, 14, 10, 11]])
            next_obs, rew, _, done, info = env.step(action)
            # print(rew)
            next_obs_list.append(next_obs)
            obs = next_obs
            rew_list.append(rew)
            if len(rew_list)>=100:
                done = True
        print(len(rew_list),np.sum(rew_list))
        all_reward.append(np.sum(rew_list))

    
        # action = np.array([random.uniform(env.input_mins[i], env.input_maxs[i]) for i in [0, 12, 13, 14, 10, 11]])
    #     next_obs, rew, done, info = env.step(action)
    #     next_obs_list.append(next_obs)
    #     # print(rew)
    #     #print(next_obs[9:12],next_obs[12:])
    # next_obs_list = np.stack(next_obs_list)
    breakpoint()
