import gymnasium as gym
import onnxruntime
import numpy as np        

class bpw_nn():
    def __init__(self, model_path, n_models=1):
        self.nmodels = n_models
        self.ymean = np.array([1.02158800e+00, 1.87408512e+05])
        self.ystd  = np.array([6.43390272e-01, 1.22543529e+05])

        self.models = [onnxruntime.InferenceSession(model_path + f'/best_model{i}.onnx') for i in range(self.nmodels)]
        self.input_name = self.models[0].get_inputs()[0].name
        self.output_name = self.models[0].get_outputs()[0].name

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean([m.run([self.output_name], {self.input_name: self.x})[0] * self.ystd + self.ymean 
                                for m in self.models[:self.nmodels]], axis=0)
        return self.y

class kstar_nn():
    def __init__(self, model_path, n_models=1, ymean=None, ystd=None):
        self.nmodels = n_models
        if ymean is None:
            self.ymean = [1.22379703, 5.2361062,  1.64438005, 1.12040048]
            self.ystd  = [0.72255576, 1.5622809,  0.96563557, 0.23868018]
        else:
            self.ymean, self.ystd = ymean, ystd
        self.models = [onnxruntime.InferenceSession(model_path + f'/best_model{i}.onnx') for i in range(self.nmodels)]
        self.input_name = self.models[0].get_inputs()[0].name
        self.output_name = self.models[0].get_outputs()[0].name

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 2 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)
        self.y = np.mean([m.run([self.output_name], {self.input_name: self.x})[0] * self.ystd + self.ymean 
                                for m in self.models[:self.nmodels]], axis=0)
        return self.y

class kstar_lstm():
    def __init__(self, model_path, n_models=1, ymean=None, ystd=None):
        self.nmodels = n_models
        if ymean is None:
            self.ymean = [1.30934765, 5.20082444, 1.47538417, 1.14439883]
            self.ystd  = [0.74135689, 1.44731883, 0.56747578, 0.23018484]
        else:
            self.ymean, self.ystd = ymean, ystd
        self.models = [onnxruntime.InferenceSession(model_path + f'/best_model{i}.onnx') for i in range(self.nmodels)]
        

    def set_inputs(self, x):
        self.x = np.array(x) if len(np.shape(x)) == 3 else np.array([x])

    def predict(self, x=None):
        if type(x) == type(np.zeros(1)):
            self.set_inputs(x)

        result_list = []
        for i in range(self.nmodels):
            _input_name = self.models[i].get_inputs()[0].name
            _output_name = self.models[i].get_outputs()[0].name
            result = self.models[i].run([_output_name], {_input_name: self.x})[0] * self.ystd + self.ymean 
            result_list.append(result)
        
        self.y = np.stack(result_list).mean(axis=0)
        return self.y

def i2f(i,decimals:np.log10(1000)):
    return float(i/10**decimals)

def f2i(f,decimals:np.log10(1000)):
    return int(f*10**decimals)