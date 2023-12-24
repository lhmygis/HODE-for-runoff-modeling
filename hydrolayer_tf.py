import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation, LSTM
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class HODE(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    f: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
    smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
    qmax: Maximum subsurface flow at full bucket     | Range: (10, 50)
    ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
    tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
    tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)
    """

    def __init__(self, h_nodes=128, mode='normal', seed = 200, **kwargs):
        self.h_nodes = h_nodes
        self.mode = mode
        self.seed = seed
        super(HODE, self).__init__(**kwargs)

    def build(self, input_shape):
        self.f = self.add_weight(name='f', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)

        self.hode_w1 = self.add_weight(name='hode_w1',
                                                shape=(5,self.h_nodes),
                                                initializer=initializers.RandomUniform(seed=self.seed - 5),
                                                trainable=True)
        self.hode_b1 = self.add_weight(name='hode_b1',
                                  shape=(self.h_nodes,),
                                  initializer=initializers.zeros(),
                                  trainable=True)
        self.hode_w2 = self.add_weight(name='hode_w2',
                                  shape=(self.h_nodes, 6),
                                  initializer=initializers.RandomUniform(seed=self.seed + 5), # avoid same initial w1 and w2
                                  trainable=True)
        self.hode_b2 = self.add_weight(name='hode_b2',
                                  shape=(1,),
                                  initializer=initializers.zeros(),
                                  trainable=True)

        super(HODE, self).build(input_shape)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (K.tanh(5 * x) + 1) / 2

    def rainsnowpartition(self, p, t, tmin):
        """
        Equations to partition incoming precipitation into rain or snow
            if t < tmin:
                psnow = p
                prain = 0
            else:
                psnow = 0
                prain = p
        """
        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s0, t, ddf, tmax):
        """
        Equations for the snow bucket
            if t > tmax:
                if s0 > 0:
                    melt = min(s0, ddf*(t - tmax))
                else:
                    melt = 0
            else:
                melt = 0
        """
        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    def soilbucket(self, s1, pet, f, smax, qmax):
        """
        Equations for the soil bucket
            if s1 < 0:
                et = 0
                qsub = 0
                qsurf = 0
            elif s1 > smax:
                et = pet
                qsub = qmax
                qsurf = s1 - smax
            else:
                et = pet * (s1 / smax)
                qsub = qmax * exp(-f * (smax - s1))
                qsurf = 0
        """
        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):  # Define step function for the RNN
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]
        dayl = step_in[:,3:4]
        srad = step_in[:,4:5]
        vp = step_in[:,5:6]

        # Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)
        # Snow bucket
        _m = self.snowbucket(s0, t, self.ddf, self.tmax)
        # Soil bucket
        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        _q_physics = _qsurf + _qsub
        #_q_nn1 = K.tanh(K.dot(p, self.prnn_neuralode_w1)+ self.prnn_neuralode_b1)
        #_q_nn2 = K.tanh(K.dot(_q_nn1, self.prnn_neuralode_w2)+ self.prnn_neuralode_b2)

        _q_hybrid = (_q_physics + \
                     K.sigmoid( K.dot(K.sigmoid(K.dot(K.concatenate((s1, p, t, dayl, srad, vp), axis=-1), self.hode_w1)+ self.hode_b1), self.hode_w2) + self.hode_b2  ) ) / 2

        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _q_hybrid
        #_ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]
        srad = inputs[:,:,3:4]
        vp = inputs[:,:,4:5]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet, dayl, srad, vp), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        [ps,pr] = self.rainsnowpartition(prcp,tmean,self.tmin)
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        q = qsub + qsurf

        if self.mode == "normal":
            return qsub + qsurf
        elif self.mode == "analysis":
            return K.concatenate([q, s0, s1, qsub, qsurf, et, pr, m, ps], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 9)

class ODERNN(Layer):

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(ODERNN, self).__init__(**kwargs)

    #用于创建自定义层权重的函数, 该函数必须要有input_shape参数
    def build(self, input_shape):
        #print('******',input_shape)    [None,2190,5]

        #将EXP-HYDRO model 计算需要的6个物理参数作为该层的6个权重:f smax qmax ddf tmin tmax
        self.f = self.add_weight(name='f', shape=(1,),
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)

        #这行目的是为了保证该层的权重定义函数build()被执行过了
        super(ODERNN, self).build(input_shape)

    def heaviside(self, x):

        #tanh()返回一个[-1,1]的数  ,  5*x越大返回值越接近1   越小越接近-1   x=0时返回值也为0
        return (K.tanh(5 * x) + 1) / 2

    #用于计算EXO-HYDRO中 flex variables(Ps,Pr) 的函数
    def rainsnowpartition(self, p, t, tmin):

        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    #用于计算EXO-HYDRO中雪融量snowmelt——M的函数, M也属于flex variables
    def snowbucket(self, s0, t, ddf, tmax):

        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    #用于计算EXO-HYDRO中蒸散发ET,集水桶可用蓄水量Qb,和集水桶饱和时产生的超容径流（𝑄s） ： ET属于flex variables, Qb(Qsub) + 𝑄s(Qsurf) 等于 Q
    def soilbucket(self, s1, pet, f, smax, qmax):

        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        # q = f((qsurb + qsurf, xt, p)) + n (xt, w, b)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]


        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)

        _m = self.snowbucket(s0, t, self.ddf, self.tmax)

        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        # _q = f((_qsurb + _qsurf, xt, p)) + ( n (xt, w, b)  - 观测量  ）
        #  q  = f((_qsurb + _qsurf, xt, p))+ NN ( 物理部分 -  观测 )

        # Water balance equations
        _ds0 = _ps - _m
        # _ds1 = _pr + _m - _et - _q
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    #call函数用于实现该层的功能逻辑, 即对于输入张量的计算, 即计算输出径流Q的地方, Keras中x(inputs)只能是一种形式 , 所以不能被事先定义
    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        # return 3 outputs:
        # last_output (the latest output of the rnn, through last time g() & *V & +b)
        # output (all outputs [wrap_number_train, wrap_length, output], through all time g() & *V & +b)
        # new_states(latest states returned by the step_do function, without through last time g() & *V & +b)
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        #outputs: outputs是一个tuple，outputs[0]为最后时刻的输出，outputs[1]为整个输出的时间序列，output[2]是一个list，是中间的隐藏状态。

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)


        if self.mode == "normal":
            return qsub+qsurf
        elif self.mode == "analysis":
            return K.concatenate([s0, m, et, qsurf, qsub, s1], axis=-1)

    # 为了能让Keras内部shape的匹配检查通过，这里需要重写compute_output_shape方法去覆盖父类中的同名方法，来保证输出shape是正确的。
    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)

class LSTMLayer(Layer):
    def __init__(self, units, seq_len, input_dim, seed=200, **kwargs):
        self.units = units
        self.seq_len = seq_len
        self.input_dim = input_dim

        super(LSTMLayer, self).__init__(**kwargs)
        self.lstm = keras.models.Sequential()


        self.lstm.add(LSTM(units=self.units, input_shape=(self.seq_len, self.input_dim),
                           activation='tanh', return_sequences=True))
        self.fc1 = Dense(units = 64, activation='relu')
        self.fc2 = Dense(units = 1)

    def call(self, inputs):

        x = self.lstm(inputs)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class ScaleLayer(Layer):


    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        #met(气象输入) = [wrap_number_train, wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]
        met = inputs[:, :, :-1]

        #flow(Q输入) = [wrap_number_train, wrap_length, 1('Q(mm)')]
        flow = inputs[:, :, -1:]
        print("flow_calculatedby_fir_rnncel:",flow)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]  五个变量的mean值
        self.met_center = K.mean(met, axis=-2, keepdims=True)

        #[wrap_number_train, 1, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]   五个变量的std值
        self.met_scale = K.std(met, axis=-2, keepdims=True)

        #[wrap_number_train,  wrap_length, 5('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)')]   五个变量进行标准化
        self.met_scaled = (met - self.met_center) / self.met_scale

        self.flow_center = K.mean(flow, axis=-2, keepdims=True)
        self.flow_scale = K.std(flow, axis=-2, keepdims=True)
        self.flow_scaled = (flow - self.flow_center) / self.flow_scale


        #五个met气象输入变量与预测变量Q重新拼接在一起  return
        return K.concatenate([self.met_scaled, flow], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape
































