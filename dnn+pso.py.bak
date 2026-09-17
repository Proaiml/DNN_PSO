"""
This  ann code written by İlhan Koçaslan.
"""
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
import matplotlib.pyplot as plt
import json
import random as rnd


bias_neurons = 1
learning_rate=0.1
iteration_size=500
input_neuron_numbers = 3
output_neuron_numbers = 1
transition_per = 2/3


hidden_neuron_numbers=[]
all_layers=[]
all_Weights={}
full_outputs_lastm=[]
full_outputstotal=[]
x_input=[
                  [0,0,0],[0,0,1],
                  [0,1,0],[0,1,1],
                  [1,0,0],[1,0,1],
                  [1,1,0],[1,1,1]
                  ]
y_output=[1,0,0,0,0,1,0,0]

costs=[]

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}


def cost_function(y_real,y_predict):
    erross = []
    for y_r, y_p in zip(y_real, y_predict):
        erross.append((y_r - y_p) ** 2)
    return (sum(erross)) / len(erross)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def errorn(output,actual_output):
    level=output-actual_output
    level= int(level)
    return level



def forward_prop(x_ar,w_ar):
    x_arn=np.array([x_ar])
    y_arn=np.array([w_ar])
    dot_product=np.dot(x_arn,y_arn.T)

    return sigmoid(dot_product[0][0])

def backward_prop():
    print("backward...")


def find_layerneuronnumb():
    global input_neuron_numbers,transition_per,output_neuron_numbers,hidden_neuron_numbers
    input_neuron_numbers_var = input_neuron_numbers
    while True:
        add = input_neuron_numbers_var * transition_per
        add = int(add)
        if add > output_neuron_numbers:
            hidden_neuron_numbers.append(add)
            input_neuron_numbers_var = hidden_neuron_numbers[-1]
        else:
            break


def alllayeradd():
    global input_neuron_numbers,hidden_neuron_numbers,all_layers,output_neuron_numbers
    all_layers.append(input_neuron_numbers)
    for kl in hidden_neuron_numbers:
        all_layers.append(kl)
    all_layers.append(output_neuron_numbers)

def preweighter():
    loop_len = all_layers[1:]
    for Lo in loop_len:
        for Loo in range(Lo):
            indes = all_layers.index(Lo)
            bottom_loop = all_layers[indes - 1]
            for Looo in range(0, bottom_loop):
                uniquestr = "w" + str(indes) + "_" + str(Loo) + "_" + str(Looo)
                all_Weights[uniquestr] = np.random.random()


def propagate_forward():
    global all_layers,full_outputs_lastm,all_Weights
    loop_len = all_layers[1:]
    weights_local = []
    for Lo in loop_len:
        for Loo in range(Lo):
            indes = all_layers.index(Lo)
            bottom_loop = all_layers[indes - 1]
            wrt = []
            for Looo in range(0, bottom_loop):
                uniquestr = "w" + str(indes) + "_" + str(Loo) + "_" + str(Looo)
                wrt.append(all_Weights[uniquestr])
            weights_local.append(wrt)

    for inu in x_input:
        print(inu)

    all_taggedweight = {}
    for tg in loop_len:
        for tgg in range(tg):
            indes1 = all_layers.index(tg)
            uniquestrnew = "w" + str(indes1) + "_" + str(tgg)
            all_taggedweight[uniquestrnew] = weights_local[0]
            weights_local = weights_local[1:]

    full_outputs = []
    for inu in x_input:
        all_intputs = []
        all_intputs.append(inu)
        indesn = all_intputs.index(inu)
        for noo in loop_len:
            uniw = all_layers.index(noo)
            ram = []
            for nooo in range(noo):
                uniquestrnewer = "w" + str(uniw) + "_" + str(nooo)
                input = all_intputs[indesn]
                resultn = forward_prop(input, all_taggedweight[uniquestrnewer])
                ram.append(resultn)
            all_intputs.append(ram)
            indesn += 1
        full_outputs.append(all_intputs)
    full_outputstotal.append(full_outputs)
    full_outputs_last = []
    for kl in full_outputs:
        full_outputs_last.append(kl[-1:][0][0])
    full_outputs_lastm=full_outputs_last
    costs.append(cost_function(y_output, full_outputs_lastm))
    return cost_function(y_output, full_outputs_lastm)


def propagate_backward(y_real,y_pred,sgmdrv,):
    subtraction=-1*(y_real-y_pred)
    sgmder=sigmoid_derivative(sgmdrv)

    print("yapım aşamasında...")


find_layerneuronnumb()
alllayeradd()
preweighter()
propagate_forward()

print(x_input)
print(hidden_neuron_numbers)
print(all_layers)
print(all_Weights)
print(full_outputstotal)
print(cost_function(y_output,full_outputs_lastm))
for y,y_hat in zip(y_output,full_outputs_lastm):
    print(y,y_hat,y-y_hat)

def f(x):
    global all_Weights
    n_particles = x.shape[0]
    agirliks=[]
    rtnlst=[]
    for kl in x:
        for kll in kl:
            agirliks.append(kll)
        sayac=0
        for wi in all_Weights.keys():
            all_Weights[wi]=agirliks[sayac]
            sayac+=1
        rtnlst.append(propagate_forward())
        agirliks=[]
    return np.array(rtnlst)




agirliksay=0
for kkm in all_Weights.keys():
    agirliksay+=1
print(agirliksay)

print(all_Weights)



dimensions = agirliksay
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=dimensions, options=options)
cost, pos = optimizer.optimize(f, iters=1000)

print(costs)

xhand=[]
for kll in range(len(costs)):
    xhand.append(kll)
print(len(costs))
print(len(xhand))
plt.xlabel("iteration number")
plt.ylabel("cost value")
plt.title("Cost Function - Time")
plt.plot(xhand,costs)
plt.show()
print(pos)
agirlik_trained=[23.19277494, 21.24999851, 3.54920469,  1.13005887,   -5.91228895,
    0.95750409, -504.79684388, 608.67782358]

sayac=0
for wi in all_Weights.keys():
    all_Weights[wi]=agirlik_trained[sayac]
    sayac+=1
print(propagate_forward())
for kkk ,lll in zip(y_output,full_outputs_lastm):
    if lll > 0.5:
        lll=1
    else:
        lll=0
    print(kkk,lll)