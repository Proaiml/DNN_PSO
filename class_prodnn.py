import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import numpy as np
import matplotlib.pyplot as plt
import json
import random as rnd
import time

class prodnnv10():
    """
    input_neuron_numbers:Giriş nöron sayısı
    output_neuron_numbers:Çıkış nöron sayısı
    transition_per:Katmanlar arası geçiş oranı
    x_input:Eğitim için giriş Verileri
    y_input=Eğitim için çıkış verileri
    train_size:Kaç Kere itere edileceği
    particle:Parçacık sayısı
    train ettikten sonra en iyi ağırlıkları normal ağırlık olarak atıyor.
    """
    def __init__(self,input_neuron_numbers,output_neuron_numbers,transition_per,x_input,y_input,train_size,particle,loop_size):

        self.input_neuron_numbers=input_neuron_numbers
        self.output_neuron_numbers=output_neuron_numbers
        self.transition_per=transition_per
        self.x_input=x_input
        self.y_input=y_input
        self.train_size=train_size
        self.loop_size=loop_size
        self.hidden_neuron_numbers = []
        self.all_layers = []
        self.all_Weights = {}
        self.full_outputs_lastmend = []
        self.full_outputstotal = []
        self.costs = []
        self.options = {'c1': 1, 'c2': 0.8, 'w': 0.9}
        self.xread=[]
        self.yread=[]
        self.particle=particle
        self.agirliksay = 0
        self.dimensions = 0
        self.all_costlocal = {}



        with open(x_input,"r") as rdn:
            self.xread=json.load(rdn)

        with open(y_input,"r") as wdn:
            self.yread=json.load(wdn)



        input_neuron_numbers_var = input_neuron_numbers
        while True:
            add = input_neuron_numbers_var * transition_per
            add = int(add)
            if add > output_neuron_numbers:
                self.hidden_neuron_numbers.append(add)
                input_neuron_numbers_var = self.hidden_neuron_numbers[-1]
            else:
                break


        self.all_layers.append(input_neuron_numbers)
        for kl in self.hidden_neuron_numbers:
            self.all_layers.append(kl)
        self.all_layers.append(output_neuron_numbers)

        loop_len = self.all_layers[1:]
        for Lo in loop_len:
            for Loo in range(Lo):
                indes = self.all_layers.index(Lo)
                bottom_loop = self.all_layers[indes - 1]
                for Looo in range(0, bottom_loop):
                    uniquestr = "w" + str(indes) + "_" + str(Loo) + "_" + str(Looo)
                    self.all_Weights[uniquestr] = np.random.random()

        for kkm in self.all_Weights.keys():
            self.agirliksay += 1

        self.dimensions=self.agirliksay
    def cost_function(self,y_real, y_predict):
        erross = []
        for y_r, y_p in zip(y_real, y_predict):
            erross.append((y_r - y_p) ** 2)
        return 100*(sum(erross)) / len(erross)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)

    def errorn(self,output, actual_output):
        level = output - actual_output
        level = int(level)
        return level

    def forward_prop(self,x_ar, w_ar):
        x_arn = np.array([x_ar])
        y_arn = np.array([w_ar])
        dot_product = np.dot(x_arn, y_arn.T)

        return self.sigmoid(dot_product[0][0])


    def propagate_forward(self):


        loop_len = self.all_layers[1:]
        weights_local = []
        for Lo in loop_len:
            for Loo in range(Lo):
                indes = self.all_layers.index(Lo)
                bottom_loop = self.all_layers[indes - 1]
                wrt = []
                for Looo in range(0, bottom_loop):
                    uniquestr = "w" + str(indes) + "_" + str(Loo) + "_" + str(Looo)
                    wrt.append(self.all_Weights[uniquestr])
                weights_local.append(wrt)

        all_taggedweight = {}
        for tg in loop_len:
            for tgg in range(tg):
                indes1 = self.all_layers.index(tg)
                uniquestrnew = "w" + str(indes1) + "_" + str(tgg)
                all_taggedweight[uniquestrnew] = weights_local[0]
                weights_local = weights_local[1:]

        full_outputs = []
        for inu in self.xread:
            all_intputs = []
            all_intputs.append(inu)
            indesn = all_intputs.index(inu)
            for noo in loop_len:
                uniw = self.all_layers.index(noo)
                ram = []
                for nooo in range(noo):
                    uniquestrnewer = "w" + str(uniw) + "_" + str(nooo)
                    input = all_intputs[indesn]
                    resultn = self.forward_prop(input, all_taggedweight[uniquestrnewer])
                    ram.append(resultn)
                all_intputs.append(ram)
                indesn += 1
            full_outputs.append(all_intputs)
        self.full_outputstotal.append(full_outputs)
        full_outputs_last = []
        for kl in full_outputs:
            full_outputs_last.append(kl[-1:][0][0])
        full_outputs_lastm = full_outputs_last
        self.full_outputs_lastmend=full_outputs_lastm
        self.costs.append(self.cost_function(self.yread, full_outputs_lastm))
        return self.cost_function(self.yread, full_outputs_lastm)

    def f(self,x):

        n_particles = x.shape[0]
        agirliks = []
        rtnlst = []
        for kl in x:
            for kll in kl:
                agirliks.append(kll)
            sayac = 0
            for wi in self.all_Weights.keys():
                self.all_Weights[wi] = agirliks[sayac]
                sayac += 1
            rtnlst.append(self.propagate_forward())
            agirliks = []
        return np.array(rtnlst)

    def trainer(self):
        self.all_costlocal={}
        cost_auto=32768

        for nn in range(self.loop_size):
            optimizer = ps.single.GlobalBestPSO(n_particles=self.particle, dimensions=self.dimensions, options=self.options)
            cost, pos = optimizer.optimize(self.f, iters=self.train_size)
            print(cost,pos)
            agg=[]
            sayac = 0
            for wi in self.all_Weights.keys():
                self.all_Weights[wi] = pos[sayac]
                agg.append(pos[sayac])
                sayac += 1

            print(cost,cost_auto)
            if cost<cost_auto:
                cost_auto=cost
                self.all_costlocal={}
                self.all_costlocal[cost]=agg
        return self.all_costlocal



    def bestweight_upload(self):
        weigts=[]
        print(self.all_costlocal,"bestweupload")
        for tnn in self.all_costlocal.values():
            for tnx in tnn:
                weigts.append(tnx)

        sayac = 0
        for wi in self.all_Weights.keys():
            self.all_Weights[wi] = weigts[sayac]
            sayac += 1


    def cost_effect(self):
        x_band=[]
        sayac=0
        for kkm in range(len(self.costs)):
            x_band.append(sayac)
            sayac+=1
        plt.xlabel("Train size")
        plt.ylabel("Cost outputs")
        plt.plot(x_band,self.costs)
        plt.show()




