import numpy
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

class neuralNetwork():

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inode=inputnodes
        self.hnode=hiddennodes
        self.onode=outputnodes
        self.lr=learningrate
        self.alllayer=[]
        self.allweights={}
        self.reversedallweights={}
        self.allerrors={}
        self.allouts={}

        self.alllayer.append(self.inode)
        for nersize in self.hnode:
            self.alllayer.append(nersize)
        self.alllayer.append(self.onode)

        wei_center=0
        for weigt_all in self.alllayer[1:]:#bütün katman ağırlıkları eklendi
            wei_center+=1
            self.back=self.alllayer[wei_center-1]
            self.current=self.alllayer[wei_center]
            self.allweights[str(wei_center)] = numpy.random.normal(0.0, pow(self.back, -0.5), (self.current, self.back))

        self.lr=learningrate
        self.activation_function= lambda x :sigmoid(x)
        pass

    def reverse_dictionary(self,takedc):  # 122

        lst_keys = []
        lst_vals = []
        for lo in takedc.keys():
            lst_keys.append(lo)
            lst_vals.append(takedc[lo])
        lst_keys.reverse()
        lst_vals.reverse()
        reversed_dict = {}
        for ro, roo in zip(lst_keys, lst_vals):
            reversed_dict[ro] = roo
        return reversed_dict

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        self.allouts["o1"]=inputs
        outcounter=1
        starting_point=0#döngüye ilk defa mı giriyoruz
        for weigh in self.allweights.values():
            if starting_point==0:
                starting_point+=1
                self.ramstart=numpy.dot(weigh,inputs)
                self.ramstart=self.activation_function(self.ramstart)
                #print(self.ramstart,"1")
                self.allouts["o2"]=self.ramstart
                outcounter+=1
            else:
                outcounter+=1
                self.ramstart1=numpy.dot(weigh,self.ramstart)
                self.ramstart1=self.activation_function(self.ramstart1)
                self.ramstart=self.ramstart1
                #print(self.ramstart1,"2")
                self.allouts["o"+str(outcounter)]=self.ramstart1

        #print(self.ramstart1,"ramstart1")#forward sonucu ağ çıktısı

        output_error=targets - self.ramstart1
        #print(output_error)

        error_center=len(self.alllayer)-2         #giriş ve çıkış katmanını saymıyoruz
        #print(error_center)

        self.reversedallweights=self.reverse_dictionary(self.allweights.copy())#ağırlıkları tersliyoruz...

        st_point=0
        for revweigh in self.reversedallweights.values():#istenilen ara katman hatası kadar giriyoruz...
            if st_point==0:
                self.remst=numpy.dot(revweigh.T,output_error)
                self.allerrors[str(error_center)]=self.remst
                st_point+=1
                error_center-=1
            else:
                self.remst1=numpy.dot(revweigh.T,self.remst)
                self.allerrors[str(error_center)]=self.remst1
                self.remst=self.remst1
                error_center-=1
            if error_center == 0:
                break

        #print(self.allerrors,"*****")



        outlastcounter=len(self.alllayer)-1
        stl_point=0
        for update_weight in self.reversedallweights.keys():
            if stl_point == 0:
                val=self.reversedallweights[update_weight]#terslenmiş ağırlık listesinde ağırlığın değerini alıyoruz.
                val+=self.lr*numpy.dot((output_error*self.ramstart1*(1-self.ramstart1)),numpy.transpose(self.allouts["o"+str(outlastcounter)]))
                self.reversedallweights[update_weight]=val
                stl_point+=1
                outlastcounter-=1
            else:
                val1=self.reversedallweights[update_weight]
                val1+=self.lr*numpy.dot((self.allerrors[str(outlastcounter)]*self.allouts["o"+str(outlastcounter+1)]*(1-self.allouts["o"+str(outlastcounter+1)])),numpy.transpose(self.allouts["o"+str(outlastcounter)]))
                self.reversedallweights[update_weight]=val1
                outlastcounter-=1




        pass
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        starting_point=0
        for weigh in self.allweights.values():
            if starting_point == 0:
                starting_point += 1
                self.ramstart = numpy.dot(weigh, inputs)
                self.ramstart = self.activation_function(self.ramstart)



            else:

                self.ramstart1 = numpy.dot(weigh, self.ramstart)
                self.ramstart1 = self.activation_function(self.ramstart1)
                self.ramstart = self.ramstart1



        #print(self.ramstart1, "ramstart1")  # forward sonucu ağ çıktısı

        return self.ramstart1
        pass

input_node=2
hidden_nodes=[50,50]
output_node=2
learning_rate=0.9


n=neuralNetwork(inputnodes=input_node,hiddennodes=hidden_nodes,outputnodes=output_node,learningrate=learning_rate)

x_list=[[0,0],[0,1],[1,0],[1,1]]
y_list=[[0,1],[1,0],[1,1],[0,1]]


for k in range(1000):
    n.train(x_list,y_list)
    print(n.query(x_list[0]),n.query(x_list[1]),n.query(x_list[2]),n.query(x_list[3]))
