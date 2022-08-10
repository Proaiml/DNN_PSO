import dnn


input_node=5
hidden_nodes=[4,10,2]
output_node=3
learning_rate=0.9
x_list=[[0,0,0,0,1],[1,1,1,1,0]]
y_list=[[0,1,0],[1,1,1]]


nn=dnn.neuralNetwork(inputnodes=input_node,hiddennodes=hidden_nodes,outputnodes=output_node,learningrate=learning_rate)

for k in range(1000):
    nn.train(inputs_list=x_list,targets_list=y_list)
    print(nn.query(x_list))
