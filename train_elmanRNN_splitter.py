# -*- coding: utf-8 -*-
""" 
Pilot for using Elman RNN for (supervised) word segmentation.
=====================================

Task: Detect the amplitudes

"""


from functions import *
import neurolab as nl
import numpy as np
import argparse
import sys




if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Trains a RNN splitter, based on a training corpus where space indicates splits')
    parser.add_argument('corpus', metavar='corpus', type=str,
                   help='The corpus')
    parser.add_argument('-hidden', metavar='hidden', type=int, default = 1,
                   help='size of the hidden layer (default=1)') 
    parser.add_argument('-epochs', metavar='epochs', type=int, default = 1,
                   help='number of training epochs')    
    args = parser.parse_args()
    
    #thisisanexample
    #100010101000000               
    #---------------
    #this is an example

    # Create train samples
    i1 = np.sin(np.arange(0, 20))
    i2 = np.sin(np.arange(0, 20)) * 2
    
    t1 = np.ones([1, 20])
    t2 = np.ones([1, 20]) * 2
    
    input = np.array([i1, i2, i1, i2]).reshape(20 * 4, 1)
    target = np.array([t1, t2, t1, t2]).reshape(20 * 4, 1)
    
    # Create network with n layers
    
    char_dict = create_char_dict(args.corpus)
    input_size = len(char_dict)
    net = nl.net.newelm([[-1,1] for n in range(0,input_size)], [5, 2], [nl.trans.TanSig(), nl.trans.TanSig()])
    net.errorf = nl.error.MSE()
    # Set initialized functions and init
    net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    net.layers[1].initf= nl.init.InitRand([-0.1, 0.1], 'wb')
    #net.init()


    # Train network
    for epoch in range(0,args.epochs):
        with open(args.corpus) as text:
            for line in text:
                input = np.array([[1],[0],[1],[0],[1],[0]])
                target = np.array([[0],[1],[0],[1],[0],[1]])
                                
                target = convertToLabels(line)
                input = translate_line(line, char_dict)
                #print input
                #print target
                #for i in range(input_size, input_size+
                
                net.layers[0].inp[1] = 0
                error = net.train(input, target, epochs=100, show=100, goal=-2,lr=0.001)
                
          
    # Simulate network
    with open(args.corpus) as f:
        i = 0
        for line in f:
            if i ==0:
                output = net.sim(translate_line(line.replace(' ',''), char_dict))
                #print output
                print len(line.replace(' ',''))
                print decode_line(line.replace(' ',''),output)
                
    #print vars(net)
    #print vars(net.layers[0])
    #net.layers[0].np['w'] = np.array([1.,2.,3.])
    #print
    #print vars(net.layers[0])
    #print net.layers[0].np
    #print len(net.layers[1])    
#    print dir(net.layers)
#    print dir(net.layers[0])
#    print dir(net.ci.real)
    
    
    
    
    
    exit()
    # Plot result
    import pylab as pl
    pl.subplot(211)
    pl.plot(error)
    pl.xlabel('Epoch number')
    pl.ylabel('Train error (default MSE)')

    pl.subplot(212)
    pl.plot(target.reshape(80))
    pl.plot(output.reshape(80))
    pl.legend(['train target', 'net output'])
    pl.show()
