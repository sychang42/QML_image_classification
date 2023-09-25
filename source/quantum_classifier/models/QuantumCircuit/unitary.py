# This module contains the set of unitary ansatze that will be used to benchmark the performances of Quantum Convolutional Neural Network (QCNN) in QCNN.ipynb module
import pennylane as qml
import numpy as np 

def U_TTN(angle, wires):
    #num_wires = 2
    #num_params = 2 #int: Number of trainable parameters that the operator depends on.


    qml.RY(angle[0], wires=wires[0]) 
    qml.RY(angle[1], wires=wires[1]) 
    qml.CNOT(wires=[wires[0], wires[1]])
        
def U_RX(angle, wires):
    #num_wires = 2
    #num_params = 2 #int: Number of trainable parameters that the operator depends on.


    qml.RX(angle[0], wires=wires[0]) 
    qml.RX(angle[1], wires=wires[1]) 
    

def U_5(angle, wires):
            
    qml.RX(angle[0], wires=wires[0]) 
    qml.RX(angle[1], wires=wires[1]) 
    qml.RZ(angle[2], wires=wires[0])
    qml.RZ(angle[3], wires=wires[1])
    qml.CRZ(angle[4], wires=[wires[1], wires[0]])
    qml.CRZ(angle[5], wires=[wires[0], wires[1]])
    qml.RX(angle[6], wires=wires[0])
    qml.RX(angle[7], wires=wires[1])
    qml.RZ(angle[8], wires=wires[0])
    qml.RZ(angle[9], wires=wires[1])
        
        
    
def U_6(angle, wires):
    #num_wires = 2
    #num_params = 10 #int: Number of trainable parameters that the operator depends on.

            
    qml.RX(angle[0], wires=wires[0]) 
    qml.RX(angle[1], wires=wires[1]) 
    qml.RZ(angle[2], wires=wires[0])
    qml.RZ(angle[3], wires=wires[1])
    qml.CRX(angle[4], wires=[wires[1], wires[0]])
    qml.CRX(angle[5], wires=[wires[0], wires[1]])
    qml.RX(angle[6], wires=wires[0])
    qml.RX(angle[7], wires=wires[1])
    qml.RZ(angle[8], wires=wires[0])
    qml.RZ(angle[9], wires=wires[1])
        
            

def U_9(angle, wires):
    #num_wires = 2
    #num_params = 2 #int: Number of trainable parameters that the operator depends on.

    qml.Hadamard(wires=wires[0]) 
    qml.Hadamard(wires=wires[1]) 
    qml.CZ(wires=[wires[0], wires[1]])
    qml.RX(angle[0], wires=wires[0])
    qml.RX(angle[1], wires=wires[1])

        

def U_13(angle, wires):
    #num_wires = 2
    #num_params = 6 #int: Number of trainable parameters that the operator depends on.
            
    qml.RY(angle[0], wires=wires[0]) 
    qml.RY(angle[1], wires=wires[1]) 
    qml.CRZ(angle[2], wires=[wires[1], wires[0]])
    qml.RY(angle[2], wires=wires[0])
    qml.RY(angle[3], wires=wires[1])
    qml.CRZ(angle[5], wires=[wires[0], wires[1]])

        


def U_14(angle, wires):
    #num_wires = 2
    #num_params = 6 #int: Number of trainable parameters that the operator depends on.

            
    qml.RY(angle[0], wires=wires[0]) 
    qml.RY(angle[1], wires=wires[1]) 
    qml.CRX(angle[2], wires=[wires[1], wires[0]])
    qml.RY(angle[2], wires=wires[0])
    qml.RY(angle[3], wires=wires[1])
    qml.CRX(angle[5], wires=[wires[0], wires[1]])

        
    
    
def U_15(angle, wires):
    #num_wires = 2
    #num_params = 4 #int: Number of trainable parameters that the operator depends on.  
            
    qml.RY(angle[0], wires=wires[0]) 
    qml.RY(angle[1], wires=wires[1]) 
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(angle[2], wires=wires[0])
    qml.RY(angle[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

        

    
def U_SO4(angle, wires):
    ##num_wires = 2
    #num_params = 6 #int: Number of trainable parameters that the operator depends on.
            
    qml.RY(angle[0], wires=wires[0]) 
    qml.RY(angle[1], wires=wires[1]) 
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[2], wires=wires[0])
    qml.RY(angle[3], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[4], wires=wires[0]) 
    qml.RY(angle[5], wires=wires[1]) 

        
    
def U_SU4(angle, wires):
    #num_wires = 2
    #num_params = 15 #int: Number of trainable parameters that the operator depends on.

            
    qml.U3(angle[0], angle[1], angle[2], wires=wires[0]) 
    qml.U3(angle[3], angle[4], angle[5], wires=wires[1]) 
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(angle[6], wires=wires[0])
    qml.RZ(angle[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(angle[8], wires=wires[0]) 
    qml.CNOT(wires=[wires[0], wires[1]]) 
    qml.U3(angle[9], angle[10], angle[11], wires=wires[0]) 
    qml.U3(angle[12], angle[13], angle[14], wires=wires[1])

        
    
def Pooling_ansatz1(angle, wires):
    #num_wires = 2
    # num_params = 2 #int: Number of trainable parameters that the operator depends on
            
    qml.CRZ(angle[0], wires=[wires[1], wires[0]])
    qml.PauliX(wires=wires[1])
    qml.CRX(angle[1], wires=[wires[1], wires[0]])
       
def Pooling_ansatz2(angle, wires):
   # num_wires = 2
   # num_params = 0 #int: Number of trainable parameters that the operator depends on.

    qml.CZ(wires=[wires[0], wires[1]])

    
    
def Pooling_ansatz3(angle, wires):
    #num_wires = 2
    #num_params = 3 #int: Number of trainable parameters that the operator depends on.
            
    qml.CRot(*angle, wires=[wires[0], wires[1]])

        
    
        
def U_ZZ(angle, wires):
    #num_wires = 2
    #num_params = 7
    
    qml.RX(angle[0], wires=wires[0]) 
    qml.RZ(angle[1], wires=wires[0]) 
    qml.RX(angle[2], wires=wires[0]) 
    
    qml.RX(angle[3], wires=wires[1]) 
    qml.RZ(angle[4], wires=wires[1])
    qml.RX(angle[5], wires=wires[1])
    qml.IsingZZ(angle[6], wires=wires) 
    
    
def U_qiskit(angle, wires):
    #num_wires = 2
    #num_params = 3 #int: Number of trainable parameters that the operator depends on.


    qml.RZ(-np.pi / 2, wires=wires[1]) 
    qml.CNOT(wires = [wires[1], wires[0]]) 
    qml.RZ(angle[0], wires = wires[0])
    qml.RZ(angle[1], wires = wires[1])
    qml.CNOT(wires = [wires[0], wires[1]]) 
    
    qml.RY(angle[2], wires=wires[1]) 

