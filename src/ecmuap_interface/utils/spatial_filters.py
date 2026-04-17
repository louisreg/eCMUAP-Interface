from numpy.typing import NDArray
from dataclasses import dataclass
import numpy as np

@dataclass
class kernel():
    __kernel: NDArray
    __label: str
    def __call__(self) -> NDArray:
        return(self.__kernel)
    
    @property
    def label(self):
        return(self.__label)

# source: Disselhorst-Klug, C., Silny, J., & Rau, G. (1997). 
# Improvement of spatial resolution in surface-EMG: a theoretical and experimental 
# comparison of different spatial filters. IEEE Transactions on Biomedical Engineering, 44(7), 567-574.

#Unit Differential Filters
unit_kernel = kernel(np.array([[1]]),"unit")
reverse_kernel = kernel(np.array([[-1]]),"reverse")

#1D Simple Differential Filters
TSD_kernel = kernel(np.array([[-1,1]]),"TSD")
LSD_kernel = kernel(np.array([[-1],[1]]),"LSD")

#1D Double Differential Filters
LDD_kernel = kernel(np.array([[-1],[2],[-1]]),"LDD")
TDD_kernel = kernel(np.array([[-1,2,-1]]),"TDD")

#2D Double Differential Filters
NDD_kernel = kernel(np.array([[0,-1,0],[-1,4,-1],[0,-1,0]]),"NDD")
IB2_kernel = kernel((1/16)*np.array([[-1,-2,-1],[-2,12,-2],[-1,-2,-1]]),"IB2")
IR_kernel = kernel((1/9)*np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),"IR")



