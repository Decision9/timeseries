from sympy import *
import numpy as np
import feather
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
data = feather.read_dataframe("house_sales.ftr")

L = [Matrix([1,2,3]),Matrix([2,1,3]),Matrix([3,2,1])]
O1 = GramSchmidt(L,True)