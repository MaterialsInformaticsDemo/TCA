# TCA

Domain Adaptation via Transfer Component Analysis

mian fun : ./Python/TCA.py  ; ./Julia/TCA.jl (Julia version) 

examples : example.py

<img width="397" alt="image" src="https://github.com/MaterialsInformaticsDemo/TCA/assets/86995074/68ffa1c6-9651-4a2b-9c66-e1a28511cfda">


### example : FMO_cao et al.
### The explicit formula we proposed in the paper
``` javascript
import numpy as np

def FMO_formular(Cr, T=673.15, t = 600, DOC = 10):

    """
    Cao B, Yang S, Sun A, Dong Z, Zhang TY. 
    Domain knowledge-guided interpretive machine learning: 
    formula discovery for the oxidation behavior of ferritic-martensitic 
    steels in supercritical water. J Mater Inf 2022;2:4. 
    http://dx.doi.org/10.20517/jmi.2022.04
    
    input:
    Cr : oxidation chromium equivalent concentration (wt.%), 10.38 <= Cr <= 30.319
    Cr(wt.%) = [Cr](wt.%) + 40.3[V](wt.%) + 2.3[Si](wt.%) + 10.7[Ni](wt.%) − 1.5[Mn](wt.%)
    T : Absolute temperature (K), 673.15 <= T <= 923.15
    t : Exposure time (h), 30 <= t <= 2000
    DOC : Dissolved oxygen concentration (ppb), 0 <= DOC <= 8000
    
    output:
    the logarithm of weight gain (mg / dm2)
    """

    # Eq.(6c) in paper
    pre_factor = 0.084*(Cr**3/(T-DOC) - np.sqrt(T+DOC)) + 0.98*(Cr-DOC/T) / np.log(Cr+DOC)+8.543
    
    # Eq.(5a) in paper
    Q = 0.084*(Cr**2-Cr+DOC) / np.exp(DOC/T) + 45.09
    
    # Eq.(5b) in paper
    m = 0.323 - 0.061 * np.exp(DOC/T) / (Cr - np.sqrt(Cr) - DOC)
    
    ln_wg = pre_factor + np.log(DOC+2.17) -  Q * 1000 / 8.314 / T + m*np.log(t)
    
    return ln_wg
```    
