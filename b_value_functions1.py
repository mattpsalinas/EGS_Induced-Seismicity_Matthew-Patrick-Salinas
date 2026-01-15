#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.utils import resample
from statistics import stdev

### FUNCTION - btrad
## Calculate B-Value using maximum likelihood method (Aki 1965)
# this function takes in an array of magnitudes and outputs b value, Mc, and 
# histogrammed values that can be used for plotting. This function calculates 
# magnitude of completeness (Mc) using the Maximum curvature +0.2 method (Weimer and Wyss 2000,
# Woessner and Weimer 2005). Should you want to use a different method of calculating Mc, use 
# the btrad_manualMc function which allows the user to input their own Mc 

# INPUTS 
# magnitudes: an array, or pandas column of magnitude values 
# hist_bins: an array of numbers that signify the bins 
# delta_b: binning width (rounding) of the magnitude values, default set to 0.01 *** this is not the bin width of the 
#         histogram bins, this is how the magnitudes are rounded. (i.e. if your magnitudes are 
#         rounded to the first decimal place, like 4.1, 5.2, 2.2, etc -> delta_b = 0.1)

# OUTPUTS
# b: b-value
# Mc: magnitude of completeness
# hist_bins: bins for plotting, should be an array where min value is below min magnitude and max value is 
#       above max magnitude
# hist: histogrammed values, contains counts and bins (which is the same as the bins input) 
def btrad(magnitudes, hist_bins, delta_b = 0.01):
    
    hist = np.histogram(magnitudes, bins = hist_bins)
    Mc = hist[1][np.argmax(hist[0])] + 0.2 
    
    magnitudes = magnitudes[magnitudes >= Mc]
    b = np.log10(np.exp(1))/(np.mean(magnitudes)-(Mc-(delta_b/2)))
    
    return b, Mc, hist

### FUNCTION - btrad_manualMc
## See documentation for btrad, in this function Mc is an input rather than an output, everything else is the same

def btrad_manualMc(magnitudes, hist_bins, Mc, delta_b = 0.01):
    
    hist = np.histogram(magnitudes, bins = hist_bins)
    
    magnitudes = magnitudes[magnitudes >= Mc]
    b = np.log10(np.exp(1))/(np.mean(magnitudes)-(Mc-(delta_b/2)))
    
    return b, hist

### FUNCTION - bpos
## Calculate b+ using method described by Van der Elst et al. (2022) **NOTE that the method described in 
# the main text is for continuous magnitudes. There is another method given in the supplementary text for # binned magnitudes. 

# this function takes in an array of magnitudes and outputs b+, and histogrammed values that can be used
# for plotting. This function calculates magnitude of completeness (Mc) using the Maximum curvature +0.2  
# method (Weimer and Wyss 2000, Woessner and Weimer 2005). However, the default is to not use Mc.

# INPUTS 
# magnitudes: an array, or pandas column of magnitude values 
# Mc_prefilt: default - False, if you want to calculate Mc and prefilter out events below the completeness
#         magnitude then change to true.
# min_diff: minimum magnitude difference to consider, default is 0.2 according to the procedure of Van der #         Elst et al 2022

# OUTPUTS
# b: b+ 
# hist: histogrammed values, contains counts and bins

def bpos(magnitudes, min_diff = 0.2, Mc_prefilt = False):
    if Mc_prefilt == True:
    
        hist = np.histogram(magnitudes, bins = np.arange(-2, 6, 0.1))
        Mc = hist[1][np.argmax(hist[0])] + 0.2 
        magnitudes = magnitudes[magnitudes >= min_diff]
        
    mag_diffs = np.diff(magnitudes)
    bins = np.arange(0, max(np.round(mag_diffs, 1))+0.1, 0.1)
    hist = np.histogram(mag_diffs, bins = bins)

    mag_diffs = mag_diffs[mag_diffs >= min_diff]
    b = np.log10(np.exp(1))/(np.mean(mag_diffs)-min_diff)

    return hist, b

### FUNCITON - bootstrap 
# Calculate the 95% confidence interval and standard deviation of N bootstrap iterations. The built-in
# functions are the same as above 

# INPUTS
# magnitudes: an array, or pandas column of magnitude values 
# hist_bins: bins for plotting, should be an array where min value is below min magnitude and max value is 
#       above max magnitude
# N: number of iterations to run for bootstrapping
# mode: 'btrad' for maximum- likelihood bvalue, 'bpos' for b+ method
# see above for descriptions of default values 

# OUTPUTS 
# b_low: lower limit of 95% confidence interval
# b_high: upper limit of 95% confidence interval
# std:> standard deviation of N b-values

def bootstrap(magnitudes, hist_bins, N, mode, delta_b = 0.01, min_diff = 0.2, Mc_prefilt = False):
    
    b_list = np.zeros(N)
    
    if mode == 'btrad':
    
        for i in range(N):
            sample = resample(magnitudes)
            
            hist = np.histogram(sample, bins = hist_bins)
            Mc = hist[1][np.argmax(hist[0])] + 0.2 

            sample = sample[sample >= Mc]
            b = np.log10(np.exp(1))/(np.mean(sample)-(Mc-(delta_b/2)))

            
            b_list[i] = b
            
    elif mode == 'bpos':
        
        for i in range(N):
            
            if Mc_prefilt == True:
                    
                hist = np.histogram(magnitudes, bins = hist_bins)
                Mc = hist[1][np.argmax(hist[0])] + 0.2 
                magnitudes = magnitudes[magnitudes >= Mc]

            mag_diffs = np.diff(magnitudes)
            mag_diffs = resample(mag_diffs)
            #sample = np.diff(resample(magnitudes))
            
            hist = np.histogram(mag_diffs, bins = hist_bins)

            mag_diffs = mag_diffs[mag_diffs >= min_diff]
            b = np.log10(np.exp(1))/(np.mean(mag_diffs)-min_diff)
            
            b_list[i] = b
            
    b_list=np.absolute(b_list)
    b_low = np.percentile(b_list, 2.5)
    b_high = np.percentile(b_list, 97.5)
    std = np.std(b_list, dtype=np.float64) if len(b_list) > 0 else 0  # or 0
    
    return [b_low, b_high, std]
