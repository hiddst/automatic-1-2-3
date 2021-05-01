# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:22:07 2021

@author: xinda
"""
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from bokeh.plotting import figure, output_file, show    

#%%

# a function to load the data

def load_tushareData(stock_id):
    import tushare as ts
    data = ts.get_hist_data(stock_id, start='2020-01-02', end='2021-04-02').iloc[::-1]
    data = data[['open', 'high', 'close', 'low', 'volume']].reset_index()
    return data


#%%
stock_id = 'hs300'
data = load_tushareData(stock_id)


#data.to_csv('hs300')
#start='2020-01-02',end='2021-04-02'
#%% SECTION 1  Compute the 'Dominant Wavelength'

# 1.1 NORMALIZATION

# a function to normalize the time series
def normalize(data, n_mv):
    # data is a pd.DataFrame containing the price time series
    #       with columns 'high' and 'low'
    
    # n_mv is number of periods used in computing the moving 
    #        average which is supposed to be close to the 
    #        dominant wave length
    n_ = math.floor(n_mv/2)


    # pp.5 (1)
    a = (data['high'] + data['low']) / 2

    # compute the moving average pp.6(2)
    b =  (a.rolling(n_ + 1).sum() + 
          a.shift(-n_).rolling(n_).sum()) / (2*n_ + 1)

    # subtract from a the moving average b 
    x = a - b
    
    return x
    

#%%
x = normalize(data, n_mv = 7).dropna()



#%%

# 1.2 CROSS-CORRELATION

# a function to compute the time series correlation w.r.t def_2, def_3
def Corr(s1, s2):
    # s1, s2 are two time series with equal length, in pd.Series format

    return (s1*s2).sum() / ((s1**2).sum()**(.5) * (s2**2).sum()**(.5)) 


# a function to get cross-correlation with time shift n 
def Corr_n(s, n):
    # s is a time series in the pd.Series format

    s_0 = s.drop(s.tail(n).index)
    s_n = s.shift(-n).dropna()
    return Corr(s_0, s_n)


# a funtion to compute the Corr_n for diffent n
def crossCorr(s, n_max): 
    # s is a time series in the pd.Series format
    # n_max the largest shifting number 
    
    corr = []
    # a for-loop get the Corr_n`s
    for n in range(1, n_max + 1, 1):
        corr.append(Corr_n(s, n))
    
    # return correlations in the list format
    return corr


#%%
corr = crossCorr(x, 100)



       
#%% SECTION 2  

# 2.1 MACD based SAR proc. (direction[])


# a function to compute the average true range
def averageTrueRange(data, signalParam):
    
    tr = pd.DataFrame()
    tr['hMl'] = data.high - data.low
    tr['hMc'] = data.high - data.shift(1).close
    tr['cMl'] = data.shift(1).close - data.low
    tr['trueRange'] = tr.max(axis = 1)
    tr['ATR'] = tr['trueRange'].rolling(math.floor(signalParam)).mean()
    
    return tr['ATR']


# a function to generate direction signals based on MACD
def macd_dir(data, timeScale):
    
    data_ = data
    # setting the parameters for MACD

    fastParam = 12 * timeScale
    slowParam = 26 * timeScale
    signalParam = 9 * timeScale

    exp1 = data_.close.ewm(span=fastParam).mean()
    exp2 = data_.close.ewm(span=slowParam).mean()
    macd = exp1 - exp2
    exp3 = macd.ewm(span=signalParam).mean()

    # compute the average true range
    data_['ATR'] = averageTrueRange(data, signalParam)

 
    # generate the direction signal 
    delta = 0.001
    data_['macd_diff'] = macd - exp3
    data_.loc[(data_.macd_diff >= delta * data_.ATR ), 'direction'] = 1
    data_.loc[(data_.macd_diff <= - delta * data_.ATR), 'direction'] = -1
    data_.direction = data_.direction.fillna(0)

    dir_ = [int(data_.iloc[0]['direction'])]
    for i in range(1, len(data_)):
        if data_.iloc[i]['direction'] != 0:
            dir_.append(data_.iloc[i]['direction'])
        else:
            dir_.append(dir_[i-1])
                
    return dir_

#%%
timeScale = 0.7
data['direction'] = macd_dir(data, timeScale)


#%% 2.2 INITIALIZATION


# a function to initialize the MinMax proc

def initialMinMax(data):
    status = [0]
    exception = [1]
    mins = []
    minsBar = []
    maxs = []
    maxsBar = []
    tempMaxBar = np.nan
    tempMinBar = np.nan

    initialMax = 0
    initialMaxBar = 0
    initialMin = 0
    initialMinBar = 0

 

    # looking for the initial extreme value 
    for i in range(1, len(data)):
        if (data.iloc[i-1].direction == 0) & (data.iloc[i].direction == 0):
            status.append(0)
            exception.append(1)
            
        elif (data.iloc[i-1].direction == 0) & (data.iloc[i].direction == 1):
            initialMin = data.iloc[:i+1].low.min()
            initialMinBar = data.iloc[:i+1].low.idxmin()
            tempMaxBar = data.iloc[initialMinBar: i+1].high.idxmax()
            
            print('initialMin:', initialMin, initialMinBar, i)
            #lastMin[i],  lastMinBar[i]= [initialMin, initialMinBar]
            maxs.append(np.nan)
            maxsBar.append(np.nan)
            mins.append(initialMin)
            minsBar.append(initialMinBar)
            status.append(data.iloc[i].direction)
            exception.append(1)
            return status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar
        
        elif (data.iloc[i-1].direction == 0) & (data.iloc[i].direction == -1):
            initialMax = data.iloc[:i+1].high.max()
            initialMaxBar = data.iloc[:i+1].high.idxmax()
            tempMinBar = data.iloc[initialMaxBar: i+1].low.idxmin()

            print('initialMax', initialMax, initialMaxBar, i)
            #lastMax[i],  lastMaxBar[i]= [initialMax, initialMaxBar]
            maxs.append(initialMax)
            maxsBar.append(initialMaxBar)
            mins.append(np.nan)
            minsBar.append(np.nan)
            status.append(data.iloc[i].direction)
            exception.append(1)
            return status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar
      

    #data.lastMax, data.lastMin = [lastMax, lastMin]
    #data.lastMaxBar, data.lastMinBar = [lastMaxBar, lastMinBar]
    
#%%    
status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar = initialMinMax(data)


#%% 2.3 Exception Detection

# a function to decide whether a exception had happened in the current step
def excep(exc_last, dir_last, dir_curr, lastMax, lastMin, high_curr, low_curr):
    # exceptional process was already active
    if exc_last == -1:
        if  ((dir_last * dir_curr == -1) |
             ((dir_last == 1) & (lastMax <= high_curr)) |
             ((dir_last == -1) & (lastMin >= low_curr)) ) :
            return 1
        else:
            return -1
        
    # check for exceptional situaltion    
    elif ((exc_last == 1) & (dir_last == dir_curr)) :
        if  (((dir_curr == 1) & (lastMin >= low_curr)) |
            ((dir_curr == -1) & (lastMax <= high_curr))) :
            return -1
        else:
            return 1
    
    else:
        return 1
        


#%% 2.4 Update the MinMax proc.




# a funtion to update the MinMax proc. after it was initialized by the function 
#   in 2.2

def updateMinMax(status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar):
      
    for i in range(len(status), len(data)):
    
        # compute Excep and Status for current stage i.e. excep[0], status[0]
        excep_curr = excep(exc_last = exception[-1], 
                       dir_last = data.iloc[i-1].direction,
                       dir_curr = data.iloc[i].direction,
                       lastMax = maxs[-1],
                       lastMin = mins[-1],
                       high_curr = data.iloc[i].high,
                       low_curr = data.iloc[i].low
                       )
    
        status_curr = excep_curr * data.iloc[i].direction
        
        if status[-1] == 1:
            if data.iloc[i].high >= data.iloc[tempMaxBar].high:
                # update tempMaxBar
                tempMaxBar = i
                
            if status_curr == -1:
                # the status has changed; fix the last max
                maxs.append(data.iloc[tempMaxBar].high)
                maxsBar.append(tempMaxBar)
            
                # initialize the new tempMinBar
                alpha = (maxsBar[-1] == minsBar[-1])
                
                if maxsBar[-1] + alpha <= i:
                    tempMinBar = data.iloc[maxsBar[-1] + alpha : i+1].low.idxmin()
                else:
                    tempMinBar = i + 1
        
        if status[-1] == -1:
            if data.iloc[i].low <= data.iloc[tempMinBar].low:
                tempMinBar = i
              
            
            if status_curr == 1:
                mins.append(data.iloc[tempMinBar].low)
                minsBar.append(tempMinBar)
                
                alpha = (maxsBar[-1] == minsBar[-1])
                
                if minsBar[-1] + alpha <= i:
                    tempMaxBar = data.iloc[minsBar[-1] + alpha : i+1].high.idxmax()
                else:
                    tempMaxBar = i + 1
            
        # update exception and status
        exception.append(excep_curr)
        status.append(status_curr)
    
    return status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar



#%%
status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar = updateMinMax(status, exception, mins, maxs, minsBar, maxsBar, tempMaxBar, tempMinBar)

# drop the nans' in the list
mins = [x for x in mins if (math.isnan(x)!=True)]
maxs = [x for x in maxs if (math.isnan(x)!=True)]
minsBar = [x for x in minsBar if (math.isnan(x)!=True)]
maxsBar = [x for x in maxsBar if (math.isnan(x)!=True)]


data['status'] = status
data['exception'] = exception


#%% 2.5 The minMaxValue series



# a function to generate the minmaxvalue series (ref: def. 2.15)
def minmaxvalue(data, minsBar, maxsBar, tempMaxBar, tempMinBar):
    
    import itertools
    a, b = maxsBar, minsBar
    minmax = []
    minmaxBar = []

    if a[0] > b[0]:
        
        # mix the two lists of bars
        minmaxBar = list(itertools.chain.from_iterable(zip(b,a)))
        if len(a) - len(b) == -1:
            minmaxBar += [b[-1]]
            
        # generate the minmaxvalue seires      
        minmax = [data.iloc[minmaxBar[0]].low] * (minmaxBar[0] + 1)
        for i in range(1, len(minmaxBar)):
            if i % 2  == 1:
                minmax += [data.iloc[minmaxBar[i]].high] * (minmaxBar[i] - minmaxBar[i-1])
            else:
                minmax += [data.iloc[minmaxBar[i]].low]  * (minmaxBar[i] - minmaxBar[i-1])
        
        # In case a bar contains minimum and maximum simultaneously,
        #     we want to see both of these levels.
        for i in range(1, len(minmaxBar)):
            if minmaxBar[i] - minmaxBar[i-1] == 0:
                if i % 2 == 1:
                    minmax[minmaxBar[i]+1] = data.iloc[minmaxBar[i]].high
                else:
                    minmax[minmaxBar[i]+1] = data.iloc[minmaxBar[i]].low
                
        # reset the initial value to be the close 
        if minmaxBar[0]>0:
            minmax[0] = data.iloc[0].close
     
                
     
    if b[0] > a[0]:
        
        minmaxBar = list(itertools.chain.from_iterable(zip(a,b)))
        if len(a) - len(b) == 1:
            minmaxBar += [a[-1]]


        minmax = [data.iloc[minmaxBar[0]].high] * (minmaxBar[0] + 1)
        for i in range(1, len(minmaxBar)):
            if i % 2 == 1:
                minmax += [data.iloc[minmaxBar[i]].low] * (minmaxBar[i] - minmaxBar[i-1])
            else:
                minmax += [data.iloc[minmaxBar[i]].high] * (minmaxBar[i] - minmaxBar[i-1])
                
        for i in range(1, len(minmaxBar)):
            if minmaxBar[i] - minmaxBar[i-1] == 0:
                if i % 2 == 1:
                    minmax[minmaxBar[i]+1] = data.iloc[minmaxBar[i]].low
                else:
                    minmax[minmaxBar[i]+1] = data.iloc[minmaxBar[i]].high     
                
        if minmaxBar[0]>0:
            minmax[0] = data.iloc[0].close
            
        
        
    # update the value after the last minbar(maxbar)    
    if  len(data) - len(minmax) > 0:
        for i in range(len(minmax), len(data)):
            if data.iloc[i].status == 1:
                minmax.append(data.iloc[tempMaxBar].high)
            elif data.iloc[i].status == -1:
                minmax.append(data.iloc[tempMinBar].low)
     
    
    
       
               
    return minmax, minmaxBar    

    

#%%

minmaxvalue, minmaxBar = minmaxvalue(data, minsBar, maxsBar, tempMaxBar, tempMinBar)
data['minmaxvalue'] = minmaxvalue

#%% SECTION 3 calibration on timeScale


#  detect the interrelation between the principle parameter (timescale) 
#         and the dominant wavelength 
#  then look for a best time scale 
#  we very the timescale from 0.6 to 6 in 0.1 steps

# a function to compute for each fixed adjustment the averaged period 
#         length given by the relevant minima and maxima which we get         
#         from the trend finder

def average_periodLength(minmaxBar):
    set_bars = list(set(minmaxBar))
    return (minmaxBar[-1] - minmaxBar[0]) * 2 /(len(set_bars)-1)


#%%

average_periodLength = average_periodLength(minmaxBar)



#%%  PLOTS

df = data
df['date'] = pd.to_datetime(df['date'])

#%%
# visualize macd
plt.plot(df.date, macd, label='MACD', color = 'grey')
plt.plot(df.date, exp3, label='Signal Line', color='cornflowerblue')
plt.legend(loc='upper left')
plt.show()


#%% visualize dir
plt.plot(df.date, df.direction, label = 'dir', color = 'maroon')
plt.legend(loc='upper left')


#%% visualize vol
df.volume.plot(title = 'vol', label = 'vol', color = 'teal')
plt.legend(loc='upper left')

#%% visualize price
df.close.plot(title = 'market_price', label = 'close', color = 'navy')
plt.legend(loc='upper left')

#%% visualize ATR
df.ATR.plot(title = 'ATR', label = 'atr', color = 'darkviolet')
plt.legend(loc='upper left')


#%% status
df.status.plot(title = 'status')


#%% exception
df.exception.plot(title = 'excep')

#%% cross-correlation

p = figure(title="cross-corr", plot_height=350, plot_width=1300)
p.line([x for x in range(1,101)], corr)
show(p)

#%% minmax series



plt.plot(df.date, df.close, label='close', color = 'dimgrey')
plt.plot(df.date, df.minmaxvalue, label='minmaxvalue', color='green')
plt.legend(loc='upper left')
plt.show()

#%% visualize the candlestick
from math import pi

inc = df.close > df.open
dec = df.open > df.close
w = 12*60*60*1000 # half day in ms

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

p = figure(x_axis_type="datetime", tools=TOOLS, plot_width=1300, title = "{} Candlestick (timeScale : {}, averagePeriodLength: {})".format(stock_id, timeScale, average_periodLength))
#p.xaxis.major_label_orientation = pi/4    #pi/4
p.grid.grid_line_alpha=.3    #0.3

p.segment(df.date, df.high, df.date, df.low, color="black")
p.vbar(df.date[inc], w, df.open[inc], df.close[inc], fill_color="#D5E1DD", line_color="black")
p.vbar(df.date[dec], w, df.open[dec], df.close[dec], fill_color="#F2583E", line_color="black")
p.line(df.date, df.minmaxvalue)


output_file("candlestick.html", title="candlestick.py ")

show(p)  # open a browser
