# SpencerStelly_Python_Portfolio
This is my portfolio for my Python codes written in BISC450C - Computational Biology.

The sections are separated by individual projects, including:

- Using Jupyter Notebooks (1 and 2)
- Python Fundamentals
- Analyzing Data (1, 2, and 3)
- Storing Values in Lists
- Using Loops
- Using Multiple Files
- Making Choices
- Functions (1, 2, 3, and 4)
- Defensive Programming
- Transcribing DNA into RNA
- Translating RNA into Protein


## Using Jupyter Notebooks (1 and 2)

```python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year','rank','company','revenue','profit']
```


```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
len(df.profit[non_numeric_profits])
```




    369




```python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006))
```


![output_11_0](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/ea326922-74cf-42d7-a3a0-6dc3258efec8)



```python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
group_by_year = df.loc[:, ['year','revenue','profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0, y = 0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955 to 2005', 'Profit (millions)')
```


![output_16_0](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/66732c37-b9e0-4e06-8adf-255f2be495ae)



```python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (millions)')
```


![output_17_0](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/413eac70-b812-4cd9-ad6d-977280f0371e)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols= 2)
title = 'Increasein mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14,4)
fig.tight_layout()

```


![output_18_0](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/8f0538d0-f6d7-41cd-aeb5-efaa3036ec97)



## Python Fundamentals

```python
# Any pyuthon interpreter can be used as a calculator:
3 + 5 * 4
```




    23




```python
# Lets save a value to a variable
weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# Weight0 = valid
# 0weight = invalid
# weight and Weight are different
```


```python
# Types of data
# There are three common types of data
# Integer numbers
# floating point numbers
# Strings
```


```python
# Floating point number
weight_kg = 60.3
```


```python
# String comprised of letters
patient_name = "Jon Smith"
```


```python
# String comprised of numbers
patient_id = '001'
```


```python
# Use variables in python

weight_lb = 2.2 * weight_kg

print(weight_lb)
```

    132.66



```python
# Lets add a prefix to our patient id

patient_id = 'inflam_' + patient_id

print(patient_id)
```

    inflam_001



```python
# Lets combine print statements

print (patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# we can call a function inside another function

print(type(60.3))

print(type(patient_id))
```

    <class 'float'>
    <class 'str'>



```python
# We can also do calculations inside the print function

print('weight in lbs:', 2.2 * weight_kg)
```

    weight in lbs: 132.66



```python
print(weight_kg)
```

    60.3



```python
weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0

## Analyzing Data (1, 2, and 3)


```python
import numpy
```


```python
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```


```python
print(data)
```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]



```python
print(type(data))
```

    <class 'numpy.ndarray'>



```python
print(data.shape)
```

    (60, 40)



```python
print('first value in data:', data[0,0])
```

    first value in data: 0.0



```python
print('middle value in data:', data[29, 19])
```

    middle value in data: 16.0



```python
print(data[0:4, 0:10])
```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3, 36:]
```


```python
print('small is')
```

    small is



```python
print(small)
```

    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Lets use a numpy function
print(numpy.mean(data))
```

    6.14875



```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)
```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
maxval = numpy.amax(data)
minval = numpy.amin(data)
stdval = numpy.std(data)
```


```python
print(maxval)
print(minval)
print(stdval)
```

    20.0
    0.0
    4.613833197118566



```python
print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)
```

    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation: 4.613833197118566



```python
# Sometimes we want to look at variation in statistical values, such as maximum inflammation per patient, 
# or average from day one

patient_0 = data[0, :] # 0 on the first acis (rows), everything on the second (columns)

print('maximum inflammation for patient 0:', numpy.amax(patient_0))
```

    maximum inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:', numpy.amax(data[2, :]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data, axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
print(numpy.mean(data, axis = 0).shape)
```

    (40,)



```python
print(numpy.mean(data, axis = 1))
```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]


## Storing Values in Lists

```python
odds = [1, 3, 5, 7]
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
print('last element:', odds [3])
print('"-1" element:', odds [-1])
```

    first element: 1
    last element: 7
    "-1" element: 7



```python
names = ['Curie', 'Darwing', 'Turing'] # Typo in Darwin's name

print ('names is originally:', names)

names[1] = 'Darwin' #Correct the name

print('final vlaue of names:', names)
```

    names is originally: ['Curie', 'Darwing', 'Turing']
    final vlaue of names: ['Curie', 'Darwin', 'Turing']



```python
#name = 'Darwin'
#name[0] = 'd'
```


```python
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11]
    removed_element: 1



```python
odds.reverse()
print('odds after reversing:', odds)
```

    odds after reversing: [11, 7, 5, 3]



```python
odds = [3,5,7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
odds = [3,5,7]
primes = list(odds)
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7]



```python
binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes [2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)
```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last: 4



```python
date = 'Monday 4 January 20123'
day = date[0:6]
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning index:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning index: Monday



```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

sond = months[8:len(months)]
print('Using len() to get last entry:', sond)
sond = months[8:]
print('Omitting ending index:', sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Omitting ending index: ['sep', 'oct', 'nov', 'dec']

## Using Loops


```python
odds = [1,3,5,7]
```


```python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5
    7



```python
odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

    1
    3
    5



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-4-01ba67d8a9e5> in <module>
          3 print(odds[1])
          4 print(odds[2])
    ----> 5 print(odds[3])
    

    IndexError: list index out of range



```python
odds = [1,3,5,7,11,13,15,17,19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    11
    13
    15
    17
    19



```python
length = 0
names = ['Curie', 'Darwin', 'Turling']
for value in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.



```python
name = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing



```python
print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3

## Using Multiple Files

```python
import glob
```


```python
print(glob.glob('inflammation*.csv'))
```

    ['inflammation-05.csv', 'inflammation-12.csv', 'inflammation-04.csv', 'inflammation-08.csv', 'inflammation-10.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-11.csv', 'inflammation-03.csv', 'inflammation-02.csv']



```python
import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```

    inflammation-01.csv



![output_2_1](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/756fd493-8e49-46f8-bbff-f925374b2ab9)


    inflammation-02.csv



![output_2_3](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/6b30b80d-4d15-4fde-85a3-59d9b931ed1e)


    inflammation-03.csv



![output_2_5](https://github.com/Spencer-Stelly/SpencerStelly_Python_Portfolio/assets/132429194/82fede6b-2450-45b6-8f65-710e8c04adba)


## Making Choices

```python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```

    not greater
    done



```python
num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater than 100')
print('...after conditional')
```

    before conditional...
    ...after conditional



```python
num = 14

if num > 0:
    print(num, 'is positivie')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    14 is positivie



```python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else: 
    print('at least one part is false')
```

    at least one part is false



```python
if (1 > 0) or (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    at least one part is true

```python
import numpy

```


```python
data = numpy.loadtxt(fname='inflammation-01.csv', delimiter=',')
```


```python
max_inflammation_0 = numpy.amax(data, axis=0)[0]

```


```python
max_inflammation20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation20 == 20:
    print('Saspictious looking maxima!')

elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero!')

else:
    print('Seems OK!')
```

    Saspictious looking maxima!



```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter=',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
    
elif numpy.sum(numpy.amin(data, axis=0)) ==0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems ok!')
```

    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!

## Functions (1, 2, 3, and 4)

```python
fahrenheit_val = 99
celsius_val = ((fahrenheit_val - 32) *(5/9))

print(celsius_val)
```

    37.22222222222222



```python
fahrenheit_val2 = 43
celsius_val2 = ((fahrenheit_val2 - 32) *(5/9))

print(celsius_val2)
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(temp):
    # Assign the converted value to a variable
    converted = ((temp - 32) * (5/9))
    # Return the values of the new variable
    return converted
```


```python
def fahr_to_celsius(temp):
    # Return converted values more efficiently using the return function without creating
    # a new variable. this code does the same thing as the previous function but it is more
    #explicit in explaining how the return command works.
    return ((temp - 32) * (5/9))
```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')

```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print('freezing point of water in Kelvin:', celsius_to_kelvin(0.))
```

    freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('boiling point of water in Kelvin:', fahr_to_kelvin(212.0))
```

    boiling point of water in Kelvin: 373.15



```python
temp_kelving = fahr_to_kelvin(212.0)
print('Temperature in Kelvin was:', temp_kelving)
```

    Temperature in Kelvin was: 373.15



```python
temp_kelving
```




    373.15




```python
def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```

    Temperature in Fahrenheit was: 212.0
    Temperature in Kelvin was: 373.15

## Defensive Programming

```python
numbers = [1.5, 2.3, 0.7, -0.001, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
print('total is:', total)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-2-13c7d5640ddd> in <module>
          2 total = 0.0
          3 for num in numbers:
    ----> 4     assert num > 0.0, 'Data should only contain positive values'
          5     total += num
          6 print('total is:', total)


    AssertionError: Data should only contain positive values



```python
def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the formal (x0, y0, x1, y1).
    (x0, y0), and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'
    
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dy / dx
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
        
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
    
    return (0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle( (0.0, 1.0, 2.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-13-f9d109085db1> in <module>
    ----> 1 print(normalize_rectangle( (0.0, 1.0, 2.0) ))
    

    <ipython-input-12-37b8f91ce89f> in normalize_rectangle(rect)
          3     input should be of the formal (x0, y0, x1, y1).
          4     (x0, y0), and (x1, y1) define the lower left and upper right corners of the rectangle respectively."""
    ----> 5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinates



```python
print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-14-f7e0d48bdfd0> in <module>
    ----> 1 print(normalize_rectangle( (4.0, 2.0, 1.0, 5.0) ))
    

    <ipython-input-12-37b8f91ce89f> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X coordinates'
          8     assert y0 < y1, 'Invalid Y coordinates'
          9 


    AssertionError: Invalid X coordinates



```python
print(normalize_rectangle( (0.0, 0.0, 1.0, 5.0) ))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle( (0.0, 0.0, 5.0, 1.0) ))
```

    (0, 0, 1.0, 0.2)

## Transcribing DNA into RNA

```python
# Prompt the user to enter the input fasta file name

input_file_name = input("Enter the name of the input fasta file:")
```

    Enter the name of the input fasta file: SUMO.txt



```python
# SUMO.txt

>lcl|NT_033779.5_cds_NP_477411.1_1 [gene=Sumo] [locus_tag=Dmel_CG4494] [db_xref=FLYBASE:FBpp0078984,GeneID:33981] [protein=small ubiquitin like modifier, isoform A] [protein_id=NP_477411.1] [location=complement(6967028..6967300)] [gbkey=CDS]
ATGTCTGACGAAAAGAAGGGAGGTGAGACCGAGCACATCAACCTGAAGGTCCTCGGCCAGGACAACGCCG
TCGTCCAGTTCAAGATCAAGAAGCACACACCCTTGAGGAAGCTGATGAACGCCTACTGCGACCGTGCCGG
ACTCTCCATGCAGGTGGTGCGCTTCCGTTTCGACGGACAGCCCATCAACGAGAACGACACTCCGACCTCG
CTGGAGATGGAGGAGGGCGACACCATCGAGGTTTACCAGCAGCAGACTGGTGGCGCTCCATAA
```


      File "<ipython-input-5-61953103c67b>", line 3
        >lcl|NT_033779.5_cds_NP_477411.1_1 [gene=Sumo] [locus_tag=Dmel_CG4494] [db_xref=FLYBASE:FBpp0078984,GeneID:33981] [protein=small ubiquitin like modifier, isoform A] [protein_id=NP_477411.1] [location=complement(6967028..6967300)] [gbkey=CDS]
        ^
    SyntaxError: invalid syntax




```python
# Open the input fasta file and read the DNA sequence

with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()
```


```python
# Transcribe the DNA to RNA

rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  SUMO_RNA.txt



```python
# Save the RNA sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print("The RNA sequence has been saved to (output_file_name)")
```

    The RNA sequence has been saved to (output_file_name)



```python
print(rna_sequence)
```

    AUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACCUGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGUUCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCCUACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGUGCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCCGACCUCGCUGGAGAUGGAGGAGGGCGACACCAUCGAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAAAUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACCUGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGUUCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCCUACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGUGCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCCGACCUCGCUGGAGAUGGAGGAGGGCGACACCAUCGAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAA


## Translating DNA into Protein

```python
# Prompt the user to enter the input RNA file name

input_file_name = input("Enter the name of the input RNA file: ")
```

    Enter the name of the input RNA file:  SUMO_RNA.txt



```python
# SUMO_RNA.txt

AUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACCUGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGU
UCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCCUACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGU
GCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCCGACCUCGCUGGAGAUGGAGGAGGGCGACACCAUC
GAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAAAUGUCUGACGAAAAGAAGGGAGGUGAGACCGAGCACAUCAACC
UGAAGGUCCUCGGCCAGGACAACGCCGUCGUCCAGUUCAAGAUCAAGAAGCACACACCCUUGAGGAAGCUGAUGAACGCC
UACUGCGACCGUGCCGGACUCUCCAUGCAGGUGGUGCGCUUCCGUUUCGACGGACAGCCCAUCAACGAGAACGACACUCC
GACCUCGCUGGAGAUGGAGGAGGGCGACACCAUCGAGGUUUACCAGCAGCAGACUGGUGGCGCUCCAUAA
```


```python
# Open the input RNA file and read the RNA sequence

with open(input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table

codon_table = {
    'AUA':'I', 'AUC':'I', 'AUU':'I', 'AUG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACU':'T',
        'AAC':'N', 'AAU':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGU':'S', 'AGA':'R', 'AGG':'R',                
        'CUA':'L', 'CUC':'L', 'CUG':'L', 'CUU':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCU':'P',
        'CAC':'H', 'CAU':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGU':'R',
        'GUA':'V', 'GUC':'V', 'GUG':'V', 'GUU':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCU':'A',
        'GAC':'D', 'GAU':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGU':'G',
        'UCA':'S', 'UCC':'S', 'UCG':'S', 'UCU':'S',
        'UUC':'F', 'UUU':'F', 'UUA':'L', 'UUG':'L',
        'UAC':'Y', 'UAU':'Y', 'UAA':'*', 'UAG':'*',
        'UGC':'C', 'UGU':'C', 'UGA':'*', 'UGG':'W',
}
```


```python
#Translate RNA to protein

protein_sequence = ""
for i in range (0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  SUMO_Protein.txt



```python
# Save the protein sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(protein_sequence)
    print("The protein sequence has been saved to {output_file_name}")
```

    The protein sequence has been saved to {output_file_name}



```python
print(protein_sequence)
```

    MSDEKKGGETEHINLKVLGQDNAVVQFKIKKHTPLRKLMNAYCDRAGLSMQVVRFRFDGQPINENDTPTSLEMEEGDTIEVYQQQTGGAP

# Notes

This computational biology course  was my introduction to coding. Combining coding with biology (patient data, transcription, translation, COVID data, etc.) allows for real-world examples of coding while putting the vocabulary in simple-to-understand formatting. 
This course has allowed me to gather a better understanding a niche field of coding AND biology!

Personal Contact: spencerstelly@hotmail.com
