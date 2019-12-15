# Numpy

NumPy is the fundamental package for scientific computing with Python. It contains among other things:

* a powerful N-dimensional array object
* sophisticated (broadcasting) functions
* tools for integrating C/C++ and Fortran code
* useful linear algebra, Fourier transform, and random number capabilities

Besides its obvious scientific uses, NumPy can also be used as an efficient multi-dimensional container of generic data. Arbitrary data-types can be defined. This allows NumPy to seamlessly and speedily integrate with a wide variety of databases.

## Summary

1. [Array Creation, Methods & Printing](#numpy-A)
2. [Basic Operations](#numpy-B)
3. [Universal Functions](#numpy-C)
4. [Indexing, Slicing and Iterating](#numpy-D)
5. [Shape Manipulation](#numpy-E)
6. [Linear Algebra](#numpy-F)
7. [Tricks and Tips](#numpy-G)

<a id='numpy-A'></a>
### Array Creation, Methods & Printing 

numpy.array is not the same as the Standard Python class array.array which only handles one-dimensional arrays and offers less functionality

|        Function         |             Action                |
|:-----------------------:|:---------------------------------:|
| arr_1 = np.array([...]) | Initialize numpy array            |
|      arr_1.ndim         | returns number of dimensions      |
|      arr_1.shape        | returns shape of the array        |
|      arr_1.dytpe        | returns type of elements in array |
|      arr_1.mean()       | returns mean of elements in array |
|      arr_1.sum()        | returns sum of elements in array  |
|      arr_1.min()        | returns min of array              |
|      arr_1.max()        | returns max of array              |
 

```python
# Build a two dimensional array (matrix):

np.array([(1.5,2,3), (4,5,6)])

# Build a matrix with elements of a specific type:

np.array( [ [1,2], [3,4] ], dtype=complex)
```

Often, the elements of an array are originally unknown, but its size is known. Hence, NumPy offers several functions to create arrays with initial placeholder content. These minimize the necessity of growing arrays, an expensive operation.

The function zeros creates an array full of zeros, the function ones creates an array full of ones, and the function empty creates an array whose initial content is random and depends on the state of the memory. By default, the dtype of the created array is float64.

```python
# Initialize with 4 zeros:

np.zeros(4)

# Initialize with 4 zeros along column:

np.zeros((1,4))
[Out] array([[ 0.,  0.,  0.,  0.]])

# Initialize with ones:

np.ones((1,4))
[Out] array([[ 1.,  1.,  1.,  1.]])

# Initialize an empty array:

np.empty(4)
[Out] array([[ 0.,  0.,  0.,  0.]])

# Initialize an array with a series:

np.arange(10, 40, 10) #arange(min,max,step)
[Out] array([10, 20, 30])

# Initialize with a more sophisticated series:

np.linspace(10,40,10) #create 10 ticks between 10 and 40
[Out] array([ 10.,  13.33333333,  16.66666667,  20.,23.33333333,  26.66666667,  30.,33.33333333, 36.66666667,40.])
```

<a id='numpy-B'></a>
### Basic Operations

```python
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(b)
[Out] [0 1 2 3]

# Substract two arrays:

a - b => array([20, 29, 38, 47])

# Raise an array's elements to the power of another array's elements:

a**b => array([     1,     30,   1600, 125000])

# Raise an array's elements to the power of a constant:

b**2 => array([0, 1, 4, 9])

# Add the array to an equation with numpy functions:

10*np.sin(a) => array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])

# Verify a condition:

a < 35 => array([ True,  True, False, False], dtype=bool)

# Elementwise product: 		a * b
# Matrix product: 			a.dot(b)
# Another matrix product: 	np.dot(a,b)

```

Some operations, such as += and *=, act in place to modify an existing array rather than create a new one.

When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

<a id='numpy-C'></a>
### Universal Functions

```python
# Within NumPy, these functions operate elementwise on an array, producing an array as output.

B = np.arange(3) => array([0, 1, 2])

np.exp(B) => array([ 1.        ,  2.71828183,  7.3890561 ])

np.sqrt(B) => array([ 0.        ,  1.        ,  1.41421356])

C = np.array([2., -1., 4.])
np.add(B,C) => array([ 2.,  0.,  6.])
```

<a id='numpy-D'></a>
### Indexing, Slicing and Iterating

```python
a = np.arange(10)**3
a => array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])

a[2:5] => array([ 8, 27, 64])

for i in a:
    print(i**(1/3.))
=> 
'''
0.0
1.0
2.0
3.0
4.0
5.0
6.0
7.0
8.0
9.0
'''

def f(x,y): return 10*x+y
b = np.fromfunction(f,(5,4),dtype=int)
print(b)
print(b[2,3])
print(b[0:5, 1])

'''

[[ 0  1  2  3]
 [10 11 12 13]
 [20 21 22 23]
 [30 31 32 33]
 [40 41 42 43]]
23
[ 1 11 21 31 41]
'''

```

<a id='numpy-E'></a>
### Shape Manipulation

```python
a = np.floor(10*np.random.random((3,4)))
a

a.ravel() # flatten the array

a.shape = (6, 2)
a.T
```
The reshape function returns its argument with a modified shape, whereas the ndarray.resize method modifies the array itself

If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated

```python
#Stacking arrays
a = np.floor(10*np.random.random((2,2)))
print(a)
b = np.floor(10*np.random.random((2,2)))
print(b)

np.vstack((a,b)) #vertical stack

np.hstack((a,b)) #horizontal stack

#np.hsplit() = split an array horizontally, according to columns 3 and 4: (3,4) 
#np.vsplit() = split an array vertically
```

<a id='numpy-F'></a>
### Linear Algebra

```python
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print(a)

a.transpose() # a.T works as well

np.linalg.inv(a)

y = np.array([[5.], [7.]])
print(y)
np.linalg.solve(a, y)
```

<a id='numpy-G'></a>
### Tricks and Tips

To change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:

```python
a = np.arange(30)
print(a)
a.shape = 2,-1,3  # -1 means "whatever is needed"
a.shape
```
