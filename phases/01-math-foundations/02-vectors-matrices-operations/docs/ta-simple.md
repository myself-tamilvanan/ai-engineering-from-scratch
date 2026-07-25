<!-- Tamil-medium simplified version of en.md -->

> **For Tamil-medium students:** This version uses simple English and Tamil hints to explain AI concepts.

# vector (வெக்டர் — எண் பட்டியல்)s, Matrices & Operations

> Every neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) is just matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiplication with extra steps.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra simple idea)
**Time:** ~60 minutes

## Learning Objectives

- Build a matrix (மேட்ரிக்ஸ் — எண் மேஜை) class with element-wise operations, matrix multiplication, transpose, determinant, and inverse
- Distinguish element-wise multiplication from matrix multiplication and explain when each applies
- Implement a single dense neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) layer (`relu(W @ x + b)`) using only the from-scratch Matrix class
- Explain broadcasting rules and how bias addition works in neural network frameworks

## The Problem

You want to build a neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு). You read the code and see this:

```
output = activation(weights @ input + bias)
```

That `@` is matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiplication. The `weights` are a matrix. The `input` is a vector (வெக்டர் — எண் பட்டியல்). If you do not know what those operations do, this line is magic. If you do know, it is the entire forward pass of a layer in three operations.

Every image your model processes is a matrix (மேட்ரிக்ஸ் — எண் மேஜை) of pixel values. Every word embedding (embedding — பொருளை எண்களாக மாற்றுதல்) is a vector (வெக்டர் — எண் பட்டியல்). Every layer of every neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) is a matrix transformation. You cannot build AI systems without being fluent in matrix operations the same way you cannot write code without understanding variables.

This lesson builds that fluency from scratch.

## The Concept

### vector (வெக்டர் — எண் பட்டியல்)s: ordered lists of numbers

A vector (வெக்டர் — எண் பட்டியல்) is a list of numbers with a direction and magnitude. In AI, vectors represent data points, feature (பண்பு — தகவல் நெடுவரிசை)s, or parameter (பாரமீட்டர் — கட்டுப்பாடு)s.

```
v = [3, 4]        -- a 2D vector (வெக்டர் — எண் பட்டியல்)
w = [1, 0, -2]    -- a 3D vector
```

A 2D vector (வெக்டர் — எண் பட்டியல்) `[3, 4]` points to coordinates (3, 4) on a plane. Its length (magnitude) is 5 (the 3-4-5 triangle).

### Matrices: grids of numbers

A matrix (மேட்ரிக்ஸ் — எண் மேஜை) is a 2D grid. Rows and columns. An m x n matrix has m rows and n columns.

```
A = | 1  2  3 |     -- 2x3 matrix (மேட்ரிக்ஸ் — எண் மேஜை) (2 rows, 3 columns)
    | 4  5  6 |
```

In neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு)s, weight matrices transform input vector (வெக்டர் — எண் பட்டியல்)s into output vectors. A layer with 784 inputs and 128 outputs uses a 128x784 weight matrix (மேட்ரிக்ஸ் — எண் மேஜை).

### Why shapes matter

matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiplication has a strict rule: `(m x n) @ (n x p) = (m x p)`. The inner dimension (பரிமாணம் — கோணம்)s must match.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimension (பரிமாணம் — கோணம்)s: 784 = 784  -- valid
```

If you get a shape mismatch error in PyTorch, this is why.

### The operations map

| Operation | What it does | neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) use |
|-----------|-------------|-------------------|
| Addition | Element-wise combine | Adding bias to output |
| scalar (scalar — ஒரு எண்) multiply | Scale every element | Learning rate * gradient (சாய்வு — மாற்றத்தின் दिशा)s |
| matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiply | Transform vector (வெக்டர் — எண் பட்டியல்)s | Layer forward pass |
| Transpose | Flip rows and columns | backpropagation (பின்னாக பரப்புதல் — தவறிலிருந்து கற்றுக்கொள்ளுதல்) |
| Determinant | Single number summary | Checking invertibility |
| Inverse | Undo a transformation | Solving linear systems |
| Identity | Do-nothing matrix | Initialization, residual connections |

### Element-wise vs matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiplication

This distinction trips up beginners constantly.

Element-wise: multiply matching positions. Both matrices must be the same shape.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiplication: dot products of rows and columns. Inner dimension (பரிமாணம் — கோணம்)s must match.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Different operations, different results, different rules.

### Broadcasting

When you add a bias vector (வெக்டர் — எண் பட்டியல்) to a matrix (மேட்ரிக்ஸ் — எண் மேஜை) of outputs, the shapes do not match. Broadcasting stretches the smaller array to fit.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector (வெக்டர் — எண் பட்டியல்) across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Every modern framework does this automatically. Understanding it prevents confusion when shapes seem wrong but the code runs.

```figure
vector (வெக்டர் — எண் பட்டியல்)-projection
```

## Build It

### Step 1: vector (வெக்டர் — எண் பட்டியல்) class

```python
class vector (வெக்டர் — எண் பட்டியல்):
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"vector (வெக்டர் — எண் பட்டியல்)({self.data})"

    def __add__(self, other):
        return vector (வெக்டர் — எண் பட்டியல்)([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return vector (வெக்டர் — எண் பட்டியல்)([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar (scalar — ஒரு எண்)):
        return vector (வெக்டர் — எண் பட்டியல்)([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: matrix (மேட்ரிக்ஸ் — எண் மேஜை) class with core operations

```python
class matrix (மேட்ரிக்ஸ் — எண் மேஜை):
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"matrix (மேட்ரிக்ஸ் — எண் மேஜை)({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar (scalar — ஒரு எண்)_multiply(self, scalar):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("matrix (மேட்ரிக்ஸ் — எண் மேஜை) is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return matrix (மேட்ரிக்ஸ் — எண் மேஜை)([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: See it work

```python
A = matrix (மேட்ரிக்ஸ் — எண் மேஜை)([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = matrix (மேட்ரிக்ஸ் — எண் மேஜை).identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect to neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு)s

```python
import random

inputs = matrix (மேட்ரிக்ஸ் — எண் மேஜை)([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix (மேட்ரிக்ஸ் — எண் மேஜை)(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix (மேட்ரிக்ஸ் — எண் மேஜை)(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

This is a single dense layer: `output = relu(W @ x + b)`. Every dense layer in every neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) does exactly this.

## Use It

NumPy does everything above in fewer lines and orders of magnitude faster.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix (மேட்ரிக்ஸ் — எண் மேஜை) multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nneural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

The `@` operator in Python calls `__matmul__`. NumPy implements it with optimized BLAS routines written in C and Fortran. Same math, 100x faster.

Broadcasting in NumPy:

```python
matrix (மேட்ரிக்ஸ் — எண் மேஜை) = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy automatically broadcasts the 1D bias across both rows. This is how bias addition works in every neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) framework.

## Ship It

This lesson produces a prompt for teaching matrix (மேட்ரிக்ஸ் — எண் மேஜை) operations through geometric simple idea. See `outputs/prompt-matrix-operations.md`.

The matrix (மேட்ரிக்ஸ் — எண் மேஜை) class built here is the foundation for the mini neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு) framework we build in Phase 3, Lesson 10.

## Exercises

1. **Verify the inverse.** Multiply `A @ A.inverse_2x2()` and confirm you get the identity matrix (மேட்ரிக்ஸ் — எண் மேஜை). Try it with three different 2x2 matrices. What happens when the determinant is zero?

2. **Implement 3x3 inverse.** Extend the matrix (மேட்ரிக்ஸ் — எண் மேஜை) class to compute inverses for 3x3 matrices using the adjugate method. Test it against NumPy's `np.linalg.inv`.

3. **Build a two-layer network.** Using only your matrix (மேட்ரிக்ஸ் — எண் மேஜை) class (no NumPy), create a two-layer neural network (வலைப்பின்னல் — மூளை போன்ற எண் அமைப்பு): input (3) -> hidden (4) -> output (2). Initialize random weights, run a forward pass, and verify all shapes are correct.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| vector (வெக்டர் — எண் பட்டியல்) | "An arrow" | An ordered list of numbers. In AI: a point in high-dimension (பரிமாணம் — கோணம்)al space. |
| matrix (மேட்ரிக்ஸ் — எண் மேஜை) | "A table of numbers" | A linear transformation. It maps vectors from one space to another. |
| Matrix multiply | "Just multiply the numbers" | Dot products between every row of the first matrix and every column of the second. Order matters. |
| Transpose | "Flip it" | Swap rows and columns. Turns an m x n matrix into n x m. Critical in backpropagation (பின்னாக பரப்புதல் — தவறிலிருந்து கற்றுக்கொள்ளுதல்). |
| Determinant | "Some number from the matrix" | Measures how much the matrix scales area (2D) or volume (3D). Zero means the transformation crushes a dimension. |
| Inverse | "Undo the matrix" | The matrix that reverses the transformation. Only exists when the determinant is not zero. |
| Identity matrix | "The boring matrix" | The matrix equivalent of multiplying by 1. Used in residual connections (ResNets). |
| Broadcasting | "Magic shape fixing" | Stretching a smaller array to match a larger one by repeating along missing dimensions. |
| Element-wise | "Regular multiplication" | Multiply matching positions. Both arrays must have the same shape (or be broadcastable). |

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - visual simple idea for every operation covered here
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - the exact rules NumPy follows
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - concise reference for ML-specific linear algebra
