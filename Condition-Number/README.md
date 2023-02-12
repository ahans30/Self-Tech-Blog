# What is Condition Number in Machine Learning?

<!-- $$X = \begin{pmatrix}
1 & x_{1}\\
1 & x_{2}\\
1 & x_{3}
\end{pmatrix}$$ -->

## Introduction

> _On the title being bit of a clickbait_

Well, technically the condition number is a thing of any system of equations. In this post, we focus on linear system of equations that are typically represented by matrices and vectors to benefit from numerical linear algebra and its methods.


## Okay, let's remember matrices first

> Just in case, feel free to skip

It is a property (of matrices) that describes an aspect of a matrix transformation. Remember, a matrix is a linear operator that transforms entire vector space (i.e. a single vector $x$ and all vectors in it alike) by performing either of below operations (or their combinations thereof):

- Scale: Stretch into bigger or smaller vectors
- Skew: Stretch certain components ($x_{i_{s}}$) bigger or smaller
- Flip: Change direction of certain (or all) components
- Rotate: Make new components into old one (eg. changing basis: [eigendecomposition](https://medium.com/swlh/eigenvalues-and-eigenvectors-5fbc8b037eed) - takes benefit of this).

## So what is a condition number finally?

Condition number quantifies the degree with which a linear system is prone to abruptly change in presence of small noise or pertubutions. In general a system is said to **ill-conditioned** if it is brittle in nature and the output of it changes rapidly in presence of small perturbation.

Altenatively, when the a linear system is NOT prone to being affected by noise, it can be called **well-conditioned**.

In summary, for our system $Ax = y$, if making small changes in our data matrix $A$ or label vector $y$, how much corresponding change will it have in our solution $x$. If the corresponding is huge, our system is poorly conditioned or ill-conditioned, otherwise it is labelled as well-conditioned.

## Okay, let's detour singular matrices before we compute condition number

> Let's visualise a transformation by singular matrices

As stated above, a matrix is a representation of a linear operator and all it does is tranform a subspace into another.

For example, $A \in \mathbb{R}^{m \times n}$ transforms a vector $x \in \mathbb{R}^{n}$ to a vector $y \in \mathbb{R}^{m}$ as depicted by $Ax = y$. For ease of understanding, assume $A \in \mathbb{R}^{n \times n}$ is a square matrix.

### Singular Matrices Transformation Visualization

In order to visualise, imagine a matrix $A \in \mathbb{R}^{2 \times 2}$ in 2D plane (our original subspace), which takes a vector $x$ and transforms it into $y$ with both $x, y \in \mathbb{R}^{2}$

Now, for $Ax = y$, when our matrix $A$ is singular, it squishies the entire subspace of 2D X-Y plane (remember matrix transforms entire subspace of which $x$ is just one vector) into a 1-D straight line. Generalizing this, we have for higher $d$ dimensions, a singular matrix transforms it into a $d-1, d-2, ...$ or $0$ (a single point) dimension(s) which is called $Rank(A)$.  That is, transformation causes loss of $\ge1$ dimensions, which makes the transformation a one-way operator. For example, $f(x, y, z) = (2x, 0, z)$ (represented by $(2, 0, 1)$) is one way in that, we can go from left to right but not from right to left.

## Let us layout how to compute condition number

> Turns out, there is a whole spectrum between singular vs non-singular. The condition number simply tells us where are we in this spectrum of invertibility.

Recall that singular matrices can be defined as those with when one or more of eigenvalues are zeros.

Finally, condition number $cn$ for any matrix $A$ with eigenvalues $\lambda_{i}$ is given by $$\frac {|\lambda|_{max}} {|\lambda|_{min}}$$.

We observe that for singular matrix, the condition number shoots to $\infty$ since $min |\lambda|$ is $0$. 

## Why? Why does this makes a difference as long as our matrix is invertible?

> Let's visualise what an ill-conditioned  matrix's transformation looks like (i.e.near-singular)

- Again, imagine a 2D plane. Now, our matrix, being invertible (though nearly singular) does NOT collapse the input subspace into a 1D line BUT it places all vectors post tranformation in a very close, dense space. 
- This is to say, that vectors that may be very apart in original subspace are placed very closed to each other in post-transformation subspace. 
- This will be the case for pairs that maybe far apart or near to each other alike, since (or thus) the transformed space is highly compact and densely packed.

## How does near-singularity makes computation errors compound?

Now remember how going from right-to-left is impossible for singular matrices. Which is saying taking inverse (or inverting) a vector from transformed subspace into original is impossible. 

> A vanilla linear system is defined as $Ax = b$ with solution as $x = A^{-1}b$

**In near-singularity matrix transformatiom, we have images (i.e. transformations) of $x$ and $y$ vectors very close to each other irrespective of how close they were in the original space. So, when we take a inverse of the matrix transformation, a small $\eta$ amount of noise can abruply change the preimage computation or inverse of matrix for solution of system.**


## Let's see an example!

_The numpy code for this is this [notebook](code.ipynb)._


$$
A = \begin{bmatrix}
1 & 2\\
2 & 3.99\\
\end{bmatrix},
x = \begin{pmatrix}
x_{1} \\
x_{2}
\end{pmatrix},
b = \begin{pmatrix}
4 \\
7.999
\end{pmatrix}
$$

Now, we compute the solution for this system is

$$x = \begin{pmatrix}
x_{1} \\
x_{2}
\end{pmatrix}
= \begin{pmatrix}
2 \\
1
\end{pmatrix}$$

Let us add some noise of $10^{-3}$ in the data vector which is case for most, if not all, real life applications. 

$$
\eta = 
\begin{pmatrix}
0.001 \\
-0.001
\end{pmatrix}
$$

$$A_{noisy} = A + \eta = 
\begin{bmatrix}
1.001 & 1.999\\
2.001 & 3.998\\
\end{bmatrix}
$$

Now, we would hope that solution for this noisy/perturbed system should be close to the actual solution above.
On solving, we get

$$x_{noisy} = \begin{pmatrix}
x_{1} \\
x_{2}
\end{pmatrix}
= \begin{pmatrix}
1 \\
1.50025013
\end{pmatrix}$$

We observe the solution has abruptly and disproportionately changed showcasing brittleness of the system.

We can similarly show brittlness (even more so then!) in computation of solution when we add noise in the label vector $y$ (as shown in code).

The condition number for this matrix A comes out to be `24992.00096006945` which is way too high for such a small system.

We can easily compute condition number in `NumPy` using code below.

```python 3
import numpy as np
import numpy.linalg as LA

A = np.array([[1, 2], [2, 3.999]])
condition_number = LA.cond(A)
```
More Reading
- [Tikhonov Regularization](https://www.statisticshowto.com/well-posed-ill/)