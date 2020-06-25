<!-- TOC -->
[TOC]
#Gradient Descend
An classic optimization method for finding the best estimators from a linear equation. In order to solve $GD$, we need to examinate the next subjects:
***
### Cost Function
$$
Error = Predicted - Actual
$$
Lets say there are a total of $N$ points in the dataset, and for all those $N$ data points, we want to minimize the error (or $Cost$).
$$ 
Cost = \frac{1}{N} 	\sum_{i=1}^{N} (Y'-Y)^2
$$
Expanded a linear function:
$$
\hat{Y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n
$$
To a vectorized form:
$$
\hat{Y}=\theta^T *x
$$
where $\theta$ is a vector of parameters weights. Solving as a normal equation of $\theta$:
$$
\hat{\theta}= (X^T*X)^{-1}*X^T*y
$$

### Derivatives
#### Power Rule
Given:
$$
f(x)=x^n
$$
Then:
$$
\frac{\partial f}{\partial x}=nx^{n-1}
$$
#### Chain Rule
If a variable $z$ depends on the variable $y$, which itself depends on the variable $x$, so that $y$ and $z$ are dependent variables, then $z$, via the intermediate variable $y$, depends on $x$ as well.
$$
\frac{\partial z}{\partial x}=\frac{\partial z}{\partial y}=\frac{\partial y}{\partial x}
$$
Example: if $y=x^2$ and $x=z^2$, then we re-write the **Chain Rule** as follows:
$$
\frac{\partial y}{\partial z}=\frac{\partial y}{\partial x}=\frac{\partial x}{\partial z}
$$
The derivatives solution for the function $f(x)=x^2$ which is $y$ w.r.t $x$, solve using the **Power Rule**:$\frac{\partial f}{\partial x}=nx^{n-1}$
$$
\frac{\partial y}{\partial x}=2x
$$
And:
$$
\frac{\partial x}{\partial z}=2z
$$
Thus:
$$
\frac{\partial y}{\partial z}=2x*2z
$$
#### Partial Derivatives
If there is a function of 2 variables: $f(x,y)$, then to find the partial derivation($\partial$) of that function $w.r.t$ one variable, treat the other variable as $constant$.
Let's say we have:
$$
f(x,y)=x^4+y^7
$$
Then the partial derivation of the function $w.r.t$ **$x$**, using the **Power Rule**:
$$
\frac{\partial f}{\partial x}=4x^3+0
$$
Treating $y$ as a $constant$. And the partial derivative function $w.r.t$ **$y$** is:
$$
\frac{\partial f}{\partial y}=0 + 7y^6
$$
Where $x$ is treated as a $constant$
### Compute Gradient Descend Algorithm
