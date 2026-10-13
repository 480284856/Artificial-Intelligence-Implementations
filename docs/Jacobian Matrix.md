# An introduction to Jacobian matrix
Mathematically, if you have a vector-valued function $\vec{y}=f(\vec{x})$, then the gradient of $\vec{y}$ with respect to $\vec{x}$ is a Jacobian matrix $J$:

$$
J=\begin{pmatrix}
\frac{\partial y}{\partial x_{1}} & \cdots  & \frac{\partial y}{\partial x_{n}}
\end{pmatrix} =\begin{pmatrix}
\frac{\partial y_{1}}{\partial x_{1}} & \cdots  & \frac{\partial y_{1}}{\partial x_{n}}\\
\vdots  & \ddots  & \vdots \\
\frac{\partial y_{m}}{\partial x_{1}} & \cdots  & \frac{\partial y_{m}}{\partial x_{n}}
\end{pmatrix}
$$
The Jacobian is defined this way because it answers: "If I slightly change the inputs, how will each output change?" Its arrangement lets matrix multiplication calculate those changes.

Start with the derivative you already know. For one input and one output,

$$
\Delta y \approx f'(x)\Delta x.
$$

The derivative converts a small input change into an approximate output change. The Jacobian does the same job when there are several inputs and outputs.

Suppose you have two inputs and two outputs:

$$
y_1=2x_1+3x_2,\\
y_2=-x_1+4x_2.
$$

Here, $y_1$ and $y_2$ are just two numbers calculated from $x_1$ and $x_2$. If the inputs change by $\Delta x_1$ and $\Delta x_2$, then:

$$
\Delta y_1=2\Delta x_1+3\Delta x_2,\\
\Delta y_2=-\Delta x_1+4\Delta x_2.
$$

Notice that we need four rates of change:
- 2: how $y_1$ responds to $x_1$.
- 3: how $y_1$ responds to $x_2$.
- -1: how $y_2$ responds to $x_1$.
- 4: how $y_2$ responds to $x_2$.

Put them into a matrix, and the two equations become one:
$$
\underbrace{\begin{pmatrix}\Delta y_1\\\Delta y_2\end{pmatrix}}_{\text{output change}} = \underbrace{\begin{pmatrix}2&3\\-1&4\end{pmatrix}}_{\text{Jacobian}} \underbrace{\begin{pmatrix}\Delta x_1\\\Delta x_2\end{pmatrix}}_{\text{input change}}. 
$$
This is why the rows and columns are arranged that way: each row calculates the change in one output; each column describes the effect of changing one input.

For example, changing only $x_1$ by 0.1 gives output changes (0.2, -0.1): the first column multiplied by 0.1.

When it comes to vectors, the change means the change of vector when input is changed slightly, which is represented by the movement of a point in geometry:
https://jacobian-changes.groovy-fig-8378.chatgpt.site

When x1 is changed slightly, it will cause a change of the vector $\begin{pmatrix}\Delta y_1\\\Delta y_2\end{pmatrix}_{x1}$, and also for x2: $\begin{pmatrix}\Delta y_1\\\Delta y_2\end{pmatrix}_{x2}$, so the total change is: $\begin{pmatrix}\Delta y_1\\\Delta y_2\end{pmatrix}_{x1}+\begin{pmatrix}\Delta y_1\\\Delta y_2\end{pmatrix}_{x2}$

![[Jacobian_matrix-1.png|700]]

> represent: to act on behalf of someone, stand as a symbol or sign for something.


https://chatgpt.com/c/6acdb850-c9c4-83ed-b88a-7d9d7e8f1368