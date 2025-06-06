## In-progress:
- Try creating a well-conditioned random matrix by substituting singular values (see `invertible-random.py`)
- If linear approximation to gelu does not give good results, the next degree is $x^2/\sqrt{2\pi} + x/2$ to try
- Dropping a layer works very well starting from 1 inner layer but not for 2+. Probably a bug.
- Create class to iteratively build up model architecture. Should use pre-defined subset of training set to validate/compare each iteration.

## Complete:
- [x] Pass in predetermined weights to the network
- [x] Generalize Net class so that architecture can be passed in as parameters

## Adding neurons to the network
Define some evolution rate, $r_{ev}$. Populate the additional values for the weights and biases with the randomly selected values in PyTorch (Kaiming Uniform) times $r_{ev}$. Smaller values of $r_{ev}$ are more likely to produce a configuration similar to the parent network.


## Removing neurons from the network
TBD


## Dropping an inner layer from the network
The approximation is valid for asymptotically linear activation functions like gelu. Consider the network $\textrm{L1} \rightarrow \textrm{L2} \rightarrow \textrm{L3}$ having weights $A$, $B$ and biases $a$, $b$. To condense the network with weights and biases $A^\*$ and $a^\*$:

$B(f(Ax + a)) + b = A^\*x+a^\*$

$\Rightarrow B(\frac{Ax + a}{2}) + b \approx A^\*x+a^\*\textrm{, for gelu}(x) \sim \frac{x}{2}$

$A^\* \approx \frac{BA}{2}$

$a^\* \approx \frac{Ba}{2}+b$



## Adding a layer to the network
TBD


## Additional results
### Generating a random matrix with a particular determinant value
- Get SVD of a random $m \times n$ matrix $A$
- Replace the singular values with the logspace ranging from 1 to the desired condition number
- To figure out the appropriate condition number for a particular determinant:
    - $\det(A) = \prod \sigma = 1 * a * a^2 * ... * a^{N-1} = a^{N(N-1)/2}$, where $a$ is the ratio between values in the logspace and $a^{N-1}$ is the condition number
    - Then it is not hard to show that $a^{N-1}=\det(A)^{2/N}$

## Failed attempts

**Failed due to $B^{\*T}B^\*$ being ~singular for almost all randomly initialized matrices and therefore non-invertible**

### Modifying number of neurons in a layer
The network $\textrm{L1} \rightarrow \textrm{L2} \rightarrow \textrm{L3}$ consists of weights $A$ and biases $a$ connecting $\textrm{L1}$ to $\textrm{L2}$, and weights $B$ and biases $b$ connecting $\textrm{L2}$ to $\textrm{L3}$. Changing the size of $\textrm{L2}$ requires these parameters to be redefined as $A^\*$, $a^\*$, $B^\*$, and $b^\*$. It is not possible to find an exact match for all inputs $x$, but when using an activation function that is asymptotically linear around small values of $x$, such as GELU (but not ReLU), the following approximation is valid when network parameters are small:

$B(f(Ax + a)) + b = B^\*(f(A^\*x + a^\*)) + b^\*$

$\Rightarrow B(\frac{Ax + a}{2}) + b \approx B^\*(\frac{A^\*x + a^\*}{2}) + b^\*\textrm{, for gelu}(x) \sim \frac{x}{2}$

$A^\* \approx (B^{\*T}B^\*)^{-1}B^{\*T}BA$

$a^\* \approx (B^{\*T}B^\*)^{-1}B^{\*T}(Ba + 2b - 2b^\*)$

$A^\*$ and $a^\*$ can be determined by using the randomly assigned values of $B^\*$ and $b^\*$, keeping the network in an approximately similar state as before the modification.

### Adding an inner layer to the network
Again, the approximation relies on the activation function being asymptotically linear for small $x$ and breaks down as the network parameters potentially grow large. Consider the network $\textrm{L1} \rightarrow \textrm{L2}$ having weights $A$ and biases $a$. To insert a layer between these with weights and biases $A^\*$, $a^\*$, $B^\*$, and $b^\*$:

$Ax + a = B^\*(f(A^\*x+a^\*)) + b^\*$

$\Rightarrow Ax + a \approx B^\*(\frac{A^\*x + a^\*}{2}) + b^\*\textrm{, for gelu}(x) \sim \frac{x}{2}$

$A^\* \approx 2(B^{\*T}B^\*)^{-1}B^{\*T}A$

$a^\* \approx 2(B^{\*T}B^\*)^{-1}B^{\*T}(a-b^\*)$


