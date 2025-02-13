### To-Do:
- [ ] Pass in predetermined weights to the network
- [x] Generalize Net class so that architecture can be passed in as parameters
- [ ] Create class to iteratively build up model architecture. Should use pre-defined subset of training set to validate/compare each iteration.

### Modifying number of neurons in a layer
The network $\textrm{L1} \rightarrow \textrm{L2} \rightarrow \textrm{L3}$ consists of weights A and biases a connecting L1 to L2, and weights B and biases b connecting L2 to L3. Changing the size of L2 requires these parameters to be redefined as A', a', B', and b'. It is not possible to find an exact match for all inputs x, but when using an activation function that is asymptotically linear around small values of x, such as GELU (but not ReLU), the following approximation is valid when network parameters are small:
$B(f(Ax + a)) + b = B'(f(A'x + a')) + b'$
$\Rightarrow B(Ax + a) + b \approx B'(A'x + a') + b'$
$A' \approx (B'^TB')^{-1}B'^TBA$
$a' \approx (B'^TB')^{-1}B'^T(Ba + b - b')$
A' and a' can be determined by using the randomly assigned values of B' and b', keeping the network in an approximately similar state as before the modification.

### Adding an inner layer to the network
Again, the approximation relies on the activation function being asymptotically linear for small x and breaks down as the network parameters potentially grow large. Consider the network $\textrm{L1} \rightarrow \textrm{L2}$ having weights A and biases a. To insert a layer between these with weights and biases A', a', B', and b':
$Ax + a = B'(f(A'x+a')) + b'$
$\Rightarrow Ax + a \approx B'(A'x + a') + b'$
$A' \approx (B'^TB')^{-1}B'^TA$
$a' \approx (B'^TB')^{-1}B'^T(a-b')$

### Dropping an inner layer from the network
The approximation is again valid for conditions described above. Consider the network $\textrm{L1} \rightarrow \textrm{L2} \rightarrow \textrm{L3}$ having weights A, B and biases a, b. To condense the network with weights and biases A' and a':
$B(f(Ax + a)) + b = A'x+a'$
$\Rightarrow B(Ax + a) + b \approx A'x+a'$
$A' \approx BA$
$a' \approx Ba+b$


