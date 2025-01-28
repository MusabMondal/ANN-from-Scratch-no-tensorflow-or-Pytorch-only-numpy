import numpy as np

# ======================================================
# 1) Base Module
# ======================================================
class Module:
    def forward(self, x):
        """
        Forward pass. 
        Should return the output for the given input `x`.
        """
        raise NotImplementedError
    
    def backward(self, grad_output):
        """
        Backward pass.
        Should return the gradient w.r.t. inputs and compute
        gradients w.r.t. internal parameters (if any).
        """
        raise NotImplementedError
    
    def parameters(self):
        """
        Return a list of parameters (numpy arrays) for this module.
        """
        return []
    
    def gradients(self):
        """
        Return a list of gradients (numpy arrays) for this module,
        in the same order as `parameters()`.
        """
        return []

# ======================================================
# 2) Layers
# ======================================================

class Linear(Module):
    """
    A fully-connected linear layer: y = xW + b
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        self.W = 0.01 * np.random.randn(in_features, out_features)
        self.b = np.zeros((1, out_features))
        
        # Gradients
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    
    def forward(self, x):
        """
        x: shape (N, in_features)
        returns: shape (N, out_features)
        """
        self.x = x  # save input for backward
        out = x.dot(self.W) + self.b
        return out
    
    def backward(self, grad_output):
        """
        grad_output: shape (N, out_features)
        returns: gradient w.r.t x, shape (N, in_features)
        """
        # Compute gradients w.r.t. parameters
        self.grad_W = self.x.T.dot(grad_output)
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input x
        grad_input = grad_output.dot(self.W.T)
        return grad_input
    
    def parameters(self):
        return [self.W, self.b]
    
    def gradients(self):
        return [self.grad_W, self.grad_b]


class ReLU(Module):
    """
    ReLU activation: ReLU(x) = max(0, x)
    """
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        grad_input = grad_output * (self.x > 0)  # derivative of ReLU
        return grad_input


class Softmax(Module):
    """
    Softmax activation. Typically used for classification outputs.
    """
    def forward(self, x):
        # for numerical stability
        max_x = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - max_x)
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out
    
    def backward(self, grad_output):
        """
        Typically, you wouldn't backprop through a raw softmax alone if combined with cross-entropy.
        However, let's implement the general gradient anyway:
        d(softmax)/dx = softmax(x) * (I - softmax(x)^T)
        but we need to carefully handle this for each sample.
        """
        batch_size, num_classes = grad_output.shape
        grad_input = np.zeros_like(grad_output)
        
        for i in range(batch_size):
            y = self.out[i].reshape(-1,1)
            jacobian = np.diagflat(y) - y.dot(y.T)  # shape (num_classes, num_classes)
            grad_input[i] = jacobian.dot(grad_output[i])
        
        return grad_input

# ======================================================
# 3) Container - Sequential
# ======================================================
class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
    
    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, grad_output):
        # backward in reverse order
        for module in reversed(self.modules):
            grad_output = module.backward(grad_output)
        return grad_output
    
    def parameters(self):
        # Collect parameters from all submodules
        params = []
        for module in self.modules:
            params += module.parameters()
        return params
    
    def gradients(self):
        # Collect gradients from all submodules
        grads = []
        for module in self.modules:
            grads += module.gradients()
        return grads

# ======================================================
# 4) Loss Function
# ======================================================
def cross_entropy_loss_with_softmax(logits, y_true):
    """
    logits: (N, num_classes) -- the outputs before or after Softmax
    y_true: (N,) integer class labels, or (N, num_classes) one-hot
    We'll assume integer class labels for clarity.
    
    We'll handle the Softmax inside this function for numerical stability.
    """
    # Convert logits to probabilities with stable softmax
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Negative log-likelihood
    N = y_true.shape[0]
    log_likelihood = -np.log(probs[np.arange(N), y_true] + 1e-15)
    loss = np.sum(log_likelihood) / N
    
    return loss, probs

def cross_entropy_backward(probs, y_true):
    """
    The gradient of the cross-entropy loss w.r.t. the logits input 
    for a combined softmax+cross-entropy is:
       dL/dlogits = (probs - y_true_one_hot) / N
    """
    N = y_true.shape[0]
    
    # Create one-hot if necessary
    # If y_true is integer shape (N,)
    y_true_one_hot = np.zeros_like(probs)
    y_true_one_hot[np.arange(N), y_true] = 1.0
    
    grad_logits = (probs - y_true_one_hot) / N
    return grad_logits

# ======================================================
# 5) Optimizer - Simple SGD
# ======================================================
class SGD:
    def __init__(self, params, grads, lr=0.01):
        self.params = params
        self.grads = grads
        self.lr = lr
    
    def step(self):
        for p, g in zip(self.params, self.grads):
            p -= self.lr * g

# ======================================================
# 6) Putting It All Together (Example)
# ======================================================
if __name__ == "__main__":
    np.random.seed(42)

    # -------------------------------
    # Create some synthetic data
    # -------------------------------
    num_samples = 300
    num_features = 2
    num_classes = 3

    X = np.random.randn(num_samples, num_features)
    y = np.random.randint(0, num_classes, size=num_samples)

    # -------------------------------
    # Define a simple network
    # [Linear -> ReLU -> Linear -> (Softmax implicitly in loss)]
    # -------------------------------
    model = Sequential(
        Linear(num_features, 16),
        ReLU(),
        Linear(16, num_classes)
        # We won't put Softmax here because we'll combine it 
        # with cross-entropy for stability
    )

    # Create an optimizer
    sgd = SGD(model.parameters(), model.gradients(), lr=0.1)

    # Training
    num_epochs = 2000
    for epoch in range(num_epochs):
        # 1. Forward pass
        logits = model.forward(X)  # shape: (N, num_classes)

        # 2. Compute loss (with integrated softmax)
        loss, probs = cross_entropy_loss_with_softmax(logits, y)

        # 3. Backward pass
        grad_logits = cross_entropy_backward(probs, y) 
        model.backward(grad_logits)  # backprop through the network

        # 4. Update parameters
        sgd.step()

        # Zero out gradients for the next iteration
        for g in model.gradients():
            g.fill(0)

        # Optional: measure accuracy
        if (epoch+1) % 200 == 0:
            predictions = np.argmax(logits, axis=1)
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("Training complete.")
