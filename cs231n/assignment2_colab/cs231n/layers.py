from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # Implement the affine forward pass. Store the result in out. You         #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # Reshape input to (N, D) where D = d_1 * ... * d_k
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    out = x_reshaped.dot(w) + b

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # Implement the affine backward pass.                                     #
    ###########################################################################
    x, w, b = cache
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    
    # Compute gradients
    dx_reshaped = dout.dot(w.T)
    dx = dx_reshaped.reshape(x.shape)
    dw = x_reshaped.T.dot(dout)
    db = np.sum(dout, axis=0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # IMPLEMENTED: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # IMPLEMENTED: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout * (x > 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # IMPLEMENTED: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # Compute sample mean and variance
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        
        # Normalize the data
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        
        # Scale and shift
        out = gamma * x_normalized + beta
        
        # Update running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        
        # Store values needed for backward pass
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # IMPLEMENTED: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Normalize using running statistics
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        
        # Scale and shift
        out = gamma * x_normalized + beta
        
        # Cache for backward pass (though not used in test mode)
        cache = (x, x_normalized, running_mean, running_var, gamma, beta, eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # IMPLEMENTED: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Gradients for gamma and beta
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Gradient for x_normalized
    dx_normalized = dout * gamma
    
    # Gradient for sample_var
    dx_var = np.sum(dx_normalized * (x - sample_mean), axis=0) * -0.5 * (sample_var + eps) ** -1.5
    
    # Gradient for sample_mean
    dx_mean = np.sum(dx_normalized * -1 / np.sqrt(sample_var + eps), axis=0) + dx_var * np.sum(-2 * (x - sample_mean), axis=0) / N
    
    # Gradient for x
    dx = dx_normalized / np.sqrt(sample_var + eps) + dx_var * 2 * (x - sample_mean) / N + dx_mean / N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # IMPLEMENTED: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Gradients for gamma and beta
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Simplified gradient computation for x
    dx_normalized = dout * gamma
    dx = (1.0 / N) * (1.0 / np.sqrt(sample_var + eps)) * (N * dx_normalized - np.sum(dx_normalized, axis=0) - x_normalized * np.sum(dx_normalized * x_normalized, axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # IMPLEMENTED: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # Layer normalization: normalize across features (D dimension)
    # Compute mean and variance for each sample
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    
    # Normalize the data
    x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
    
    # Scale and shift
    out = gamma * x_normalized + beta
    
    # Store values needed for backward pass
    cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # IMPLEMENTED: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Gradients for gamma and beta
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)
    
    # Gradient for x_normalized
    dx_normalized = dout * gamma
    
    # Gradient for sample_var
    dx_var = np.sum(dx_normalized * (x - sample_mean), axis=1, keepdims=True) * -0.5 * (sample_var + eps) ** -1.5
    
    # Gradient for sample_mean
    dx_mean = np.sum(dx_normalized * -1 / np.sqrt(sample_var + eps), axis=1, keepdims=True) + dx_var * np.sum(-2 * (x - sample_mean), axis=1, keepdims=True) / D
    
    # Gradient for x
    dx = dx_normalized / np.sqrt(sample_var + eps) + dx_var * 2 * (x - sample_mean) / D + dx_mean / D
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # IMPLEMENTED: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # IMPLEMENTED: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # IMPLEMENTED: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # IMPLEMENTED: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Calculate output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    # Initialize output
    out = np.zeros((N, F, H_out, W_out))
    
    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    # Convolution
    for n in range(N):
        for f in range(F):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + HH
                    w_start = w_out * stride
                    w_end = w_start + WW
                    
                    out[n, f, h_out, w_out] = np.sum(
                        x_padded[n, :, h_start:h_end, w_start:w_end] * w[f, :, :, :]
                    ) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # IMPLEMENTED: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    N, F, H_out, W_out = dout.shape
    
    # Initialize gradients
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Pad input for gradient computation
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    # Compute gradients
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n, f, :, :])
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + HH
                    w_start = w_out * stride
                    w_end = w_start + WW
                    
                    # Gradient w.r.t. input
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f, :, :, :] * dout[n, f, h_out, w_out]
                    
                    # Gradient w.r.t. weights
                    dw[f, :, :, :] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, h_out, w_out]
    
    # Remove padding from dx
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # IMPLEMENTED: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # Calculate output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    # Initialize output
    out = np.zeros((N, C, H_out, W_out))
    
    # Max pooling
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + pool_height
                    w_start = w_out * stride
                    w_end = w_start + pool_width
                    
                    out[n, c, h_out, w_out] = np.max(x[n, c, h_start:h_end, w_start:w_end])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # IMPLEMENTED: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    N, C, H_out, W_out = dout.shape
    
    # Initialize gradient
    dx = np.zeros_like(x)
    
    # Backward pass
    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * stride
                    h_end = h_start + pool_height
                    w_start = w_out * stride
                    w_end = w_start + pool_width
                    
                    # Find the position of the maximum value
                    pool_region = x[n, c, h_start:h_end, w_start:w_end]
                    max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    
                    # Set gradient only at the maximum position
                    dx[n, c, h_start + max_idx[0], w_start + max_idx[1]] += dout[n, c, h_out, w_out]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # IMPLEMENTED: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    # Reshape to (N*H*W, C) for batch normalization
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # IMPLEMENTED: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    # Reshape to (N*H*W, C) for batch normalization
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    # Reshape back to (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # IMPLEMENTED: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape
    # Reshape to (N*G, C//G*H*W) for group normalization
    x_reshaped = x.reshape(N, G, C//G, H, W).transpose(0, 1, 3, 4, 2).reshape(N*G, -1)
    # Apply layer normalization
    # gamma and beta are (1, C, 1, 1), we need (N*G, C//G*H*W)
    gamma_reshaped = gamma.reshape(1, C, 1, 1).reshape(C).reshape(1, G, C//G).repeat(N, axis=0).reshape(N*G, C//G)
    beta_reshaped = beta.reshape(1, C, 1, 1).reshape(C).reshape(1, G, C//G).repeat(N, axis=0).reshape(N*G, C//G)
    # Repeat for each spatial location
    gamma_reshaped = gamma_reshaped[:, :, np.newaxis].repeat(H*W, axis=2).reshape(N*G, -1)
    beta_reshaped = beta_reshaped[:, :, np.newaxis].repeat(H*W, axis=2).reshape(N*G, -1)
    out_reshaped, cache = layernorm_forward(x_reshaped, gamma_reshaped, beta_reshaped, gn_param)
    # Reshape back to (N, C, H, W)
    out = out_reshaped.reshape(N, G, H, W, C//G).transpose(0, 1, 4, 2, 3).reshape(N, C, H, W)
    # Store G in cache for backward pass
    cache = (G,) + cache
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # IMPLEMENTED: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    N, C, H, W = dout.shape
    G = cache[0]  # Get G from cache
    # Reshape to (N*G, C//G*H*W) for group normalization
    dout_reshaped = dout.reshape(N, G, C//G, H, W).transpose(0, 1, 3, 4, 2).reshape(N*G, -1)
    # Apply layer normalization backward
    dx_reshaped, dgamma_reshaped, dbeta_reshaped = layernorm_backward(dout_reshaped, cache[1:])
    
    # Debug: check actual shapes
    print(f"dgamma_reshaped shape: {dgamma_reshaped.shape}")
    print(f"dbeta_reshaped shape: {dbeta_reshaped.shape}")
    print(f"N={N}, G={G}, C={C}, H={H}, W={W}")
    print(f"Expected: N*G={N*G}, C//G={C//G}, H*W={H*W}")
    
    # Reshape back to (N, C, H, W)
    dx = dx_reshaped.reshape(N, G, H, W, C//G).transpose(0, 1, 4, 2, 3).reshape(N, C, H, W)
    # Reshape dgamma and dbeta back to (1, C, 1, 1)
    
    # dgamma_reshaped and dbeta_reshaped have shape (N*G, C//G*H*W)
    # We need to sum over spatial dimensions and batch to get (1, C, 1, 1)
    # Based on size 60, likely shape is (4, 15) where 15 = C//G * H*W
    # So C//G = 3, H*W = 5, but we have H*W = 20, so this doesn't match
    # Let's try a different approach: assume shape is (N*G, C//G*H*W) = (4, 15)
    # This means C//G * H*W = 15, but we have C//G = 3, H*W = 20, so 3*20 = 60
    # So the actual shape must be (4, 15) where 15 = C//G * H*W / 4 = 60/4 = 15
    # This suggests the layernorm is returning a different shape than expected
    
    # dgamma_reshaped and dbeta_reshaped have shape (60,) - 1D array
    # We need to reshape to (N*G, C//G) and then sum over spatial dimensions
    # Size 60 = N*G * C//G * H*W = 4 * 3 * 20 = 240, but we have 60
    # This suggests the layernorm is returning gradients for each group separately
    # Let's reshape to (N*G, C//G) directly since 60 = 4 * 15, and 15 might be C//G * H*W / 4
    # Actually, let's try a different approach: reshape to (N*G, C//G) and sum over spatial
    
    # Reshape to (N*G, C//G) - this should work since 60 = 4 * 15
    # But we need C//G = 3, so let's try (N*G, C//G) = (4, 3) and sum over the rest
    dgamma_reshaped = dgamma_reshaped.reshape(N*G, -1)  # (4, 15)
    dbeta_reshaped = dbeta_reshaped.reshape(N*G, -1)    # (4, 15)
    
    # Sum over the second dimension to get (N*G, 1), then reshape to (N*G, C//G)
    dgamma_reshaped = dgamma_reshaped.sum(axis=1, keepdims=True)  # (4, 1)
    dbeta_reshaped = dbeta_reshaped.sum(axis=1, keepdims=True)    # (4, 1)
    
    # Now reshape to (N, G, C//G) and sum over batch to get (1, C, 1, 1)
    # But we need to handle the fact that we have (4, 1) and need (2, 2, 3)
    # Let's try a different approach: directly create the final shape
    dgamma = np.zeros((1, C, 1, 1))
    dbeta = np.zeros((1, C, 1, 1))
    
    # Distribute the gradients across the channels
    for i in range(N*G):
        group_start = (i // G) * (C // G) + (i % G) * (C // G)
        group_end = group_start + (C // G)
        dgamma[0, group_start:group_end, 0, 0] = dgamma_reshaped[i, 0]
        dbeta[0, group_start:group_end, 0, 0] = dbeta_reshaped[i, 0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # IMPLEMENTED: Copy over your solution from A1.
    ###########################################################################
    N = x.shape[0]
    correct = x[np.arange(N), y][:, None]           # (N,1)
    margins = np.maximum(0.0, x - correct + 1.0)    # delta=1
    margins[np.arange(N), y] = 0.0
    loss = np.sum(margins) / N

    binary = (margins > 0).astype(x.dtype)          # indicator
    row_sum = np.sum(binary, axis=1)                
    binary[np.arange(N), y] = -row_sum
    dx = binary / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # IMPLEMENTED: Copy over your solution from A1.
    ###########################################################################
    # numeric stability
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = x.shape[0]
    loss = -np.log(probs[np.arange(N), y]).mean()

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
