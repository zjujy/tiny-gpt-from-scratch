"""
Tiny GPT From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_vocab
def build_vocab(text):
    """Return a sorted list of unique characters in text."""
    # TODO: return a sorted list of every unique character in text
    return sorted(set(text))

# Step 2 - build_stoi
def build_stoi(vocab):
    """Return a dict mapping each character in vocab to its index."""
    # TODO: map each character in vocab to its integer position
    return {x: i for i, x in enumerate(vocab)}

# Step 3 - build_itos
def build_itos(vocab):
    """Return a dict mapping each index 0..len(vocab)-1 to its character."""
    # TODO: build an int-to-string lookup from the vocab list
    return {i: x for i, x in enumerate(vocab)}

# Step 4 - encode_char
def encode_char(ch, stoi):
    """Return the integer token id for a single character ch using stoi."""
    # TODO: look up ch in the stoi mapping and return its id
    return stoi[ch]

# Step 5 - encode_string
def encode_string(text, stoi):
    """Encode a full string into a list of token ids using stoi."""
    # TODO: map each char in text through stoi (via encode_char) into a list of ids
    return [stoi[c] for c in text]

# Step 6 - decode_int
def decode_int(token_id, itos):
    """Return the single character mapped to token_id by itos."""
    # TODO: look up the character for token_id in the itos dict
    return itos[token_id]

# Step 7 - decode_ids
def decode_ids(ids, itos):
    """Decode a list of token ids into a string using itos."""
    # TODO: map each id through decode_int and join the characters into one string.
    return ''.join([itos[id] for id in ids])

# Step 8 - make_1d_array
import numpy as np

def make_1d_array(values):
    """Create a 1D NumPy array from a Python list of numbers."""
    # TODO: convert the input list into a 1D numpy ndarray
    return np.asarray(values)

# Step 9 - get_array_shape
import numpy as np

def get_array_shape(arr):
    """Return the shape tuple of a NumPy array."""
    # TODO: return the shape of arr
    return np.shape(arr)

# Step 10 - get_array_dtype
import numpy as np

def get_array_dtype(arr):
    """Return the dtype of a NumPy array."""
    # TODO: return the dtype attribute of arr
    return arr.dtype

# Step 11 - make_2d_zeros
import numpy as np

def make_2d_zeros(rows, cols):
    """Return a 2D NumPy array of zeros with shape (rows, cols)."""
    # TODO: allocate a (rows, cols) array of zeros and return it
    return np.zeros((rows, cols))

# Step 12 - make_2d_random
import numpy as np

def make_2d_random(rows, cols, seed):
    """Return a (rows, cols) array of uniform floats in [0, 1) seeded by `seed`."""
    # TODO: build a seeded RNG and draw a (rows, cols) uniform sample in [0, 1).
    rng = np.random.default_rng(seed)
    return rng.random((rows, cols))

# Step 13 - index_element
def index_element(arr, i, j):
    """Return the scalar element at position (i, j) of a 2D array."""
    # TODO: return the value at row i, column j of arr
    return arr[i, j]

# Step 14 - slice_row
import numpy as np

def slice_row(arr, i):
    """Return row i of a 2D array as a 1D view."""
    # TODO: return the i-th row of arr as a 1D array of shape (C,)
    return arr[i, :]

# Step 15 - slice_column
import numpy as np

def slice_column(arr, j):
    """Return column j of a 2D array as a 1D array of length R."""
    # TODO: index into arr to extract the j-th column as a 1D array.
    return arr[:, j]

# Step 16 - slice_subblock
import numpy as np

def slice_subblock(arr, r0, r1, c0, c1):
    """Return the sub-block arr[r0:r1, c0:c1] of a 2D array."""
    # TODO: return the rectangular sub-block of arr bounded by rows [r0,r1) and cols [c0,c1).
    return arr[r0: r1, c0: c1]

# Step 17 - elementwise_add
import numpy as np

def elementwise_add(a, b):
    """Return the elementwise sum of two same-shape arrays."""
    # TODO: return a new array whose entries are the pairwise sums of a and b
    return a + b

# Step 18 - elementwise_multiply
import numpy as np

def elementwise_multiply(a, b):
    """Return the elementwise product of two same-shape arrays."""
    # TODO: compute the elementwise (Hadamard) product of a and b
    return a * b

# Step 19 - scalar_broadcast_add
import numpy as np

def scalar_broadcast_add(arr, scalar):
    """Return a new array equal to arr with scalar added to every element."""
    # TODO: add a Python scalar to every element of an array via broadcasting
    return arr + scalar

# Step 20 - vector_matrix_broadcast_add
import numpy as np

def vector_matrix_broadcast_add(matrix, vector):
    """Add a 1D vector to each row of a 2D matrix via broadcasting."""
    # TODO: return matrix + vector broadcast across rows
    return matrix + vector

# Step 21 - array_exp
import numpy as np

def array_exp(arr):
    """Return the elementwise exponential of arr."""
    # TODO: apply elementwise exponential to arr and return the result
    return np.exp(arr)

# Step 22 - array_log
import numpy as np

def array_log(arr):
    """Return the elementwise natural log of arr (assumes arr > 0)."""
    # TODO: apply elementwise natural log to arr and return the result
    return np.log(arr)

# Step 23 - sum_all
import numpy as np

def sum_all(arr):
    """Return the sum of every element of arr as a scalar."""
    # TODO: collapse every element of arr into a single scalar total
    return np.sum(arr).item()

# Step 24 - sum_axis0
import numpy as np

def sum_axis0(arr):
    """Sum a 2D array along axis 0, collapsing rows into a 1D vector of column sums."""
    # TODO: reduce the row dimension of arr so the result has shape (C,).
    return sum(arr)

# Step 25 - sum_axis1
import numpy as np

def sum_axis1(arr):
    """Sum a 2D array along axis 1, returning a 1D array of row sums."""
    # TODO: collapse the column dimension by summing each row
    return np.sum(arr, axis=1)

# Step 26 - max_along_axis
import numpy as np

def max_along_axis(arr, axis):
    """Return the maximum of arr along the given axis, with that axis removed."""
    # TODO: compute the maximum value of arr along the given axis
    return np.max(arr, axis=axis)

# Step 27 - matmul
import numpy as np

def matmul(a, b):
    """Return the matrix product a @ b for 2D arrays a (M,K) and b (K,N)."""
    # TODO: compute the matrix product of a and b
    return a @ b

# Step 28 - transpose_matrix
def transpose_matrix(arr):
    """Return the transpose of a 2D array."""
    # TODO: return the transpose of arr using the .T attribute
    return np.transpose(arr)

# Step 29 - sum_keepdims
import numpy as np

def sum_keepdims(arr, axis):
    """Sum along `axis` while keeping that dimension as size 1."""
    # TODO: sum along the given axis preserving the reduced dim as size 1
    return np.sum(arr, axis=axis, keepdims=True)

# Step 30 - naive_softmax_1d
import numpy as np

def naive_softmax_1d(logits):
    """Compute softmax of a 1D logits vector via the direct exp/sum formula."""
    # TODO: exponentiate the logits, then divide by their total sum
    logits = np.array(logits)
    exp_logits = array_exp(logits)
    sum_exp = sum_all(exp_logits)
    return exp_logits / sum_exp

# Step 31 - softmax_overflow_demo
import math

def softmax_overflow_demo(large_value):
    """Show that naive exp overflows on a large logit.

    Return {'naive_exp': float, 'overflowed': bool}.
    """
    # TODO: exponentiate large_value via array_exp and report whether it is inf.
    logits = np.array([large_value])
    exp_logits = array_exp(logits)
    naive_exp = float(exp_logits[0])
    overflowed = math.isinf(naive_exp)
    return {
        "naive_exp": naive_exp,
        "overflowed": overflowed
    }

# Step 32 - stable_softmax_1d
import numpy as np

def stable_softmax_1d(logits):
    """Numerically stable softmax over a 1D logits vector."""
    # TODO: subtract the max before exponentiating, then normalize.
    max_val = max_along_axis(logits, axis=0)
    shifted_logits = logits - max_val
    exp_logits = array_exp(shifted_logits)
    sum_exp = sum_all(exp_logits)
    return exp_logits / sum_exp

# Step 33 - stable_softmax_2d_rowwise
import numpy as np

def stable_softmax_2d_rowwise(logits):
    """Row-wise numerically stable softmax of a 2D logits array."""
    # TODO: turn each row of logits into a probability distribution without overflowing
    
    row_max = max_along_axis(logits, axis=1)
    row_max = row_max.reshape(-1, 1)

    shifted_logits = logits - row_max
    exp_logits = array_exp(shifted_logits)

    sum_exp = sum_keepdims(exp_logits, axis=1)
    return exp_logits / sum_exp

# Step 34 - read_text_file
def read_text_file(text_blob):
    """Return text_blob unchanged after validating it is a non-empty string."""
    # TODO: validate that text_blob is a non-empty str and return it as the corpus string
    if not isinstance(text_blob, str):
        raise TypeError(f"Expected a string, but got {type(text_blob).__name__}")
    if len(text_blob) == 0:
        raise ValueError("Input text cannot be empty")
    return text_blob

# Step 35 - encode_corpus_to_int_array
def encode_corpus_to_int_array(text, stoi):
    """Convert the corpus string into a 1D NumPy int64 array of token ids."""
    # TODO: map every character in text through stoi and return as a 1D int64 array
    return np.array(encode_string(text, stoi), dtype=np.int64)

# Step 36 - pick_split_point
def pick_split_point(n, train_frac):
    """Return integer split index so data[:idx] is train and data[idx:] is val."""
    # TODO: compute the integer split index from n and train_frac
    return int(n * train_frac)

# Step 37 - slice_train_and_val
def slice_train_and_val(data, split_idx):
    """Split a 1D token-id array into (train, val) at split_idx."""
    # TODO: return (data[:split_idx], data[split_idx:])
    return data[:split_idx], data[split_idx:]

# Step 38 - pick_block_size
def pick_block_size(default_size):
    """Return the context length (block_size) for training windows."""
    # TODO: return an integer block size, at least 1, derived from default_size
    return max(1, default_size)

# Step 39 - slice_x_at_offset
import numpy as np

def slice_x_at_offset(data, i, block_size):
    """Return the input window data[i : i + block_size]."""
    # TODO: extract a single input window of length block_size starting at index i
    return data[i: i + block_size]

# Step 40 - slice_y_at_offset
import numpy as np

def slice_y_at_offset(data, i, block_size):
    """Return the target window of length block_size starting at i+1."""
    # TODO: extract the target window Y = data[i+1 : i+1+block_size] shifted by one.
    return data[i + 1: i + 1 + block_size]

# Step 41 - sample_random_batch_offsets
import numpy as np

def sample_random_batch_offsets(data_len, block_size, batch_size, rng):
    """Sample batch_size random valid starting offsets for (block_size+1)-windows."""
    # TODO: sample batch_size offsets in the valid range for a (block_size+1)-window.
    return rng.integers(low=0, high=data_len - block_size, size=batch_size, dtype=np.int64)

# Step 42 - stack_x_batch
import numpy as np

def stack_x_batch(data, offsets, block_size):
    """Stack per-offset X windows into a 2D batch matrix of shape (B, block_size)."""
    # TODO: for each offset, take a length-block_size slice of data and stack them as rows
    rows = [slice_x_at_offset(data, int(i), block_size) for i in offsets]
    return np.stack(rows)

# Step 43 - stack_y_batch
import numpy as np

def stack_y_batch(data, offsets, block_size):
    """Stack per-offset Y windows into a 2D (B, block_size) target matrix."""
    # TODO: for each offset, take the length-block_size slice starting at i+1 and stack rows
    rows = [slice_y_at_offset(data, i, block_size) for i in offsets]
    return np.stack(rows)

# Step 44 - get_batch
def get_batch(data, block_size, batch_size, rng):
    # TODO: package one training batch (X, Y) of shape (batch_size, block_size) from data using rng.
    offsets = sample_random_batch_offsets(len(data), block_size, batch_size, rng)
    X = stack_x_batch(data, offsets, block_size)
    y = stack_y_batch(data, offsets, block_size)
    return X, y

# Step 45 - allocate_count_matrix
import numpy as np

def allocate_count_matrix(vocab_size):
    """Allocate a (V, V) integer zero matrix for bigram counts."""
    # TODO: return a (vocab_size, vocab_size) integer array of zeros.
    return make_2d_zeros(vocab_size, vocab_size).astype(np.int64)

# Step 46 - loop_fill_counts
import numpy as np

def loop_fill_counts(n_matrix, data):
    """Increment n_matrix[curr, next] for every consecutive pair in data."""
    # TODO: walk consecutive (current, next) pairs in data and add 1 to the matching cell
    for i in range(len(data) - 1):
        n_matrix[data[i], data[i + 1]] += 1
    return n_matrix

# Step 47 - vectorize_counts_add_at
import numpy as np

def vectorize_counts_add_at(vocab_size, data):
    """Build (V, V) bigram counts from a 1D id array using vectorized scatter-add."""
    # TODO: allocate counts, then scatter-add 1 at each (data[:-1], data[1:]) pair
    n_matrix = allocate_count_matrix(vocab_size)
    np.add.at(n_matrix, (data[:-1], data[1:]), 1)
    return n_matrix

# Step 48 - add_one_smoothing
import numpy as np

def add_one_smoothing(n_matrix):
    """Return n_matrix with every entry incremented by 1 (Laplace smoothing)."""
    # TODO: apply +1 Laplace smoothing to the bigram count matrix
    return n_matrix + 1

# Step 49 - row_sums_of_counts
def row_sums_of_counts(n_matrix):
    """Return per-row sums of n_matrix with shape (V, 1)."""
    # TODO: compute per-row sums of the count matrix as a column vector for normalization.
    return sum_keepdims(n_matrix, axis=1)

# Step 50 - normalize_counts_to_probs
def normalize_counts_to_probs(n_matrix):
    """Normalize a (V, V) count matrix into a row-stochastic probability matrix."""
    # TODO: divide each row of n_matrix by its row sum to produce probabilities
    return n_matrix / row_sums_of_counts(n_matrix)

# Step 51 - sample_next_token
def sample_next_token(p_matrix, current_id, rng):
    """Sample the next token id from P[current_id] using rng."""
    # TODO: draw one categorical sample from the row of p_matrix at current_id
    probs = p_matrix[current_id]
    return int(rng.choice(len(probs), p=probs))

# Step 52 - generate_sequence
def generate_sequence(p_matrix, start_id, length, rng):
    """Autoregressively sample `length` token ids from a bigram matrix, starting with `start_id`."""
    # TODO: build a length-L int array starting at start_id, then sample each next id from p_matrix
    out = np.empty(length, dtype=np.int64)
    out[0] = start_id
    for i in range(1, length):
        out[i] = sample_next_token(p_matrix, int(out[i - 1]), rng)
    return out

# Step 53 - decode_generated_sequence
def decode_generated_sequence(ids, itos):
    """Decode a generated 1D array/list of token ids into a string via itos."""
    # TODO: turn ids into a readable string using itos
    return decode_ids(ids, itos)

# Step 54 - log_prob_of_pair
def log_prob_of_pair(p_matrix, current_id, next_id):
    """Return the log probability of a single (current, next) bigram."""
    # TODO: pick out P[current_id, next_id] and return its natural log
    return array_log(index_element(p_matrix, current_id, next_id))

# Step 55 - sum_negative_log_probs
def sum_negative_log_probs(p_matrix, data):
    # TODO: sum the negative log probabilities of all consecutive bigrams in data
    return -np.sum([log_prob_of_pair(p_matrix, data[i], data[i + 1]) for i in range(len(data) - 1)])

# Step 56 - average_nll
def average_nll(p_matrix, data):
    # TODO: return mean negative log likelihood per bigram over consecutive pairs in data.
    return sum_negative_log_probs(p_matrix, data) / (len(data) - 1)

# Step 57 - initialize_w_random
import numpy as np

def initialize_w_random(vocab_size, rng):
    """Return a (vocab_size, vocab_size) float64 matrix of N(0,1) samples drawn from rng."""
    # TODO: sample a (vocab_size, vocab_size) array of standard normal values using rng
    return rng.standard_normal((vocab_size, vocab_size), dtype=np.float64)

# Step 58 - scale_w_small
import numpy as np

def scale_w_small(w_matrix, scale):
    """Return w_matrix scaled by the given small factor."""
    # TODO: return a new array equal to w_matrix multiplied by scale
    return scale * w_matrix

# Step 59 - one_hot_encode_batch
import numpy as np

def one_hot_encode_batch(ids, vocab_size):
    """Convert a 1D array of token ids into a (N, vocab_size) one-hot matrix."""
    # TODO: allocate an (N, vocab_size) zero matrix and set one 1 per row at ids[i]
    arr = make_2d_zeros(len(ids), vocab_size)
    for i in range(len(ids)):
        arr[i][ids[i]] = 1
    return arr

# Step 60 - forward_logits_onehot
def forward_logits_onehot(onehot, w_matrix):
    # TODO: compute logits for the neural bigram model as the matrix product of one-hot inputs and W.
    return matmul(onehot, w_matrix)

# Step 61 - observe_lookup_equivalence
import numpy as np

def observe_lookup_equivalence(w, ids):
    """Show that one-hot @ W equals W[ids] for a small example.
    Returns a dict with keys 'onehot_result' and 'index_result'.
    """
    # TODO: compute logits two ways and return both in a dict
    vocab_size = w.shape[1]
    onehot = one_hot_encode_batch(ids, vocab_size)
    onehot_result = forward_logits_onehot(onehot, w)
    index_result = w[ids]
    return {
        "onehot_result": onehot_result,
        "index_result": index_result,
    }

# Step 62 - forward_logits_lookup
def forward_logits_lookup(w, ids):
    """Return logits (B, V) by gathering rows of w at positions ids."""
    # TODO: return the logits for a batch of token ids by direct row lookup into W.
    return w[ids]

# Step 63 - logits_to_probs_rowwise
def logits_to_probs_rowwise(logits):
    # TODO: convert a (B, V) logits matrix into a row-wise probability matrix
    return stable_softmax_2d_rowwise(logits)

# Step 64 - gather_correct_token_probs
def gather_correct_token_probs(probs, targets):
    """Return probs[i, targets[i]] for each i, shape (B,)."""
    # TODO: pick out the probability assigned to the correct next token for each batch row
    batch_size = probs.shape[0]
    rows = np.arange(batch_size)
    return probs[rows, targets]

# Step 65 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, targets):
    """Mean negative log-likelihood over a batch."""
    # TODO: gather correct-token probs, take log, average the negatives
    return -np.mean(array_log(gather_correct_token_probs(probs, targets)))

# Step 66 - derive_dlogits_on_paper
def derive_dlogits_on_paper():
    """Return a string summarizing the derivation of dL/dlogits for mean cross-entropy."""
    # TODO: return a short written derivation ending in dL/dlogits = (probs - onehot(targets)) / B
    return (
        "Gradient of mean cross-entropy w.r.t. logits\n"
        "---------------------------------------------\n"
        "Let B = batch size, V = vocab size.\n"
        "Logits z: (B, V).  Softmax probs p = softmax(z) with\n"
        "    p[b, j] = exp(z[b, j]) / sum_k exp(z[b, k]).\n"
        "Targets y: (B,) integer class ids; onehot(y) has shape (B, V).\n"
        "\n"
        "Per-example loss (negative log-likelihood of the true class):\n"
        "    L_b = -log p[b, y[b]]\n"
        "\n"
        "Mean loss over the batch:\n"
        "    L = (1 / B) * sum_b L_b\n"
        "\n"
        "Step 1 — derivative of the per-example NLL w.r.t. the logits.\n"
        "Using the softmax Jacobian d p[b, j] / d z[b, i] =\n"
        "    p[b, j] * (delta_{ij} - p[b, i]),\n"
        "the chain rule gives the classic result\n"
        "    dL_b / dz[b, :] = p[b, :] - onehot(y[b]).\n"
        "Intuition: gradient pushes each predicted probability toward the\n"
        "target (subtract 1 at the true class, 0 elsewhere).\n"
        "\n"
        "Step 2 — average over the batch (the 1/B from the mean loss):\n"
        "    dL / dz = (1 / B) * sum_b (p[b, :] - onehot(y[b])).\n"
        "\n"
        "Final formula (shape (B, V), matching the logits):\n"
        "    dL/dlogits = (probs - onehot(targets)) / B\n"
    )

# Step 67 - compute_dlogits
def compute_dlogits(probs, targets):
    """Gradient of mean cross-entropy w.r.t. logits. probs: (B,V), targets: (B,)."""
    # TODO: return dL/dlogits of shape (B, V) averaged over the batch.
    batch_size, vocab_size = probs.shape
    onehot = one_hot_encode_batch(targets, vocab_size)
    return (probs - onehot) / batch_size

# Step 68 - derive_dw_on_paper
def derive_dw_on_paper():
    """Return a short written derivation of dL/dW for the lookup-as-matmul forward."""
    # TODO: return a fixed multi-line string describing the scatter-add gradient.
    return (
        "Forward: logits = onehot(ids) @ W, equivalently logits[b] = W[ids[b]].\n"
        "Shapes: ids (B,), onehot O (B, V), W (V, D), logits (B, D), dlogits (B, D).\n"
        "Chain rule: dL/dW = O.T @ dlogits, shape (V, D).\n"
        "Since O has a single 1 per row at column ids[b], O.T @ dlogits sums rows of dlogits into rows of dW.\n"
        "Row v of dW equals the sum of dlogits[b] over all b with ids[b] == v.\n"
        "Implementation: scatter-add dlogits rows into dW at indices ids."
    )

# Step 69 - compute_dw_scatter_add
import numpy as np

def compute_dw_scatter_add(ids, dlogits, vocab_size):
    """Scatter-add dlogits rows into dW at positions given by ids."""
    # TODO: build a (vocab_size, vocab_size) dW and accumulate dlogits[b] into row ids[b].
    dW = np.zeros((vocab_size, dlogits.shape[1]))
    # for b in range(len(ids)):
    #     dw[ids[b]] += dlogits[b]
    np.add.at(dW, ids, dlogits)
    return dW

# Step 70 - sgd_update_w
import numpy as np

def sgd_update_w(w, dw, learning_rate):
    """Apply one SGD step: return w - learning_rate * dw as a new array."""
    # TODO: subtract the scaled gradient from the weights and return the new matrix
    return w - dw * learning_rate

# Step 71 - run_one_training_step
def run_one_training_step(w, ids, targets, learning_rate):
    """Run forward, loss, backward, and SGD update once. Return {'w': new_w, 'loss': float}."""
    # TODO: chain the upstream forward/loss/backward/update helpers into one step
    logits = forward_logits_lookup(w, ids)
    probs = logits_to_probs_rowwise(logits)
    loss = cross_entropy_loss(probs, targets)
    dlogits = compute_dlogits(probs, targets)
    dw = compute_dw_scatter_add(ids, dlogits, w.shape[0])
    new_w = sgd_update_w(w, dw, learning_rate)
    return {
        "w": new_w,
        "loss": loss 
    }

# Step 72 - train_neural_bigram_loop
def train_neural_bigram_loop(w, data, block_size, batch_size, learning_rate, num_steps, log_every):
    """Run the neural bigram training loop and return {'w', 'loss_history'}."""
    # TODO: repeatedly sample a batch, run one training step, and log loss every log_every steps
    loss_history = []
    rng = np.random.default_rng()
    for step in range(num_steps):
        batch = get_batch(data, block_size, batch_size, rng)
        x, y = batch
        ids = x.reshape(-1)
        targets = y.reshape(-1)
        out = run_one_training_step(w, ids, targets, learning_rate)
        w = out["w"]
        if step % log_every == 0:
            loss_history.append(out["loss"])
    
    return {
        "w": w,
        "loss_history": loss_history
    }

# Step 73 - sample_from_neural_bigram
def sample_from_neural_bigram(w, start_id, num_tokens, itos):
    """Generate a string by repeatedly sampling from softmax of W[id]."""
    # TODO: starting from start_id, sample num_tokens new ids and decode the full sequence...
    ids = [start_id]
    current_id = start_id
    for _ in range(num_tokens):
        logits = forward_logits_lookup(
            w, np.array([current_id])
        )
        probs = logits_to_probs_rowwise(logits)
        next_id = np.random.choice(
            w.shape[0], p=probs[0]
        )
        ids.append(next_id)
        current_id = next_id
    return decode_ids(ids, itos)

# Step 74 - linear_forward
def linear_forward(x, w):
    # TODO: compute Y = X @ W and return {'y': Y, 'cache': {'x': x, 'w': w}}.
    y = matmul(x, w)
    cache = {
        "x": x,
        "w": w
    }

    return {
        "y": y,
        "cache": cache
    }

# Step 75 - derive_dx_on_paper
def derive_dx_on_paper():
    """Return notes deriving dL/dX = dY @ W.T for Y = X @ W."""
    # TODO: return a multi-line string with the derivation and shape check
    return (
        "Y = X @ W\n"
        "dL/dX = dY @ W.T\n"
        "shapes: X (B, In), W (In, Out), dY (B, Out) -> dL/dX (B, In)"
    )

# Step 76 - derive_linear_dw_on_paper
def derive_linear_dw_on_paper():
    """Return a string with the derivation of dL/dW for Y = X @ W."""
    # TODO: return notes that include the final identity dL/dW = X.T @ dY
    return (
        "Forward: Y = X @ W/\n"
        "Let dY = dL/dY be the upstream gradient.\n"
        "By the chain rule, dL/dW = X.T @ dY.\n"
        "Shapes: X (B, In), W (In, Out), dY (B, Out).\n"
        "Therefore dW has shape (D_in, D_out), matching W."
    )

# Step 77 - linear_backward_dx
def linear_backward_dx(dy, cache):
    # TODO: compute the gradient of the loss w.r.t. the linear layer input X given dy and cache
    w = cache["w"]
    return matmul(dy, transpose_matrix(w))

# Step 78 - linear_backward_dw
def linear_backward_dw(dy, cache):
    """Return dL/dW for a linear layer Y = X @ W."""
    # TODO: compute the weight gradient using x from cache and the upstream dy
    return matmul(transpose_matrix(cache["x"]), dy)

# Step 79 - bias_add_forward
def bias_add_forward(x, b):
    """Add bias vector b (D,) to every row of x (B, D).

    Returns {'y': ndarray (B, D), 'cache': {'b_shape': tuple}}.
    """
    # TODO: add b to each row of x and cache b's shape for the backward pass
    y = vector_matrix_broadcast_add(x, b)
    cache = {
        "b_shape": b.shape
    }

    return {
        "y": y,
        "cache": cache
    }

# Step 80 - bias_add_backward_db
def bias_add_backward_db(dy, cache):
    """Compute db from upstream gradient dy for y = x + b."""
    # TODO: sum the upstream gradient over the batch dimension to get db of shape (D,)
    return sum_axis0(dy)

# Step 81 - relu_forward
def relu_forward(x):
    """Apply elementwise ReLU and cache the input for backward.

    Returns a dict with keys 'y' (activated array) and 'cache' (dict with 'x').
    """
    # TODO: apply elementwise ReLU and cache the input for backward.
    return {
        "y": np.maximum(0, x),
        "cache": {
            "x": x
        }
    }

# Step 82 - relu_backward
def relu_backward(dy, cache):
    """Backward pass for ReLU. cache['x'] holds the original input."""
    # TODO: return dx with gradient zeroed where the cached input was non-positive.
    return dy * (cache["x"] > 0)

# Step 83 - softmax_cross_entropy_backward
def softmax_cross_entropy_backward(probs, targets):
    """Return dL/dlogits for mean cross-entropy with softmax probs."""
    # TODO: produce the (B, V) gradient of mean cross-entropy w.r.t. logits.
    return compute_dlogits(probs, targets)

# Step 84 - layernorm_forward_mean
import numpy as np

def layernorm_forward_mean(x):
    """Return the per-row mean of x with shape (B, 1)."""
    # TODO: compute the per-row mean of x, preserving the reduced axis as size 1
    D = x.shape[-1]
    mean = sum_keepdims(x, axis=-1) / D 
    return mean

# Step 85 - layernorm_forward_variance
import numpy as np

def layernorm_forward_variance(x, mean):
    """Compute the per-row (biased) variance of x given its per-row mean.

    Args:
        x: ndarray of shape (B, D).
        mean: ndarray of shape (B, 1), the per-row mean of x.

    Returns:
        var: ndarray of shape (B, 1), the per-row variance.
    """
    # TODO: compute per-row variance using mean and return a (B, 1) array
    diff = x - mean
    squared_diff = diff ** 2
    D = x.shape[-1]
    return sum_keepdims(squared_diff, axis=-1) / D

# Step 86 - layernorm_forward_normalize
import numpy as np

def layernorm_forward_normalize(x, mean, var, eps):
    """Normalize each row of x to zero mean and unit variance."""
    # TODO: subtract the per-row mean and divide by sqrt(var + eps)
    return (x - mean) / np.sqrt(var + eps)

# Step 87 - layernorm_forward_affine
def layernorm_forward_affine(x, gamma, beta, eps):
    """Run LayerNorm forward over rows of x with affine params gamma, beta."""
    # TODO: normalize each row to zero mean / unit variance, then apply gamma and beta.
    mean = layernorm_forward_mean(x)
    var = layernorm_forward_variance(x, mean)
    x_hat = layernorm_forward_normalize(x, mean, var, eps)
    scaled = elementwise_multiply(x_hat, gamma)
    y = vector_matrix_broadcast_add(scaled, beta)

    cache = {
        "x": x,
        "x_hat": x_hat,
        "mean": mean,
        "var": var,
        "gamma": gamma,
        "eps": eps
    }

    return {
        "y": y,
        "cache": cache
    }

# Step 88 - layernorm_backward_subtract_mean
import numpy as np

def layernorm_backward_subtract_mean(dy, cache):
    """Gradient through y = x - mean(x, axis=1, keepdims=True).

    dy: (B, D) upstream gradient w.r.t. the centered output.
    cache: dict with keys 'x' (B, D) and 'mean' (B,).
    Returns dx of shape (B, D).
    """
    # TODO: compute the gradient contribution of the subtract-mean op
    return dy - dy.mean(axis=-1, keepdims=True)

# Step 89 - layernorm_backward_divide_std
def layernorm_backward_divide_std(dy, cache):
    """Propagate dy through the divide-by-std step of LayerNorm."""
    # TODO: propagate the upstream gradient through the divide-by-std step of LayerNorm
    var = cache["var"]
    eps = cache["eps"]
    std = np.sqrt(var + eps)
    dx = dy / std
    return dx

# Step 90 - layernorm_backward_full
import numpy as np

def layernorm_backward_full(dy, cache):
    """Full LayerNorm backward. Return {'dx', 'dgamma', 'dbeta'}."""
    # TODO: chain rule back through affine, divide-by-std, and subtract-mean.
    x_hat = cache["x_hat"]
    var = cache["var"]
    gamma = cache["gamma"]
    eps = cache["eps"]
    dbeta = np.sum(dy, axis=0)
    dgamma = np.sum(dy * x_hat, axis=0)

    g = dy * gamma
    mean_g = np.mean(g, axis=-1, keepdims=True)
    mean_g_xhat = np.mean(
        g * x_hat, axis=-1, keepdims=True
    )

    dx = (
        g - mean_g - x_hat * mean_g_xhat
    ) / np.sqrt(var + eps)

    return {
        "dx": dx,
        "dgamma": dgamma,
        "dbeta": dbeta
    }

# Step 91 - layernorm_backward_implementation
def layernorm_backward_implementation(d_out, cache):
    # TODO: return {'dx', 'dgamma', 'dbeta'} gradients for LayerNorm given d_out and the forward cache.
    grads = layernorm_backward_full(d_out, cache)
    return {
        "dx": grads["dx"],
        "dgamma": grads["dgamma"],
        "dbeta": grads["dbeta"]
    }

# Step 92 - create_token_embedding
def create_token_embedding(vocab_size, d_model, scale=0.02):
    """Initialize the token embedding matrix E of shape (vocab_size, d_model)."""
    # TODO: return a (vocab_size, d_model) array of small random values controlled by scale
    w = np.random.randn(vocab_size, d_model)
    return scale_w_small(w, scale)

# Step 93 - token_embedding_forward
def token_embedding_forward(token_ids, embedding_matrix):
    """Look up token embeddings for a batch of integer token ids.

    Inputs:
        token_ids: ndarray of shape (B, T), dtype int
        embedding_matrix: ndarray of shape (V, d_model)
    Returns:
        out: ndarray of shape (B, T, d_model)
        cache: dict with keys 'token_ids', 'vocab_size'
    """
    # TODO: look up the embedding row for each token id and build the cache
    y = embedding_matrix[token_ids]
    cache = {
        "token_ids": token_ids,
        "vocab_size": embedding_matrix.shape[0]
    }

    return y, cache

# Step 94 - token_embedding_backward
import numpy as np

def token_embedding_backward(d_out, cache):
    # TODO: scatter-add d_out into a (vocab_size, d_model) dE using cache['token_ids'].
    token_ids = cache["token_ids"]
    vocab_size = cache["vocab_size"]
    d_model = d_out.shape[-1]
    dE = np.zeros((vocab_size, d_model), dtype=d_out.dtype)
    flat_ids = token_ids.reshape(-1)
    flat_grad = d_out.reshape(-1, d_model)
    np.add.at(dE, flat_ids, flat_grad)
    return dE

# Step 95 - create_positional_embedding
def create_positional_embedding(block_size, d_model, scale=0.02):
    """Initialize the learned positional embedding matrix P of shape (block_size, d_model)."""
    # TODO: build a (block_size, d_model) matrix of small random values scaled by `scale`
    P = make_2d_random(block_size, d_model, seed=None)
    return scale_w_small(P, scale)

# Step 96 - slice_positional_embedding
import numpy as np

def slice_positional_embedding(positional_matrix, seq_len):
    """Return the first seq_len rows of the positional embedding matrix."""
    # TODO: return the leading seq_len rows of positional_matrix as a (seq_len, d_model) array.
    return slice_subblock(positional_matrix, 0, seq_len, 0, positional_matrix.shape[1])

# Step 97 - add_token_and_positional_embeddings
def add_token_and_positional_embeddings(token_emb, pos_emb):
    """Sum token embeddings (B,T,d_model) and positional embeddings (T,d_model)."""
    # TODO: combine token and positional embeddings into a single (B,T,d_model) tensor
    return elementwise_add(token_emb, pos_emb)

# Step 98 - embedding_sum_backward
def embedding_sum_backward(d_out):
    """Backprop through H = token_emb + pos_emb (with broadcasting over batch)."""
    # TODO: route d_out to both branches, reducing over the batch axis for pos_emb.
    d_token_emb = d_out
    d_pos_emb = sum_axis0(d_out)
    return {
        "d_token_emb": d_token_emb,
        "d_pos_emb": d_pos_emb,
    }

# Step 99 - create_qkv_projections
def create_qkv_projections(d_model, d_head, scale=0.02):
    # TODO: return a dict with 'Wq','Wk','Wv', each of shape (d_model, d_head)
    return {
        'Wq': scale_w_small(make_2d_random(d_model, d_head, seed=0), scale),
        'Wk': scale_w_small(make_2d_random(d_model, d_head, seed=1), scale),
        'Wv': scale_w_small(make_2d_random(d_model, d_head, seed=2), scale)
    }

# Step 100 - compute_query
import numpy as np

def compute_query(x, w_q):
    """Project x (B, T, d_model) into queries Q (B, T, d_head) using w_q."""
    # TODO: project x into the query space using w_q
    return matmul(x, w_q)

# Step 101 - compute_key
def compute_key(x, w_k):
    """Project x through Wk to get keys K of shape (B, T, d_head)."""
    # TODO: project the (B, T, d_model) input through w_k to produce (B, T, d_head) keys.
    return matmul(x, w_k)

# Step 102 - compute_value
def compute_value(x, w_v):
    # TODO: project x of shape (B, T, d_model) by w_v of shape (d_model, d_head)
    return matmul(x, w_v)

# Step 103 - compute_attention_scores
import numpy as np

def compute_attention_scores(q, k):
    """Return raw attention scores Q @ K^T with shape (B, T, T)."""
    # TODO: compute raw attention scores Q @ K^T per batch element
    return matmul(q, k.transpose(0, 2, 1))

# Step 104 - scale_attention_scores
import numpy as np

def scale_attention_scores(scores, d_head):
    """Rescale (B, T, T) attention scores by a function of d_head."""
    # TODO: rescale the scores so their variance does not grow with d_head.
    return scores / math.sqrt(d_head)

# Step 105 - build_causal_mask
import numpy as np

def build_causal_mask(seq_len):
    """Return a (seq_len, seq_len) boolean lower-triangular mask."""
    # TODO: build a (T, T) boolean mask where True marks allowed (query, key) pairs
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

# Step 106 - apply_causal_mask
import numpy as np

def apply_causal_mask(scaled_scores, causal_mask):
    """Replace future positions in scaled_scores with -inf using causal_mask."""
    # TODO: return a (B,T,T) array where positions with causal_mask False are -inf...
    return np.where(causal_mask, scaled_scores, -np.inf)

# Step 107 - softmax_attention_weights
import numpy as np

def softmax_attention_weights(masked_scores):
    """Row-wise stable softmax over the last axis of (B, T, T) scores."""
    # TODO: apply numerically stable softmax along the last axis of masked_scores
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    shifted = masked_scores - max_scores
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

# Step 108 - attention_weighted_values
import numpy as np

def attention_weighted_values(attn, v):
    """Combine attention weights with values: out = attn @ V.

    attn: (B, T, T) softmaxed attention weights
    v:    (B, T, d_head) value vectors
    returns: (B, T, d_head)
    """
    # TODO: mix the value vectors using the attention weights
    return attn @ v

# Step 109 - apply_output_projection
import numpy as np

def apply_output_projection(attn_out, w_o):
    """Project attention output (B,T,d_head) through Wo (d_head,d_model)."""
    # TODO: return attn_out projected through w_o to shape (B, T, d_model)
    return attn_out @ w_o

# Step 110 - output_projection_backward
def output_projection_backward(d_proj, cache):
    """Backprop through proj = attn_out @ w_o. Return {'d_attn_out', 'dw_o'}."""
    # TODO: backprop through proj = attn_out @ w_o, return gradients for attn_out and w_o
    attn_out = cache['attn_out']
    w_o = cache['w_o']

    '''
        proj = attn_out @ w_o
        attn_out: (B, T, d_head)
        w_o: (d_head, d_model)
        proj: (B, T, d_model)
    '''

    d_attn_out = matmul(d_proj, np.transpose(w_o))
    attn_out_T = np.transpose(attn_out, (0, 2, 1))
    dw_o = np.sum(attn_out_T @ d_proj, axis=0)
    return {
        'd_attn_out': d_attn_out,
        'dw_o': dw_o,
    }

# Step 111 - attention_value_backward
import numpy as np

def attention_value_backward(d_attn_out, cache):
    """Backprop through out = attn @ V.

    d_attn_out: (B, T, d_head) upstream gradient w.r.t. attention output.
    cache: dict with 'attn' of shape (B, T, T) and 'v' of shape (B, T, d_head).
    Returns dict with 'd_attn' (B, T, T) and 'd_v' (B, T, d_head).
    """
    # TODO: backprop through out = attn @ V to obtain gradients for attn and V.
    attn = cache["attn"]
    v = cache["v"]
    '''
        out = attn @ v
        d_attn_out: (B, T, d_head)
        attn: (B, T, T)
        v: (B, T, d_head)
    '''

    d_attn = d_attn_out @ np.transpose(v, (0, 2, 1))
    d_v = np.transpose(attn, (0, 2, 1)) @ d_attn_out 
    return {
        "d_attn": d_attn,
        "d_v": d_v
    }

# Step 112 - masked_softmax_backward
import numpy as np

def masked_softmax_backward(d_attn, cache):
    """Backprop through the masked row-wise softmax.

    d_attn: ndarray of shape (B, T, T) -- gradient w.r.t. attention weights.
    cache: dict with 'attn' (B,T,T) and 'causal_mask' (T,T) boolean.
    Returns d_masked_scores of shape (B, T, T).
    """
    # TODO: propagate the softmax Jacobian per row and zero out masked positions.
    attn = cache['attn']
    causal_mask = cache['causal_mask']

    dot = np.sum(d_attn * attn, axis=-1, keepdims=True)
    d_scores = attn * (d_attn - dot)
    d_scores = np.where(causal_mask, d_scores, 0.0)
    return d_scores

# Step 113 - scale_scores_backward
import numpy as np

def scale_scores_backward(d_scaled_scores, d_head):
    """Backprop through the 1/sqrt(d_head) attention score scaling."""
    # TODO: propagate d_scaled_scores back through the sqrt(d_head) scaling
    return d_scaled_scores / math.sqrt(d_head)

# Step 114 - qk_scores_backward
import numpy as np

def qk_scores_backward(d_scores, cache):
    """Backprop through scores = Q @ K^T.

    d_scores: (B, T, T)
    cache: dict with 'q' and 'k', each (B, T, d_head)
    returns: {'d_q': (B, T, d_head), 'd_k': (B, T, d_head)}
    """
    # TODO: backprop scores = Q @ K^T to obtain gradients for Q and K
    q = cache["q"]
    k = cache["k"]
    d_q = d_scores @ k
    d_k = np.transpose(d_scores, (0, 2, 1)) @ q

    return {
        'd_q': d_q,
        'd_k': d_k,
    }

# Step 115 - qkv_projection_backward
def qkv_projection_backward(d_q, d_k, d_v, cache):
    # TODO: backprop through Q=x@Wq, K=x@Wk, V=x@Wv to get dx and dw_q, dw_k, dw_v.
    x = cache['x']
    w_q = cache['w_q']
    w_k = cache['w_k']
    w_v = cache['w_v']

    dx = d_q @ np.transpose(w_q) + d_k @ np.transpose(w_k) + d_v @ np.transpose(w_v)
    x_T = np.transpose(x, (0, 2, 1))

    dw_q = np.sum(x_T @ d_q, axis=0)
    dw_k = np.sum(x_T @ d_k, axis=0)
    dw_v = np.sum(x_T @ d_v, axis=0)
    return {
        "dx": dx,
        "dw_q": dw_q,
        "dw_k": dw_k,
        "dw_v": dw_v
    }

# Step 116 - choose_attention_head_config
def choose_attention_head_config(d_model, n_heads):
    """Return a config dict {'n_heads', 'd_head', 'd_model'} for multi-head attention."""
    # TODO: split d_model into n_heads equal-sized d_head chunks and return the config dict
    if n_heads <= 0 or d_model % n_heads:
        raise ValueError(
            f"n_heads{n_heads} must be positive and evenly divide d_model={d_model}"
        )

    return {
        'n_heads': n_heads,
        'd_head': d_model // n_heads,
        'd_model': d_model,
    }

# Step 117 - create_multihead_qkv_projections
def create_multihead_qkv_projections(d_model, scale=0.02):
    """Initialize Wq, Wk, Wv as (d_model, d_model) matrices for multi-head attention."""
    # TODO: build a dict with keys 'Wq', 'Wk', 'Wv', each a scaled (d_model, d_model) random matrix
    Wq = scale_w_small(make_2d_random(d_model, d_model, seed=0), scale)
    Wk = scale_w_small(make_2d_random(d_model, d_model, seed=1), scale)
    Wv = scale_w_small(make_2d_random(d_model, d_model, seed=2), scale)

    return {
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv
    }

# Step 118 - create_multihead_output_projection
def create_multihead_output_projection(d_model, scale=0.02):
    """Initialize Wo of shape (d_model, d_model) for multi-head attention output projection."""
    # TODO: build a (d_model, d_model) random matrix and scale it down by `scale`.
    return scale_w_small(make_2d_random(d_model, d_model, seed=0), scale)

# Step 119 - reshape_to_heads
import numpy as np

def reshape_to_heads(x, n_heads, d_head):
    """Reshape (B, T, d_model) into (B, T, n_heads, d_head)."""
    # TODO: split the last dimension of x into n_heads chunks of size d_head
    B, T, d_model = x.shape
    return np.reshape(x, (B, T, n_heads, d_head))

# Step 120 - transpose_heads_to_front
import numpy as np

def transpose_heads_to_front(x_heads):
    """Transpose (B, T, n_heads, d_head) to (B, n_heads, T, d_head)."""
    # TODO: move the heads axis in front of the time axis
    return np.transpose(x_heads, (0, 2, 1, 3))

# Step 121 - get_multihead_n_heads
def get_multihead_n_heads(config):
    # TODO: return the number of attention heads stored in the multi-head config dict.
    return config["n_heads"]

# Step 122 - get_multihead_sequence_length
import numpy as np

def get_multihead_sequence_length(x):
    """Return T from x of shape (B, T, d_model)."""
    # TODO: return the sequence length T from the (B, T, d_model) tensor.
    return x.shape[1]

# Step 123 - compute_d_head
def compute_d_head(d_model, n_heads):
    # TODO: return the per-head dimension d_head for multi-head attention.
    if d_model % n_heads:
        raise ValueError("n_heads does not evenly divide d_model")
    return d_model // n_heads

# Step 124 - multihead_masked_softmax_scores
def multihead_masked_softmax_scores(scores, mask):
    """Apply causal mask and row-wise softmax to multi-head attention scores.

    Args:
        scores: ndarray of shape (B, n_heads, T, T)
        mask:   ndarray of shape (T, T), True where positions are kept

    Returns:
        weights: ndarray of shape (B, n_heads, T, T)
    """
    # TODO: mask future positions then row-wise softmax over the last axis
    
    masked = apply_causal_mask(scores, mask)
    B, H, T, _ = masked.shape
    flat = masked.reshape(B * H * T, T)
    attn = stable_softmax_2d_rowwise(flat)

    return attn.reshape(B, H, T, T)

# Step 125 - multihead_weighted_sum
import numpy as np

def multihead_weighted_sum(weights, v_heads):
    """Compute per-head attention output as weights @ V across all heads."""
    # TODO: combine attention weights with values across heads
    return weights @ v_heads

# Step 126 - transpose_heads_to_back
def transpose_heads_to_back(x_heads):
    # TODO: move the heads axis back so the result has shape (B, T, n_heads, d_head).
    return np.transpose(x_heads, (0, 2, 1, 3))

# Step 127 - get_multihead_output_sequence_length
def get_multihead_output_sequence_length(x_heads_back):
    """Return T from a (B, T, n_heads, d_head) tensor."""
    # TODO: read the sequence-length dimension from x_heads_back's shape
    return x_heads_back.shape[1]

# Step 128 - merge_heads_to_d_model
import numpy as np

def merge_heads_to_d_model(x_heads_back):
    """Reshape (B, T, n_heads, d_head) into (B, T, d_model)."""
    # TODO: collapse the last two axes into a single d_model axis
    B, T, n_heads, d_head = x_heads_back.shape
    return x_heads_back.reshape(B, T, n_heads * d_head)

# Step 129 - multihead_output_projection_forward
def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    # TODO: project merged through w_out, add b_out, and stash inputs in the cache.
    lin = linear_forward(merged, w_out)
    bias = bias_add_forward(lin['y'], b_out)
    cache = {
      'merged': lin['cache']['x'],
      'w_out': lin['cache']['w']
    }
    return {'out': bias['y'], 'cache': cache}

# Step 130 - multihead_reshape_transpose_backward
def multihead_reshape_transpose_backward(d_merged, shape_info):
    """Invert merge_heads_to_d_model to recover (B, n_heads, T, d_head) gradients."""
    # TODO: undo the merge/transpose/reshape chain from the forward pass
    n_heads = shape_info['n_heads']
    d_head = shape_info['d_head']
    return transpose_heads_to_front(reshape_to_heads(d_merged, n_heads, d_head))

# Step 131 - ffn_linear_one_forward
def ffn_linear_one_forward(x, w1, b1):
    """First FFN linear: lift (B, T, d_model) up to (B, T, d_ff) and add bias."""
    # TODO: apply the first FFN linear that expands d_model to d_ff
    lin = linear_forward(x, w1)
    bias = bias_add_forward(lin['y'], b1)
    cache = {
        'x': x,
        'w1': w1,
    }
    return {'h1': bias['y'], 'cache': cache}

# Step 132 - ffn_activation_forward
def ffn_activation_forward(h1):
    """Apply ReLU to FFN hidden pre-activations.

    Args:
        h1: ndarray of shape (B, T, d_ff)

    Returns:
        a1: ndarray of shape (B, T, d_ff)
        cache: dict with key 'h1'
    """
    # TODO: apply ReLU activation in the FFN hidden layer and cache h1
    out = relu_forward(h1)
    cache = {'h1': h1}
    return out['y'], cache

# Step 133 - ffn_linear_two_forward
def ffn_linear_two_forward(a1, w2, b2):
    # TODO: project a1 (B, T, d_ff) down to (B, T, d_model) using w2 and b2, return h2 and cache
    lin = linear_forward(a1, w2)
    bias = bias_add_forward(lin['y'], b2)
    cache = {
        'a1': a1,
        'w2': w2
    }
    return {'h2': bias['y'], 'cache': cache}

# Step 134 - ffn_backward
def ffn_backward(d_out, cache):
    """Backprop through linear2 -> ReLU -> linear1 of the FFN.

    cache keys: 'x', 'w1', 'h1', 'a1', 'w2'.
    Returns dict with keys: 'dx', 'dw1', 'db1', 'dw2', 'db2'.
    """
    # TODO: route d_out back through linear2, ReLU, and linear1 to get input and param grads
    x = cache['x']
    w1 = cache['w1']
    h1 = cache['h1']
    a1 = cache['a1']
    w2 = cache['w2']

    x_flat = x.reshape(-1, x.shape[-1])
    h1_flat = h1.reshape(-1, h1.shape[-1])
    a1_flat = a1.reshape(-1, a1.shape[-1])
    d_out_flat = d_out.reshape(-1, d_out.shape[-1])

    db2 = bias_add_backward_db(
        d_out_flat,
        {"b_shape": (w2.shape[1],)}
    )
    dw2 = linear_backward_dw(
        d_out_flat,
        {"x": a1_flat}
    )
    da1 = linear_backward_dx(
        d_out_flat,
        {"w": w2}
    )

    dh1 = relu_backward(
        da1, 
        {"x": h1_flat}
    )

    db1 = bias_add_backward_db(
        dh1,
        {"b_shape": (w1.shape[1], 1)}
    )
    dw1 = linear_backward_dw(
        dh1,
        {"x": x_flat}
    )
    dx_flat = linear_backward_dx(
        dh1,
        {"w": w1}
    )

    dx = dx_flat.reshape(x.shape)
    return {
        "dx": dx,
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2
    }

# Step 135 - residual_forward
def residual_forward(x, sublayer_out):
    """Return x + sublayer_out for a residual connection."""
    # TODO: add the sublayer output to its input to form a residual connection.
    return elementwise_add(x, sublayer_out)

# Step 136 - residual_backward
def residual_backward(d_y):
    """Backprop through y = x + sublayer_out. Returns (d_x, d_sublayer_out)."""
    # TODO: route the upstream gradient to both branches of the residual add.
    return np.array(d_y, copy=True), np.array(d_y, copy=True)

# Step 137 - pre_layernorm_sublayer_forward
def pre_layernorm_sublayer_forward(x, ln_params, sublayer_fn, sublayer_params):
    # TODO: apply LayerNorm to x, run sublayer_fn on the result, then residual-add back to x.
    gamma = ln_params['gamma']
    beta = ln_params['beta']
    eps = ln_params.get('eps', 1e-5)

    ln_result = layernorm_forward_affine(x, gamma, beta, eps)
    x_norm = ln_result['y']

    sublayer_result = sublayer_fn(x_norm, sublayer_params)
    sublayer_out = sublayer_result['y']

    y = residual_forward(x, sublayer_out)

    cache = {
        "x": x,
        "ln_cache": ln_result["cache"],
        "sublayer_cache": sublayer_result["cache"]
    }

    return {
        "y": y,
        "cache": cache
    }

# Step 138 - transformer_block_forward
def transformer_block_forward(x, block_params):
    """Run one pre-LN Transformer block forward.

    Args:
        x: ndarray of shape (B, T, d_model).
        block_params: dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        dict with 'y' (B, T, d_model) and 'cache' with keys
        'attn_branch' and 'ffn_branch'.
    """
    # TODO: compose pre-LN attention sublayer then pre-LN FFN sublayer with residuals
    attn_p = block_params['attn']
    ffn_p = block_params['ffn']

    def attn_sublayer(x_in, p):
        B, T, d_model = x_in.shape
        n_heads = p.get('n_heads', block_params.get('n_heads', 1))
        d_head = d_model // n_heads

        Q = x_in @ p['Wq']
        K = x_in @ p['Wk']
        V = x_in @ p['Wv']

        def to_heads(v):
            return transpose_heads_to_front(reshape_to_heads(v, n_heads, d_head))
        
        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)

        scores = Q @ np.transpose(K, (0, 1, 3, 2)) / np.sqrt(d_head)
        causal_mask = build_causal_mask(T)
        attn_w = multihead_masked_softmax_scores(scores, causal_mask)

        out = attn_w @ V
        out = np.transpose(out, (0, 2, 1, 3))
        merged = np.reshape(out, (B, T, d_model))

        bo = p.get('bo', np.zeros(d_model))
        proj = multihead_output_projection_forward(merged, p['Wo'], bo)
        return {'y': proj['out'], 'cache': proj['cache']}

    def ffn_sublayer(x_in, p):
        l1 = ffn_linear_one_forward(x_in, p['w1'], p['b1'])
        a1, act_cache = ffn_activation_forward(l1['h1'])
        l2 = ffn_linear_two_forward(a1, p['w2'], p['b2'])
        return {
            'y': l2['h2'],
            'cache': {
                'linear_one': l1['cache'],
                'activation': act_cache,
                'linear_two': l2['cache'],
            },
        }
    attn_branch = pre_layernorm_sublayer_forward(
        x, block_params['ln1'], attn_sublayer, attn_p
    )
    ffn_branch = pre_layernorm_sublayer_forward(
        attn_branch['y'], block_params['ln2'], ffn_sublayer, ffn_p
    )

    return {
        'y': ffn_branch['y'],
        'cache': {
            'attn_branch': attn_branch['cache'],
            'ffn_branch': ffn_branch['cache'],
        },
    }

# Step 139 - transformer_block_backward
def transformer_block_backward(d_y, cache, block_params):
    """Backward pass for a pre-LN Transformer block.

    Args:
        d_y: upstream gradient w.r.t. block output, shape (B, T, D).
        cache: dict from transformer_block_forward, with keys 'attn_branch' and 'ffn_branch'.
        block_params: nested dict with keys 'ln1', 'attn', 'ln2', 'ffn'.

    Returns:
        (d_x, grads) where d_x has shape (B, T, D) and grads is a nested dict
        with keys 'ln1', 'ln2', 'attn', 'ffn' mirroring block_params.
    """
    # Tip: recover x from cache['attn_branch']['x'] and call _complete_block_cache(x, block_params)
    # to guarantee every field the backward helpers need is present, no matter what the forward saved.
    # TODO: reverse the FFN branch then the attention branch, summing residual + sublayer gradients
    x = cache['attn_branch']['x']
    full_cache = _complete_block_cache(x, block_params)

    attn_branch = full_cache["attn_branch"]
    ffn_branch = full_cache["ffn_branch"]

    d_ln2_out, ffn_grads = _ffn_sublayer_backward(
        d_y,
        ffn_branch["sublayer_cache"],
        block_params["ffn"]
    )

    d_h1_ln, dgamma2, dbeta2 = layernorm_backward_affine(
        d_ln2_out,
        ffn_branch["ln_cache"]
    )

    d_h1 = d_y + d_h1_ln

    d_ln1_out, attn_grads = _attn_sublayer_backward(
        d_h1,
        attn_branch["sublayer_cache"],
        block_params["attn"]
    )

    d_x_ln, dgamma1, dbeta1 = layernorm_backward_affine(
        d_ln1_out,
        attn_branch["ln_cache"]
    )

    d_x = d_h1 + d_x_ln

    grads = {
        "ln1": {
            "gamma": dgamma1,
            "beta": dbeta1
        },
        "ln2": {
            "gamma": dgamma2,
            "beta": dbeta2
        },
        "attn": attn_grads,
        "ffn": ffn_grads
    }

    return d_x, grads

# Step 140 - stack_transformer_blocks
import numpy as np

def stack_transformer_blocks(n_layers, d_model, n_heads, d_ff):
    """Build a list of n_layers Transformer block parameter dicts.

    Each block dict has keys 'ln1', 'attn', 'ln2', 'ffn'.
    """
    # TODO: create n_layers initialized block parameter dicts and return them as a list
    blocks = []
    scale = 0.02
    for i in range(n_layers):
        ln1 = {
            "gamma": np.ones(d_model),
            "beta": np.zeros(d_model)
        }

        qkv = create_multihead_qkv_projections(d_model, 0.02)
        Wo = create_multihead_output_projection(d_model, 0.02)

        attn = {
            "Wq": qkv["Wq"],
            "Wk": qkv["Wk"],
            "Wv": qkv["Wv"],
            "Wo": Wo,
            "bo": np.zeros(d_model)
        }

        ln2 = {
            "gamma": np.ones(d_model),
            "beta": np.zeros(d_model)
        }

        W1 = scale_w_small(
            make_2d_random(d_model, d_ff, seed=10),
            0.02
        )
        W2 = scale_w_small(
            make_2d_random(d_ff, d_model, seed=11),
            0.02
        )

        ffn = {
            "W1": W1,
            "b1": np.zeros(d_ff),
            "W2": W2,
            "b2": np.zeros(d_model)
        }

        block = {
            "ln1": ln1,
            "attn": attn,
            "ln2": ln2,
            "ffn": ffn
        }

        blocks.append(block)
    return blocks

# Step 141 - forward_through_all_blocks
def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    # TODO: thread x through each block in `blocks`, collecting per-block caches
    y = x
    caches = []

    for block_params in blocks:
        result = transformer_block_forward(y, block_params)
        y = result['y']
        caches.append(result["cache"])
    return y, caches

# Step 142 - backward_through_all_blocks
def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    # TODO: walk the blocks in reverse, calling transformer_block_backward each step.
    d_x = d_y
    grads = []

    for i in reversed(range(len(blocks))):
      d_x, grads_block = transformer_block_backward(
        d_x,
        caches[i],
        blocks[i]
      )
      grads.append(grads_block)
    grads.reverse()
    return d_x, grads

# Step 143 - final_layernorm_forward
def final_layernorm_forward(x, gamma, beta):
    """Apply LayerNorm to a (B, T, d_model) tensor with affine params gamma, beta.

    Returns (y, cache) where cache has keys 'x', 'mean', 'var', 'x_hat', 'gamma'.
    """
    # TODO: normalize each (b, t) position across the d_model channels, then apply gamma/beta.
    eps = 1e-5
    result = layernorm_forward_affine(x, gamma, beta, eps)
    y = result["y"]
    cache = result["cache"]

    return y, cache

# Step 144 - lm_head_linear_forward
def lm_head_linear_forward(x, w_lm, b_lm):
    """Project hidden states (B,T,d_model) to logits (B,T,vocab_size)."""
    # TODO: project final hidden states to vocab-size logits via the language model head.
    B, T, D = x.shape
    x_flat = x.reshape(B * T, D)
    linear_result = linear_forward(x_flat, w_lm)

    bias_result = bias_add_forward(
        linear_result["y"], b_lm
    )

    vocab_size = w_lm.shape[1]
    logits = bias_result["y"].reshape(B, T, vocab_size)

    cache = {
        "x": x,
        "w_lm": w_lm,
    }

    return {
        "logits": logits,
        "cache": cache
    }

# Step 145 - full_model_forward
def full_model_forward(x_ids, model_params):
    """Run embeddings, all blocks, final LN, and LM head; return logits and caches."""
    # TODO: chain token+positional embeddings, Transformer blocks, final LayerNorm, and LM head.
    B, T = x_ids.shape

    tok_out, tok_cache = token_embedding_forward(
        x_ids,
        model_params["tok_emb"]
    )

    pos_out = slice_positional_embedding(
        model_params["pos_emb"], T
    )

    x = add_token_and_positional_embeddings(
        tok_out,
        pos_out
    )

    x, block_caches = forward_through_all_blocks(
        x,
        model_params["blocks"]
    )

    x, ln_cache = final_layernorm_forward(
        x,
        model_params["ln_f"]["gamma"],
        model_params["ln_f"]["beta"]
    )

    lm_result = lm_head_linear_forward(
        x,
        model_params["lm_head"]["w_lm"],
        model_params["lm_head"]["b_lm"]
    )

    logits = lm_result["logits"]

    caches = {
        "emb": tok_cache,
        "blocks": block_caches,
        "ln_f": ln_cache,
        "lm_head": lm_result["cache"]
    }

    return logits, caches

# Step 146 - full_model_backward
def full_model_backward(d_logits, caches, model_params):
    """Propagate d_logits back through LM head, final LN, blocks, and embeddings.

    Args:
        d_logits: (B, T, V) gradient w.r.t. the model output
        caches: nested dict from full_model_forward with keys
                'emb', 'blocks', 'ln_f', 'lm_head'
        model_params: nested dict matching the forward's parameter tree

    Returns:
        grads: nested dict mirroring model_params with keys
               'tok_emb', 'pos_emb', 'blocks', 'ln_f': {'gamma', 'beta'},
               'lm_head': {'w_lm', 'b_lm'}
    """
    # TODO: walk the forward chain in reverse, returning a grads tree shaped like model_params
    lm_cache = caches["lm_head"]
    x_lm = lm_cache["x"]
    w_lm = lm_cache["w_lm"]

    B, T, V = d_logits.shape
    D = x_lm.shape[-1]

    x_flat = x_lm.reshape(-1, D)
    d_logits_flat = d_logits.reshape(-1, V)

    dw_lm = x_flat.T @ d_logits_flat
    db_lm = d_logits.sum(axis=(0, 1))

    d_ln_out = d_logits @ w_lm.T

    ln_cache = caches["ln_f"]

    x_hat = ln_cache["x_hat"]
    var = ln_cache["var"]
    gamma = ln_cache["gamma"]
    eps = ln_cache.get('eps', 1e-5)

    dgamma = np.sum(d_ln_out * x_hat, axis=(0, 1))
    dbeta = np.sum(d_ln_out, axis=(0, 1))

    g = d_ln_out * gamma

    mean_g = np.mean(g, axis=-1, keepdims=True)

    mean_g_xhat = np.mean(
        g * x_hat, axis=-1, keepdims=True
    )

    d_blocks_out = (
        g - mean_g - x_hat * mean_g_xhat
    ) / np.sqrt(var + eps)

    d_emb, block_grads = backward_through_all_blocks(
        d_blocks_out,
        caches["blocks"],
        model_params["blocks"]
    )

    emb_cache = caches["emb"]
    token_ids = emb_cache["tok_cache"]["token_ids"]
    d_tok_emb = np.zeros_like(model_params["tok_emb"])
    np.add.at(d_tok_emb, token_ids, d_emb)

    seq_len = emb_cache["seq_len"]
    d_pos_emb = np.zeros_like(model_params["pos_emb"])
    d_pos_emb[:seq_len] = d_emb.sum(axis=0)

    grads = {
    "tok_emb": d_tok_emb,
    "pos_emb": d_pos_emb,
    "blocks": block_grads,
    "ln_f": {
        "gamma": dgamma,
        "beta": dbeta
    },
    "lm_head": {
        "w_lm": dw_lm,
        "b_lm": db_lm
        }
    }

    return grads

# Step 147 - initialize_adam_moments
import numpy as np

def initialize_adam_moments(model_params):
    """Allocate zeroed Adam first- and second-moment buffers matching model_params."""
    # TODO: walk the nested parameter dict and build parallel (m, v) zero buffers
    def _zeros_like_tree(tree):
        if isinstance(tree, dict):
            return {k: _zeros_like_tree(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [_zeros_like_tree(v) for v in tree]
        if isinstance(tree, np.ndarray):
            return np.zeros_like(tree)
        return tree
    
    m = _zeros_like_tree(model_params)
    v = _zeros_like_tree(model_params)
    return m, v

# Step 148 - initialize_adam_step_counter
def initialize_adam_step_counter():
    """Return the initial Adam step counter t."""
    # TODO: return the starting value of the Adam time-step counter.
    return 0

# Step 149 - adam_increment_step
def adam_increment_step(t):
    """Return t + 1 so Adam bias correction sees a positive step."""
    # TODO: return the next Adam step counter value
    return t + 1

# Step 150 - adam_update_first_moment
import numpy as np

def adam_update_first_moment(m, grad, beta1):
    """Return the updated Adam first-moment estimate."""
    # TODO: blend previous moment m with current grad using decay beta1
    return beta1 * m + (1 - beta1) * grad

# Step 151 - adam_update_second_moment (not yet solved)
# TODO: implement

# Step 152 - adam_bias_correction (not yet solved)
# TODO: implement

# Step 153 - adam_parameter_update (not yet solved)
# TODO: implement

# Step 154 - wire_full_training_loop (not yet solved)
# TODO: implement

# Step 155 - logging_and_validation_loss (not yet solved)
# TODO: implement

# Step 156 - encode_prompt (not yet solved)
# TODO: implement

# Step 157 - crop_context_to_block_size (not yet solved)
# TODO: implement

# Step 158 - forward_to_get_logits (not yet solved)
# TODO: implement

# Step 159 - take_last_position_logits (not yet solved)
# TODO: implement

# Step 160 - apply_temperature (not yet solved)
# TODO: implement

# Step 161 - top_k_filter (not yet solved)
# TODO: implement

# Step 162 - softmax_to_probs (not yet solved)
# TODO: implement

# Step 163 - sample_one_token (not yet solved)
# TODO: implement

# Step 164 - append_token_to_sequence (not yet solved)
# TODO: implement

# Step 165 - generation_loop_for_n_steps (not yet solved)
# TODO: implement

# Step 166 - decode_final_sequence (not yet solved)
# TODO: implement

