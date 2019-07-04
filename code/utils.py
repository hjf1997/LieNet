import torch
import numpy as np


def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(A.shape) == 2:
        return A.mm(B) #np.dot(A, B)

    # Old (slower) implementation:
    # a = A.reshape(np.hstack([np.shape(A), [1]]))
    # b = B.reshape(np.hstack([[np.shape(B)[0]], [1], np.shape(B)[1:]]))
    # return np.sum(a * b, axis=2)

    # Approx 5x faster, only supported by numpy version >= 1.6:
    return torch.einsum('ijk,ikl->ijl', A, B)


def multiskew(A):
    # Inspired by MATLAB multiskew function by Nicholas Boumal.
    return 0.5 * (A - multitransp(A))


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if len(A.shape) == 2:
        return A.t()
    return A.permute(0, 2, 1)


def retr(x, u, k):
    x = x.permute(2, 0, 1)
    u = u.permute(2, 0, 1)
    def retri(y):
        q, r = torch.qr(y)
        return torch.matmul(q, torch.diag(torch.sign(torch.sign(torch.diag(r)) + 0.5)))
    y = x + multiprod(x, u)
    if k == 0:
        print(y.shape)
        return retri(y)
    else:
        for i in range(k):
            y[i] = retri(y[i])
        return y.permute(1, 2, 0)


def egrad2rgrad(X, H):
    X = X.permute(2, 0, 1)
    H = H.permute(2, 0, 1)
    k = multiskew(multiprod(multitransp(X), H))
    k = k.permute(1, 2, 0)
    return k


# def egrad2rgrad(x, egrad):
#     x = x.permute(2, 0, 1)
#     egrad = egrad.permute(2, 0, 1)
#     d_hat = torch.matmul(torch.matmul(egrad, x.permute(0, 2, 1)), x)
#     rgrad = egrad - d_hat
#     rgrad = rgrad.permute(1, 2, 0)
#     return rgrad