import torch
import numpy as np
import math
from typing import Callable
from typing import Any
from torch.autograd import Function, gradcheck
from torch.functional import Tensor
from torch.types import Number
from .manifolds import SymmetricPositiveDefinite

# define the epsilon precision depending on the tensor datatype
EPS = {torch.float32: 1e-4, torch.float64: 1e-7}


def ensure_sym(A: Tensor) -> Tensor:
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return 0.5 * (A + A.transpose(-1,-2))


def spd_2point_interpolation(A : Tensor, B : Tensor, t : Number) -> Tensor:
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    return rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t)) @ rm_sq


class sym_modeig:
    """Basic class that modifies the eigenvalues with an arbitrary elementwise function
    """

    @staticmethod
    def forward(M : Tensor, fun : Callable[[Tensor], Tensor], fun_param : Tensor = None,
                ensure_symmetric : bool = False, ensure_psd : bool = False) -> Tensor:
        """Modifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        M : torch.Tensor
            (batch) of symmetric matrices
        fun : Callable[[Tensor], Tensor]
            elementwise function          
        ensure_symmetric : bool = False (optional) 
            if ensure_symmetric=True, then M is symmetrized          
        ensure_psd : bool = False (optional) 
            if ensure_psd=True, then the eigenvalues are clamped so that they are > 0                  
        -------
        Returns : torch.Tensor with modified eigenvalues
        """
        if ensure_symmetric:
            M = ensure_sym(M)

        # compute the eigenvalues and vectors
        s, U = torch.linalg.eigh(M)
        if ensure_psd:
            s = s.clamp(min=EPS[s.dtype])

        # modify the eigenvalues
        smod = fun(s, fun_param)
        X = U @ torch.diag_embed(smod) @ U.transpose(-1,-2)

        return X, s, smod, U

    @staticmethod
    def backward(dX : Tensor, s : Tensor, smod : Tensor, U : Tensor, 
                    fun_der : Callable[[Tensor], Tensor], fun_der_param : Tensor = None) -> Tensor:   
        """Backpropagates the derivatives

        Source: Brooks et al. 2019, Riemannian batch normalization for SPD neural networks, NeurIPS

        Parameters
        ----------
        dX : torch.Tensor
            (batch) derivatives that should be backpropagated
        s : torch.Tensor
            eigenvalues of the original input
        smod : torch.Tensor
            modified eigenvalues
        U : torch.Tensor
            eigenvector of the input
        fun_der : Callable[[Tensor], Tensor]
            elementwise function derivative               
        -------
        Returns : torch.Tensor containing the backpropagated derivatives
        """

        # compute Lowener matrix
        # denominator
        L_den = s[...,None] - s[...,None].transpose(-1,-2)
        # find cases (similar or different eigenvalues, via threshold)
        is_eq = L_den.abs() < EPS[s.dtype]
        L_den[is_eq] = 1.0
        # case: sigma_i != sigma_j
        L_num_ne = smod[...,None] - smod[...,None].transpose(-1,-2)
        L_num_ne[is_eq] = 0
        # case: sigma_i == sigma_j
        sder = fun_der(s, fun_der_param)
        L_num_eq = 0.5 * (sder[...,None] + sder[...,None].transpose(-1,-2))
        L_num_eq[~is_eq] = 0
        # compose Loewner matrix
        L = (L_num_ne + L_num_eq) / L_den
        dM = U @  (L * (U.transpose(-1,-2) @ ensure_sym(dX) @ U)) @ U.transpose(-1,-2)
        return dM


class sym_reeig(Function):
    """
    Rectifies the eigenvalues of a batch of symmetric matrices in the tensor M (last two dimensions).
    """
    @staticmethod
    def value(s : Tensor, threshold : Tensor) -> Tensor:
        return s.clamp(min=threshold.item())

    @staticmethod
    def derivative(s : Tensor, threshold : Tensor) -> Tensor:
        return (s>threshold.item()).type(s.dtype)

    @staticmethod
    def forward(ctx: Any, M: Tensor, threshold : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_reeig.value, threshold, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, threshold)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, threshold = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_reeig.derivative, threshold), None, None

    @staticmethod
    def tests():
        """
        Basic unit tests and test to check gradients
        """
        ndim = 2
        nb = 1
        # generate random base SPD matrix
        A = torch.randn((1,ndim,ndim), dtype=torch.double)
        U, s, _ = torch.linalg.svd(A)

        threshold = torch.tensor([1e-3], dtype=torch.double)

        # generate batches
        # linear case (all eigenvalues are above the threshold)
        s = threshold * 1e1 + torch.rand((nb,ndim), dtype=torch.double) * threshold
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)

        assert (sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True))) 

        # non-linear case (some eigenvalues are below the threshold)
        s = torch.rand((nb,ndim), dtype=torch.double) * threshold
        s[::2] += threshold 
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (~sym_reeig.apply(M, threshold, False).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))   

        # linear case, all eigenvalues are identical
        s = torch.ones((nb,ndim), dtype=torch.double)
        M = U @ torch.diag_embed(s) @ U.transpose(-1,-2)
        assert (sym_reeig.apply(M, threshold, True).allclose(M))
        M.requires_grad_(True)
        assert(gradcheck(sym_reeig.apply, (M, threshold, True)))


class sym_abseig(Function):
    """
    Computes the absolute values of all eigenvalues for a batch symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.abs()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.sign()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_abseig.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_abseig.derivative), None


class sym_logm(Function):
    """
    Computes the matrix logarithm for a batch of SPD matrices.
    Ensures that the input matrices are SPD by clamping eigenvalues.
    During backprop, the update along the clamped eigenvalues is zeroed
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        # ensure that the eigenvalues are positive
        return s.clamp(min=EPS[s.dtype]).log()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        # compute derivative 
        sder = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_logm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_logm.derivative), None


class sym_expm(Function):
    """
    Computes the matrix exponential for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        return s.exp()

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_expm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_expm.derivative), None


class sym_powm(Function):
    """
    Computes the matrix power for a batch of symmetric matrices.
    """
    @staticmethod
    def value(s : Tensor, exponent : Tensor) -> Tensor:
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s : Tensor, exponent : Tensor) -> Tensor:
        return exponent * s.pow(exponent=exponent-1.)

    @staticmethod
    def forward(ctx: Any, M: Tensor, exponent : Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_powm.value, exponent, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U, exponent)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U, exponent = ctx.saved_tensors
        dM = sym_modeig.backward(dX, s, smod, U, sym_powm.derivative, exponent)

        dXs = (U.transpose(-1,-2) @ ensure_sym(dX) @ U).diagonal(dim1=-1,dim2=-2)
        dexp = dXs * smod * s.log()

        return dM, dexp, None


class sym_sqrtm(Function):
    """
    Computes the matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).sqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = 0.5 * s.rsqrt()
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_sqrtm.derivative), None


class sym_invsqrtm(Function):
    """
    Computes the inverse matrix square root for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).rsqrt()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invsqrtm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invsqrtm.derivative), None


class sym_invsqrtm2(Function):
    """
    Computes the square root and inverse square root matrices for a batch of SPD matrices.
    """

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        Xsq, s, smod, U = sym_modeig.forward(M, sym_sqrtm.value, ensure_symmetric=ensure_symmetric)
        smod2 = sym_invsqrtm.value(s)
        Xinvsq = U @ torch.diag_embed(smod2) @ U.transpose(-1,-2)
        ctx.save_for_backward(s, smod, smod2, U)
        return Xsq, Xinvsq

    @staticmethod
    def backward(ctx: Any, dXsq: Tensor, dXinvsq: Tensor):
        s, smod, smod2, U = ctx.saved_tensors
        dMsq = sym_modeig.backward(dXsq, s, smod, U, sym_sqrtm.derivative)
        dMinvsq = sym_modeig.backward(dXinvsq, s, smod2, U, sym_invsqrtm.derivative)

        return dMsq + dMinvsq, None


class sym_invm(Function):
    """
    Computes the inverse matrices for a batch of SPD matrices.
    """
    @staticmethod
    def value(s : Tensor, param:Tensor = None) -> Tensor:
        return s.clamp(min=EPS[s.dtype]).reciprocal()

    @staticmethod
    def derivative(s : Tensor, param:Tensor = None) -> Tensor:
        sder = -1. * s.pow(-2)
        # pick subgradient 0 for clamped eigenvalues
        sder[s<=EPS[s.dtype]] = 0
        return sder

    @staticmethod
    def forward(ctx: Any, M: Tensor, ensure_symmetric : bool = False) -> Tensor:
        X, s, smod, U = sym_modeig.forward(M, sym_invm.value, ensure_symmetric=ensure_symmetric)
        ctx.save_for_backward(s, smod, U)
        return X

    @staticmethod
    def backward(ctx: Any, dX: Tensor):
        s, smod, U = ctx.saved_tensors
        return sym_modeig.backward(dX, s, smod, U, sym_invm.derivative), None


def spd_mean_kracher_flow(X : Tensor, G0 : Tensor = None, maxiter : int = 50, dim = 0, weights = None, return_dist = False, return_XT = False) -> Tensor:
    """
    Computes the mean and variance for a batch of SPD matrices.
    """

    if X.shape[dim] == 1:
        if return_dist:
            return X, torch.tensor([0.0], dtype=X.dtype, device=X.device)
        else:
            return X

    if weights is None:
        n = X.shape[dim]
        weights = torch.ones((*X.shape[:-2], 1, 1), dtype=X.dtype, device=X.device)
        weights /= n

    if G0 is None:
        G = (X * weights).sum(dim=dim, keepdim=True)
    else:
        G = G0.clone()

    nu = 1.
    dist = tau = crit = torch.finfo(X.dtype).max
    i = 0

    while (crit > EPS[X.dtype]) and (i < maxiter) and (nu > EPS[X.dtype]):
        i += 1

        Gsq, Ginvsq = sym_invsqrtm2.apply(G)
        XT = sym_logm.apply(Ginvsq @ X @ Ginvsq)
        GT = (XT * weights).sum(dim=dim, keepdim=True)
        G = Gsq @ sym_expm.apply(nu * GT) @ Gsq

        if return_dist:
            dist = torch.norm(XT - GT, p='fro', dim=(-2,-1))
        crit = torch.norm(GT, p='fro', dim=(-2,-1)).max()
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    if return_dist:
        return G, dist
    if return_XT:
        return G, XT
    return G
 

def get_label_ratio(labels,ratio_level):
     labs = labels.unique()
     labs = labs.tolist()
     labels_num = len(labs)
     source_imbalance_ratio_dict = {}
     target_imbalance_ratio_dict = {}
     source_sample_sizes = [1] * labels_num
     target_sample_sizes = [ratio_level] * (labels_num-1) + [1]    

     source_sample_size_iter = iter(source_sample_sizes)
     target_sample_size_iter = iter(target_sample_sizes)

     for _, clss in enumerate(labs): 
        source_imbalance_ratio_dict[clss] = next(source_sample_size_iter)
        target_imbalance_ratio_dict[clss] = next(target_sample_size_iter)


     return source_imbalance_ratio_dict, target_imbalance_ratio_dict

# kmeans clustering on the SPD manifold
def kmeans(X,k,init,max_iter):

    spd = SymmetricPositiveDefinite()
    centroids = init.clone()
    prev_labels = None
    counter = 0
    for _ in range(max_iter):
        
        counter += 1
        distances = np.zeros((X.shape[0], centroids.shape[0]))

        # 计算每个样本点到聚类中心的距离
        for i in range(X.shape[0]):
             for j in range(centroids.shape[0]):
                distances[i, j] = spd.dist(X[i], centroids[j], keepdim=True)
        
        # 分配每个样本到最近的聚类中心
        labels = np.argmin(distances, axis=1)


        # 如果聚类中心不再变化，提前结束迭代
        if np.array_equal(labels, prev_labels):
            break

        prev_labels = labels.copy()

        # 更新聚类中心
        new_centroids = [spd_mean_kracher_flow(X[labels==j]) for j in range(k)]

        centroids = torch.cat(new_centroids)

    return labels 
















