import warnings
import torch
import torch.nn as nn
from collections import deque

def stable_svd(A, rel_cutoff=1e-11):
    return StableSVDFunction.apply(A, rel_cutoff)
import torch


class StableSVDFunction(torch.autograd.Function):
    """
    Reduced SVD with a stabilized backward pass.

    Forward:
        A = U @ diag(S) @ Vh

    Backward:
        Avoids exploding / NaN gradients when:
          - singular values are very small
          - two singular values are equal or nearly equal
    """

    @staticmethod
    def forward(ctx, A, rel_cutoff=1e-11):

        if A.ndim != 2:
            raise ValueError(
                f"StableSVDFunction expects a 2D matrix, got {A.shape}"
            )

        # gesvd is generally more robust than the default gesvdj on CUDA
        if A.is_cuda:
            U, S, Vh = torch.linalg.svd(
                A,
                full_matrices=False,
                driver="gesvd"
            )
        else:
            U, S, Vh = torch.linalg.svd(
                A,
                full_matrices=False
            )

        ctx.save_for_backward(U, S, Vh)
        ctx.rel_cutoff = float(rel_cutoff)

        return U, S, Vh

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_Vh):

        U, S, Vh = ctx.saved_tensors
        rel_cutoff = ctx.rel_cutoff

        V = Vh.transpose(-2, -1)

        # Autograd may give None if one of the outputs
        # does not contribute to the loss.
        if grad_U is None:
            grad_U = torch.zeros_like(U)

        if grad_S is None:
            grad_S = torch.zeros_like(S)

        if grad_Vh is None:
            grad_Vh = torch.zeros_like(Vh)

        grad_V = grad_Vh.transpose(-2, -1)

        # ---------------------------------------------------------
        # 1. Stabilize very small singular values
        # ---------------------------------------------------------

        s_max = S.max()
        cutoff = rel_cutoff * s_max

        S_safe = torch.where(
            S >= cutoff,
            S,
            torch.zeros_like(S)
        )

        inv_S = torch.where(
            S_safe > 0,
            1.0 / S_safe,
            torch.zeros_like(S_safe)
        )

        # ---------------------------------------------------------
        # 2. Stabilize 1 / (s_i^2 - s_j^2)
        #
        # This is the part that normally explodes when two
        # singular values become equal / almost equal.
        # ---------------------------------------------------------

        S2 = S_safe.square()

        diff = S2[:, None] - S2[None, :]

        gap_tol = rel_cutoff * torch.clamp(
            S2.max(),
            min=torch.finfo(S.dtype).tiny
        )

        inv_diff = torch.where(
            diff.abs() > gap_tol,
            1.0 / diff,
            torch.zeros_like(diff)
        )

        # ---------------------------------------------------------
        # Standard SVD backward formula
        # ---------------------------------------------------------

        Ut_dU = U.T @ grad_U
        Vt_dV = V.T @ grad_V

        skew_U = Ut_dU - Ut_dU.T
        skew_V = Vt_dV - Vt_dV.T

        D = torch.diag(S_safe)

        middle = (
            torch.diag(grad_S)
            - (inv_diff * skew_U) @ D
            - D @ (inv_diff * skew_V)
        )

        grad_A = U @ middle @ V.T

        # ---------------------------------------------------------
        # Rectangular-matrix terms
        # ---------------------------------------------------------

        m = U.shape[0]
        n = V.shape[0]
        r = S.numel()

        D_inv = torch.diag(inv_S)

        if m > r:
            I_m = torch.eye(
                m,
                device=U.device,
                dtype=U.dtype
            )

            grad_A += (
                (I_m - U @ U.T)
                @ grad_U
                @ D_inv
                @ V.T
            )

        if n > r:
            I_n = torch.eye(
                n,
                device=V.device,
                dtype=V.dtype
            )

            grad_A += (
                U
                @ D_inv
                @ grad_V.T
                @ (I_n - V @ V.T)
            )

        return grad_A, None


import math
import warnings
from collections import deque

import torch
import torch.nn as nn


class RRLayer(nn.Module):
    r"""
    Rank Reduction (RR) layer.

    During training, the layer computes a truncated SVD across the batch
    dimension and reconstructs the input using the top ``rank`` singular
    components.

    During evaluation, the layer projects the input onto a learned inference
    basis obtained from the recent training bases.

    A custom basis can always be supplied through the ``basis`` argument of
    :meth:`forward`, in which case the train/eval behavior is bypassed.

    Args:
        rank (int):
            Number of singular values to retain.

        basis_history_size (int, optional):
            Number of recent batch bases to keep when estimating the inference
            basis. Default: 20.

    Shape:
        - Input: ``(N, *)``
        - Output: ``(N, *)``

    Example:
        >>> rr = RRLayer(rank=8)
        >>> rr.train()
        >>> y = rr(torch.randn(32, 768))

        >>> rr.eval()
        >>> y = rr(torch.randn(32, 768))
    """

    def __init__(
        self,
        rank: int,
        basis_history_size: int = 20,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(
                f"rank must be positive, got {rank}."
            )

        if basis_history_size <= 0:
            raise ValueError(
                f"basis_history_size must be positive, got "
                f"{basis_history_size}."
            )

        self.rank = rank
        self.basis_history_size = basis_history_size

        self.register_buffer(
            "inference_basis",
            None,
            persistent=True,
        )

        self._basis_bank = deque(
            maxlen=basis_history_size,
        )

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, "
            f"basis_history_size={self.basis_history_size}"
        )

    def train(self, mode: bool = True):
        """
        Switch between training and evaluation mode.

        When switching from training to evaluation for the first time,
        an inference basis is automatically computed from the stored
        training bases.
        """

        previous_mode = self.training

        super().train(mode)

        if (
            previous_mode
            and not mode
            and self.inference_basis is None
            and len(self._basis_bank) > 0
        ):
            self.finalize_basis()

        return self

    @torch.no_grad()
    def finalize_basis(self) -> None:
        """
        Build the inference basis from the stored training bases.
        """

        if len(self._basis_bank) == 0:
            raise RuntimeError(
                "Cannot finalize basis: no stored bases available."
            )

        device = next(self.parameters(), None)

        if device is None:
            device = self.inference_basis.device \
                if self.inference_basis is not None \
                else self._basis_bank[0].device
        else:
            device = device.device

        W = torch.cat(
            [
                basis.to(device)
                for basis in self._basis_bank
            ],
            dim=1,
        )

        U, _, _ = stable_SVD(W)

        r = min(self.rank, U.shape[1])

        self.inference_basis = U[:, :r]

    def _validate_inputs(
        self,
        x: torch.Tensor,
        basis: torch.Tensor | None,
    ) -> None:
        """
        Validate inputs for RRLayer.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be a torch.Tensor, got {type(x)}."
            )

        if x.ndim < 2:
            raise ValueError(
                "RRLayer expects input of shape (N, ...), "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input tensor contains NaN or Inf values."
            )

        M = math.prod(x.shape[1:])
        N = x.shape[0]

        rank_max = min(M, N)

        if self.rank > rank_max:
            warnings.warn(
                f"Requested rank={self.rank}, but the maximum "
                f"achievable rank is {rank_max}. "
                f"Using rank={rank_max}.",
                stacklevel=2,
            )

        if basis is not None:

            if not isinstance(basis, torch.Tensor):
                raise TypeError(
                    f"basis must be a torch.Tensor, got {type(basis)}."
                )

            if basis.ndim != 2:
                raise ValueError(
                    "basis must have shape (M, r), "
                    f"got shape {tuple(basis.shape)}."
                )

            if basis.shape[0] != M:
                raise ValueError(
                    f"basis has {basis.shape[0]} rows but "
                    f"expected {M}."
                )

            if basis.shape[1] == 0:
                raise ValueError(
                    "basis must contain at least one column."
                )

            if basis.device != x.device:
                raise ValueError(
                    f"basis is on {basis.device} while "
                    f"x is on {x.device}."
                )

            if basis.dtype != x.dtype:
                raise ValueError(
                    f"basis dtype ({basis.dtype}) does not match "
                    f"x dtype ({x.dtype})."
                )

            if not torch.isfinite(basis).all():
                raise ValueError(
                    "basis contains NaN or Inf values."
                )

    def forward(
        self,
        x: torch.Tensor,
        basis: torch.Tensor | None = None,
        return_factors: bool = False,
    ):
        self._validate_inputs(x, basis)

        original_shape = x.shape
        batch_size = original_shape[0]

        X = torch.movedim(x, 0, -1)
        X = X.reshape(-1, batch_size)

        if basis is not None:

            basis_used = basis

            coeffs = basis_used.T @ X

            X_hat = basis_used @ coeffs

        elif self.training:

            U, S, Vh = torch.linalg.svd(X, full_matrices=False)

            r = min(self.rank, S.shape[0])

            basis_used = U[:, :r]

            coeffs = (
                S[:r].unsqueeze(1)
                * Vh[:r]
            )

            X_hat = basis_used @ coeffs

            self._basis_bank.append(
                basis_used.detach().cpu()
            )

        else:

            if self.inference_basis is None:
                raise RuntimeError(
                    "RRLayer has no inference basis. "
                    "Call finalize_basis() or run training "
                    "before evaluation."
                )

            basis_used = self.inference_basis

            coeffs = basis_used.T @ X

            X_hat = basis_used @ coeffs

        output = X_hat.reshape(
            *original_shape[1:],
            batch_size,
        )

        output = torch.movedim(
            output,
            -1,
            0,
        )

        if return_factors:
            return output, basis_used, coeffs

        return output
