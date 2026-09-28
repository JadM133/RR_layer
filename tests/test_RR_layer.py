import pytest
import torch

from RR_layer import RRLayer
import io
import torch.nn as nn

# ============================================================
# Constructor
# ============================================================

def test_invalid_rank():
    with pytest.raises(ValueError):
        RRLayer(rank=0)


def test_invalid_basis_history_size():
    with pytest.raises(ValueError):
        RRLayer(
            rank=4,
            basis_history_size=0,
        )


# ============================================================
# Input validation
# ============================================================

def test_input_must_be_tensor():
    rr = RRLayer(rank=4)

    with pytest.raises(TypeError):
        rr([1, 2, 3])


def test_input_must_have_batch_dimension():
    rr = RRLayer(rank=4)

    x = torch.randn(10)

    with pytest.raises(ValueError):
        rr(x)


def test_empty_batch():
    rr = RRLayer(rank=4)

    x = torch.randn(0, 10)

    with pytest.raises(ValueError):
        rr(x)


def test_nan_input():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)
    x[0, 0] = torch.nan

    with pytest.raises(ValueError):
        rr(x)


def test_invalid_basis_ndim():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(16)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


def test_invalid_basis_rows():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(17, 4)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


def test_empty_basis_columns():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.empty(16, 0)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


# ============================================================
# Forward pass
# ============================================================

def test_output_shape_preserved():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 32)

    y = rr(x)

    assert y.shape == x.shape


def test_output_shape_preserved_3d():
    rr = RRLayer(rank=4)

    x = torch.randn(16, 3, 32)

    y = rr(x)

    assert y.shape == x.shape


def test_return_factors_shapes():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    y, basis, coeffs = rr(
        x,
        return_factors=True,
    )

    assert y.shape == x.shape

    assert basis.shape == (16, 4)

    assert coeffs.shape == (4, 8)


def test_factorization_reconstructs_output():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    y, basis, coeffs = rr(
        x,
        return_factors=True,
    )

    reconstructed = basis @ coeffs

    reconstructed = reconstructed.reshape(
        16,
        8,
    )

    reconstructed = reconstructed.T

    assert torch.allclose(
        y,
        reconstructed,
        atol=1e-5,
    )


# ============================================================
# Explicit basis projection
# ============================================================

def test_projection_branch_matches_formula():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(16, 4)
    basis, _ = torch.linalg.qr(basis)

    y = rr(
        x,
        basis=basis,
    )

    X = x.T

    expected = basis @ (basis.T @ X)

    expected = expected.T

    assert torch.allclose(
        y,
        expected,
        atol=1e-5,
    )


# ============================================================
# Basis collection
# ============================================================

def test_basis_collection_occurs_during_training():
    rr = RRLayer(
        rank=4,
        basis_history_size=10,
    )

    rr.train()

    for _ in range(3):
        rr(torch.randn(8, 16))

    assert len(rr._basis_bank) == 3


def test_basis_history_size_respected():
    rr = RRLayer(
        rank=4,
        basis_history_size=5,
    )

    rr.train()

    for _ in range(20):
        rr(torch.randn(8, 16))

    assert len(rr._basis_bank) == 5


# ============================================================
# Finalization
# ============================================================

def test_finalize_basis_creates_inference_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.finalize_basis()

    assert rr.inference_basis is not None

    assert rr.inference_basis.shape == (
        16,
        4,
    )


def test_finalize_without_bases_raises():
    rr = RRLayer(rank=4)

    with pytest.raises(RuntimeError):
        rr.finalize_basis()


# ============================================================
# Eval mode
# ============================================================

def test_eval_auto_finalizes_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    assert rr.inference_basis is not None


def test_eval_uses_inference_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    x = torch.randn(8, 16)

    y1 = rr(x)

    basis = rr.inference_basis

    y2 = rr(
        x,
        basis=basis,
    )

    assert torch.allclose(
        y1,
        y2,
        atol=1e-5,
    )


# ============================================================
# Gradients
# ============================================================

def test_backward_pass():
    rr = RRLayer(rank=4)

    x = torch.randn(
        8,
        16,
        requires_grad=True,
    )

    y = rr(x)

    loss = y.pow(2).mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(x.grad).all()

# ============================================================
# Train -> Eval transition
# ============================================================

def test_eval_uses_inference_basis_after_finalize():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(10):
        rr(torch.randn(8, 16))

    assert rr.inference_basis is None

    rr.eval()

    assert rr.inference_basis is not None

    x = torch.randn(8, 16)

    y_eval = rr(x)

    y_manual = rr(
        x,
        basis=rr.inference_basis,
    )

    assert torch.allclose(
        y_eval,
        y_manual,
        atol=1e-4,
        rtol=1e-4,
    )

# ============================================================
# Saving and loading
# ============================================================

def test_save_load_finalized_basis():
    """
    Regression test:
    A finalized inference_basis must survive serialization
    and load into a fresh RRLayer initialized with None.
    """
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    assert rr.inference_basis is not None

    # Save checkpoint
    buffer = io.BytesIO()
    torch.save(rr.state_dict(), buffer)

    # Create a completely fresh layer
    rr_loaded = RRLayer(rank=4)

    assert rr_loaded.inference_basis is None

    # Load checkpoint
    buffer.seek(0)
    state = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    assert "inference_basis" in state

    rr_loaded.load_state_dict(state, strict=True)

    # Verify basis
    assert rr_loaded.inference_basis is not None

    assert torch.equal(
        rr.inference_basis,
        rr_loaded.inference_basis,
    )

    # Verify identical inference results
    rr_loaded.eval()

    x = torch.randn(8, 16)

    with torch.no_grad():
        y_original = rr(x)
        y_loaded = rr_loaded(x)

    assert torch.allclose(
        y_original,
        y_loaded,
        atol=1e-6,
        rtol=1e-6,
    )


def test_save_load_without_finalized_basis():
    """
    A layer that has not been finalized should also
    save and load successfully.
    """
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(3):
        rr(torch.randn(8, 16))

    assert rr.inference_basis is None

    # Save
    buffer = io.BytesIO()
    torch.save(rr.state_dict(), buffer)

    # Load
    buffer.seek(0)

    state = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    rr_loaded = RRLayer(rank=4)

    rr_loaded.load_state_dict(state, strict=True)

    assert rr_loaded.inference_basis is None

    # The loaded layer should still support training
    x = torch.randn(8, 16)

    y = rr_loaded(x)

    assert y.shape == x.shape

    assert torch.isfinite(y).all()


def test_save_load_nested_autoencoder():
    """
    Regression test for the original problem:
    Loading an RRLayer as a submodule of an autoencoder.

    The checkpoint contains 'latent.inference_basis',
    which must not be treated as an unexpected key.
    """

    class AE(nn.Module):

        def __init__(self):
            super().__init__()

            self.encoder = nn.Linear(16, 16)

            self.latent = RRLayer(rank=4)

            self.decoder = nn.Linear(16, 16)

        def forward(self, x):
            x = self.encoder(x)
            x = self.latent(x)
            return self.decoder(x)

    # Initialize and train original model
    model = AE()

    model.train()

    for _ in range(5):
        x = torch.randn(8, 16)
        model(x)

    model.eval()

    assert model.latent.inference_basis is not None

    # Reference prediction
    x_test = torch.randn(8, 16)

    with torch.no_grad():
        y_original = model(x_test)

    # Save the complete model state
    buffer = io.BytesIO()

    torch.save(
        model.state_dict(),
        buffer,
    )

    # Create a fresh model
    model_loaded = AE()

    assert model_loaded.latent.inference_basis is None

    # Load the entire checkpoint normally
    buffer.seek(0)

    state = torch.load(
        buffer,
        map_location="cpu",
        weights_only=True,
    )

    assert "latent.inference_basis" in state

    model_loaded.load_state_dict(
        state,
        strict=True,
    )

    # Check that the basis was restored
    assert model_loaded.latent.inference_basis is not None

    assert torch.equal(
        model.latent.inference_basis,
        model_loaded.latent.inference_basis,
    )

    # Check complete autoencoder predictions
    model_loaded.eval()

    with torch.no_grad():
        y_loaded = model_loaded(x_test)

    assert torch.allclose(
        y_original,
        y_loaded,
        atol=1e-6,
        rtol=1e-6,
    )


def test_loading_replaces_existing_basis_with_different_shape():
    """
    Loading must also work when inference_basis
    already exists but has different dimensions.
    """

    # Original model with 16 features
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    assert rr.inference_basis.shape == (16, 4)

    # Save
    state = rr.state_dict()

    # Another layer previously trained on 24 features
    rr_loaded = RRLayer(rank=4)

    rr_loaded.train()

    for _ in range(5):
        rr_loaded(torch.randn(8, 24))

    rr_loaded.eval()

    assert rr_loaded.inference_basis.shape == (24, 4)

    # Load original checkpoint
    rr_loaded.load_state_dict(
        state,
        strict=True,
    )

    # Existing basis should have been replaced
    assert rr_loaded.inference_basis.shape == (16, 4)

    assert torch.equal(
        rr.inference_basis,
        rr_loaded.inference_basis,
    )

    # Inference must still work
    x = torch.randn(8, 16)

    with torch.no_grad():
        y_original = rr(x)
        y_loaded = rr_loaded(x)

    assert torch.allclose(
        y_original,
        y_loaded,
        atol=1e-6,
        rtol=1e-6,
    )
    
