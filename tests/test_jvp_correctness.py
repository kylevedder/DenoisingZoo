"""Test JVP (Jacobian-vector product) correctness.

Verifies that torch.func.jvp computes correct derivatives, which is
essential for MeanFlow loss computation.
"""

import torch
import torch.nn as nn
import pytest


class TestLinearJVP:
    """Test JVP for linear models where closed-form solution exists."""

    def test_jvp_linear_layer(self):
        """For f(x) = Wx, JVP should equal W @ v."""
        torch.manual_seed(42)
        D_in, D_out = 8, 16
        B = 4

        W = torch.randn(D_out, D_in)
        x = torch.randn(B, D_in)
        v = torch.randn(B, D_in)  # tangent vector

        def linear_fn(x):
            return x @ W.T  # (B, D_out)

        _, jvp_result = torch.func.jvp(linear_fn, (x,), (v,))

        # Expected: v @ W.T (same linear transformation)
        expected = v @ W.T

        assert torch.allclose(jvp_result, expected, atol=1e-5), \
            "JVP of linear model should be W @ tangent"

    def test_jvp_linear_layer_with_bias(self):
        """For f(x) = Wx + b, JVP should still be W @ v (bias doesn't affect derivative)."""
        torch.manual_seed(42)
        D_in, D_out = 8, 16
        B = 4

        W = torch.randn(D_out, D_in)
        b = torch.randn(D_out)
        x = torch.randn(B, D_in)
        v = torch.randn(B, D_in)

        def affine_fn(x):
            return x @ W.T + b

        _, jvp_result = torch.func.jvp(affine_fn, (x,), (v,))

        expected = v @ W.T

        assert torch.allclose(jvp_result, expected, atol=1e-5), \
            "JVP should not be affected by bias term"


class TestIdentityJVP:
    """Test JVP for identity transformation."""

    def test_jvp_identity(self):
        """For f(x) = x, JVP should equal the tangent vector v."""
        B = 4
        D = 16
        x = torch.randn(B, D)
        v = torch.randn(B, D)

        def identity_fn(x):
            return x

        _, jvp_result = torch.func.jvp(identity_fn, (x,), (v,))

        assert torch.allclose(jvp_result, v), \
            "JVP of identity should equal tangent"

    def test_jvp_scalar_multiply(self):
        """For f(x) = c*x, JVP should be c*v."""
        B = 4
        D = 16
        c = 3.5
        x = torch.randn(B, D)
        v = torch.randn(B, D)

        def scale_fn(x):
            return c * x

        _, jvp_result = torch.func.jvp(scale_fn, (x,), (v,))

        expected = c * v

        assert torch.allclose(jvp_result, expected), \
            "JVP of scalar multiply should be c*v"


class TestFiniteDifferenceComparison:
    """Compare JVP to numerical finite difference approximation."""

    def test_jvp_matches_finite_difference_linear(self):
        """JVP should match finite difference for simple model."""
        torch.manual_seed(42)
        B = 4
        D = 8

        W = torch.randn(D, D)
        x = torch.randn(B, D)
        v = torch.randn(B, D)

        def model(x):
            return torch.tanh(x @ W.T)

        # Compute JVP
        _, jvp_result = torch.func.jvp(model, (x,), (v,))

        # Compute numerical approximation
        eps = 1e-4
        f_plus = model(x + eps * v)
        f_minus = model(x - eps * v)
        numerical_jvp = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(jvp_result, numerical_jvp, atol=1e-3), \
            "JVP should match finite difference approximation"

    def test_jvp_matches_finite_difference_mlp(self):
        """JVP should match finite difference for MLP."""
        torch.manual_seed(42)
        B = 4
        D = 8

        # Simple MLP
        model = nn.Sequential(
            nn.Linear(D, 32),
            nn.ReLU(),
            nn.Linear(32, D),
        )
        model.eval()

        x = torch.randn(B, D)
        v = torch.randn(B, D)

        def forward_fn(x):
            return model(x)

        # Compute JVP
        _, jvp_result = torch.func.jvp(forward_fn, (x,), (v,))

        # Numerical approximation
        eps = 1e-4
        numerical_jvp = (forward_fn(x + eps * v) - forward_fn(x - eps * v)) / (2 * eps)

        assert torch.allclose(jvp_result, numerical_jvp, atol=1e-3), \
            "JVP should match finite difference for MLP"

    def test_jvp_matches_finite_difference_conv(self):
        """JVP should match finite difference for CNN."""
        torch.manual_seed(42)
        B = 2
        C = 3
        H = W = 16

        # Simple CNN
        model = nn.Sequential(
            nn.Conv2d(C, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, C, 3, padding=1),
        )
        model.eval()

        x = torch.randn(B, C, H, W)
        v = torch.randn(B, C, H, W)

        def forward_fn(x):
            return model(x)

        # Compute JVP
        _, jvp_result = torch.func.jvp(forward_fn, (x,), (v,))

        # Numerical approximation
        eps = 1e-4
        numerical_jvp = (forward_fn(x + eps * v) - forward_fn(x - eps * v)) / (2 * eps)

        assert torch.allclose(jvp_result, numerical_jvp, atol=1e-2), \
            "JVP should match finite difference for CNN"


class TestJVPWithTimeConditioning:
    """Test JVP in time-conditioned setting (like flow matching)."""

    def test_jvp_with_fixed_time(self):
        """JVP should only differentiate w.r.t. x, not t."""
        torch.manual_seed(42)
        B = 4
        D = 8

        W_x = torch.randn(D, D)
        W_t = torch.randn(D, 1)

        x = torch.randn(B, D)
        t = torch.rand(B, 1)
        v = torch.randn(B, D)  # tangent for x only

        def model_fn(x):
            # Model that depends on both x and t (but t is fixed)
            return torch.tanh(x @ W_x.T + t @ W_t.T)

        # JVP w.r.t. x only
        _, jvp_result = torch.func.jvp(model_fn, (x,), (v,))

        # Numerical approximation
        eps = 1e-4
        numerical_jvp = (model_fn(x + eps * v) - model_fn(x - eps * v)) / (2 * eps)

        assert torch.allclose(jvp_result, numerical_jvp, atol=1e-3), \
            "JVP should correctly differentiate w.r.t. x with fixed t"


class TestJVPChainRule:
    """Test that JVP correctly applies chain rule."""

    def test_jvp_composition(self):
        """JVP of f(g(x)) should follow chain rule."""
        torch.manual_seed(42)
        D = 8
        B = 4

        W1 = torch.randn(D, D)
        W2 = torch.randn(D, D)

        x = torch.randn(B, D)
        v = torch.randn(B, D)

        def g(x):
            return torch.relu(x @ W1.T)

        def f(y):
            return y @ W2.T

        def composed(x):
            return f(g(x))

        # JVP of composed function
        _, jvp_composed = torch.func.jvp(composed, (x,), (v,))

        # Manual chain rule: Jf(g(x)) @ Jg(x) @ v
        # For this test, just verify consistency
        y = g(x)
        _, jvp_f = torch.func.jvp(f, (y,), (v,))  # This isn't quite right

        # Verify with finite difference
        eps = 1e-4
        numerical = (composed(x + eps * v) - composed(x - eps * v)) / (2 * eps)

        assert torch.allclose(jvp_composed, numerical, atol=1e-2), \
            "JVP of composition should follow chain rule"


class TestJVPWithVelocity:
    """Test JVP specifically in the MeanFlow context."""

    def test_jvp_with_model_output_as_tangent(self):
        """In MeanFlow, tangent is the model output itself."""
        torch.manual_seed(42)
        B = 4
        D = 8

        model = nn.Sequential(
            nn.Linear(D, 32),
            nn.Tanh(),
            nn.Linear(32, D),
        )
        model.eval()

        x = torch.randn(B, D)

        # First pass: get v_t
        with torch.no_grad():
            v_t = model(x)

        # JVP with v_t as tangent (this is what MeanFlow does)
        def forward_fn(x):
            return model(x)

        _, jvp_result = torch.func.jvp(forward_fn, (x,), (v_t,))

        # Verify with finite difference
        eps = 1e-4
        numerical_jvp = (model(x + eps * v_t) - model(x - eps * v_t)) / (2 * eps)

        assert torch.allclose(jvp_result, numerical_jvp, atol=1e-2), \
            "JVP with model output as tangent should be correct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
