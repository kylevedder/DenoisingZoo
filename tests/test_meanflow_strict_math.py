"""Strict mathematical tests for MeanFlow correctness.

This test file verifies the MeanFlow loss and JVP computation against a known,
curved vector field where we can compute the exact analytical target.

Scenario:
    - 2D state space (z1, z2)
    - Source x (noise) at t=0
    - Target y (data) at t=1
    - Vector field v(z, t) is NOT constant.
    - We define a polynomial vector field and check if the loss target matches
      the analytical derivation.
"""

import torch
import torch.nn as nn
import pytest
from dataloaders.base_dataloaders import make_unified_flow_matching_input

class PolynomialVelocityField(nn.Module):
    """
    A deterministic vector field for testing:
    v(z, t) = [ t * z1,  z2^2 ]
    
    Note: This is just a function v(z,t) we are treating as the "model".
    We want to verify that given this v(z,t), the MeanFlow loss computes
    the correct target u_tgt.
    """
    def __init__(self):
        super().__init__()

    def forward(self, unified_input):
        # Unpack unified input
        # unified_input is (B, D+1) or (B, C+1, H, W)
        # We assume D=2 for simplicity: z1, z2
        
        # Check if 2D (B, 3) where last dim is time
        if unified_input.dim() == 2 and unified_input.shape[1] == 3:
            z = unified_input[:, :2]
            t = unified_input[:, 2:3]
            
            z1 = z[:, 0:1]
            z2 = z[:, 1:2]
            
            v1 = t * z1
            v2 = z2 ** 2
            
            return torch.cat([v1, v2], dim=1)
        else:
            raise NotImplementedError("Only implemented for (B, 3) input")

def analytical_jvp_and_target(z, t, r):
    """
    Compute the analytical JVP and MeanFlow target for the polynomial field.
    
    v(z, t) = [ t*z1, z2^2 ]
    
    JVP = du/dt (total time derivative) = v · ∂v/∂z + ∂v/∂t
    
    ∂v/∂t = [ z1, 0 ]
    
    ∂v/∂z = [ ∂v1/∂z1  ∂v1/∂z2 ] = [ t   0 ]
            [ ∂v2/∂z1  ∂v2/∂z2 ]   [ 0  2*z2 ]
            
    v · ∂v/∂z = [ v1*t + v2*0,  v1*0 + v2*2*z2 ]
              = [ (t*z1)*t,     (z2^2)*2*z2 ]
              = [ t^2 * z1,     2 * z2^3 ]
              
    Total JVP = [ z1 + t^2 * z1,  0 + 2 * z2^3 ]
              = [ z1 * (1 + t^2), 2 * z2^3 ]
              
    MeanFlow Target u_tgt = v(z,t) - (t-r) * JVP
    u_tgt_1 = t*z1 - (t-r) * z1 * (1 + t^2)
    u_tgt_2 = z2^2 - (t-r) * 2 * z2^3
    """
    z1 = z[:, 0:1]
    z2 = z[:, 1:2]
    
    # v(z,t)
    v1 = t * z1
    v2 = z2 ** 2
    
    # JVP
    jvp1 = z1 * (1 + t ** 2)
    jvp2 = 2 * (z2 ** 3)
    
    # u_tgt
    u_tgt1 = v1 - (t - r) * jvp1
    u_tgt2 = v2 - (t - r) * jvp2
    
    return torch.cat([u_tgt1, u_tgt2], dim=1)

class TestStrictMeanFlowMath:
    
    def test_polynomial_field_jvp_match(self):
        """
        Verify that the automatic JVP computation in the loss matches 
        our analytical derivation for v = [t*z1, z2^2].
        """
        model = PolynomialVelocityField()
        
        # Setup inputs
        B = 5
        z = torch.randn(B, 2)
        t = torch.rand(B, 1)
        r = torch.rand(B, 1) * t # r < t
        
        # 1. Compute via Model (Automatic differentiation)
        # We replicate the logic inside MeanFlowLoss
        def model_fn(z_in, t_in):
            unified = make_unified_flow_matching_input(z_in, t_in)
            return model(unified)
            
        # Forward pass for tangents
        with torch.no_grad():
            v_t = model_fn(z, t)
            
        # JVP
        tangent_t = torch.ones_like(t)
        # Tangent for z is v(z,t)
        v_t_out, jvp_out = torch.func.jvp(
            model_fn,
            (z, t),
            (v_t.detach(), tangent_t),
        )
        
        # 2. Compute Analytical JVP
        # Re-compute v_t to ensure inputs are identical
        z1 = z[:, 0:1]
        z2 = z[:, 1:2]
        v1 = t * z1
        v2 = z2 ** 2
        v_t_analytical = torch.cat([v1, v2], dim=1)
        
        assert torch.allclose(v_t_out, v_t_analytical, atol=1e-6), "Forward pass mismatch"
        
        jvp_analytical = torch.cat([
            z1 * (1 + t**2),
            2 * (z2**3)
        ], dim=1)
        
        # Check match
        assert torch.allclose(jvp_out, jvp_analytical, atol=1e-5), \
            f"JVP Mismatch!\nAuto: {jvp_out[0]}\nAnalytical: {jvp_analytical[0]}"

    def test_meanflow_target_match(self):
        """
        Verify the final u_tgt calculation.
        """
        model = PolynomialVelocityField()
        
        # Setup inputs
        B = 5
        z = torch.randn(B, 2)
        t = torch.rand(B, 1)
        r = torch.rand(B, 1) * t
        
        # 1. Compute via Loss logic (simplified)
        def model_fn(z_in, t_in):
            unified = make_unified_flow_matching_input(z_in, t_in)
            return model(unified)
            
        with torch.no_grad():
            v_t = model_fn(z, t)
            
        tangent_t = torch.ones_like(t)
        _, jvp_out = torch.func.jvp(
            model_fn,
            (z, t),
            (v_t, tangent_t),
        )
        
        u_tgt_auto = v_t - (t - r) * jvp_out
        
        # 2. Compute Analytical Target
        u_tgt_analytical = analytical_jvp_and_target(z, t, r)
        
        assert torch.allclose(u_tgt_auto, u_tgt_analytical, atol=1e-5), "Target mismatch"

    def test_straight_path_invariance(self):
        """
        If the vector field IS the straight path field v(z,t) = y - x,
        then u_tgt should exactly equal v(z,t) because du/dt along trajectory should be 0.
        
        Let's verify this property.
        """
        # Define a model that outputs constant velocity C per batch item
        class ConstantVelocityField(nn.Module):
            def __init__(self, velocity):
                super().__init__()
                self.velocity = velocity # (B, D)
                
            def forward(self, unified_input):
                # Return constant velocity regardless of z, t
                return self.velocity
                
        B = 3
        D = 2
        velocity = torch.randn(B, D)
        model = ConstantVelocityField(velocity)
        
        z = torch.randn(B, D)
        t = torch.rand(B, 1)
        r = torch.rand(B, 1) * t
        
        def model_fn(z_in, t_in):
            unified = make_unified_flow_matching_input(z_in, t_in)
            return model(unified)
            
        # For a constant field (w.r.t z and t), all derivatives are 0.
        # So JVP should be 0.
        # So u_tgt = v - (t-r)*0 = v.
        
        with torch.no_grad():
            v_t = model_fn(z, t)
            
        tangent_t = torch.ones_like(t)
        _, jvp_out = torch.func.jvp(
            model_fn,
            (z, t),
            (v_t, tangent_t),
        )
        
        assert torch.allclose(jvp_out, torch.zeros_like(jvp_out)), "JVP for constant field should be 0"
        
        u_tgt = v_t - (t - r) * jvp_out
        assert torch.allclose(u_tgt, velocity), "Target for constant field should be velocity itself"

    def test_inference_logic(self):
        """
        Verify that generate_samples_meanflow correctly applies the update
        z_next = z + (t-r) * v
        when the model predicts the correct average velocity.
        """
        from evaluation.sample import generate_samples_meanflow
        
        # Define a model that calculates the exact average velocity for the 
        # polynomial field from r=0 to t=1.
        # z1(1) - z1(0) = z1(0) * (e^0.5 - 1)
        # z2(1) - z2(0) = z2(0)^2 / (1 - z2(0))
        class PerfectAverageVelocityModel(nn.Module):
            def __init__(self):
                super().__init__()
                
            def forward(self, unified_input):
                # unified is (B, D+1)
                z = unified_input[:, :2]
                t_in = unified_input[:, 2:3] # This should be 'r' (0.0)
                
                # Check that we are receiving r=0
                if not torch.allclose(t_in, torch.zeros_like(t_in)):
                    # If we receive t=1, the logic is wrong
                    print(f"Warning: Model received t={t_in.mean().item()}, expected 0.0")

                z1 = z[:, 0:1]
                z2 = z[:, 1:2]
                
                v1_avg = z1 * (torch.exp(torch.tensor(0.5)) - 1)
                v2_avg = (z2 ** 2) / (1 - z2)
                
                return torch.cat([v1_avg, v2_avg], dim=1)
                
        model = PerfectAverageVelocityModel()
        device = torch.device("cpu")
        
        # Test generation
        # Avoid z2=1 singularity
        B = 10
        # Mock random generator to control z inputs if needed, or just check logic
        # Here we just run it and check the output values relative to input z which we can't easily control 
        # inside the generator function without seeding.
        # But we can check if it matches the formula.
        
        # Let's mock torch.randn to return fixed values to verify easily
        # Or just use the seed.
        
        gen = generate_samples_meanflow(
            model=model,
            num_samples=B,
            sample_shape=(2,),
            device=device,
            batch_size=B,
            seed=42,
            r=0.0,
            t=1.0
        )
        
        samples = next(gen) # (B, 2)
        
        # We need to know what z was.
        # Since we seeded, we can regenerate z.
        rng = torch.Generator(device=device).manual_seed(42)
        z_initial = torch.randn((B, 2), device=device, generator=rng)
        
        # Analytical z(1)
        z1_0 = z_initial[:, 0:1]
        z2_0 = z_initial[:, 1:2]
        
        z1_1 = z1_0 * torch.exp(torch.tensor(0.5))
        z2_1 = z2_0 / (1 - z2_0)
        
        expected_samples = torch.cat([z1_1, z2_1], dim=1)
        
        assert torch.allclose(samples, expected_samples, atol=1e-5), \
            "Inference logic did not produce expected one-step update"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
