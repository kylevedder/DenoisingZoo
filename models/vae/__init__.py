from models.vae.vae import VAE, VAEOutput, kl_divergence

__all__ = ["VAE", "VAEOutput", "kl_divergence"]

# Optional: SD VAE (requires diffusers)
try:
    from models.vae.sd_vae import SDVAE, SDVAEOutput, load_sd_vae
    __all__.extend(["SDVAE", "SDVAEOutput", "load_sd_vae"])
except ImportError:
    pass
