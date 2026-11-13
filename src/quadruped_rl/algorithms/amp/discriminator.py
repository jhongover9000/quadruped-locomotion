"""AMP discriminator D(s, s') scoring whether a state transition looks like the
reference data. Least-squares GAN loss with gradient penalty; style reward
r_style = max(0, 1 - 0.25 (D(s, s') - 1)^2).

TODO(phase 4)."""
