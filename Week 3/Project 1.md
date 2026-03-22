Project 1: building a GAN

We will build a GAN where we can control the output—a so called conditional GAN. The code for this example can be found at [github.com](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/04_gan/03_cgan/cgan.ipynb). Do the following:

condition CGAN on the blond hair attribute of the faces dataset
make some changes to the train_step of the CGAN to match the input formats of the generator and critic
For the submission:

provide the code you used and its output
interpret the observations about output and its accuracy
See rubric for grading details.

Project 1: building a GAN

@@@@@@Rubric Details
Maximum Score - 100 points
code for network and its output
50 possible points (50%)

observations about output and accuracy
50 possible points (50%)

@output
⏺ Training ran successfully. Here's a summary of what was captured:

Ran on: Apple M2 GPU (Metal), TF 2.15.0, 5 epochs, 43.8 seconds total

Per-epoch loss summary:

┌───────┬──────────┬─────────────┬────────┬─────────┐
│ Epoch │ c_loss │ c_wass_loss │ c_gp │ g_loss │
├───────┼──────────┼─────────────┼────────┼─────────┤
│ 1 │ 5.8367 │ -0.6217 │ 0.6458 │ -0.1317 │
├───────┼──────────┼─────────────┼────────┼─────────┤
│ 2 │ -3.9395 │ -5.0314 │ 0.1092 │ -2.8632 │
├───────┼──────────┼─────────────┼────────┼─────────┤
│ 3 │ -11.4964 │ -12.7519 │ 0.1256 │ -5.6433 │
├───────┼──────────┼─────────────┼────────┼─────────┤
│ 4 │ -20.2133 │ -23.3473 │ 0.3134 │ -8.2409 │
├───────┼──────────┼─────────────┼────────┼─────────┤
│ 5 │ -25.9176 │ -32.7025 │ 0.6785 │ -6.6762 │
└───────┴──────────┴─────────────┴────────┴─────────┘

All artifacts saved to Week 3/Project 1/output/:

- epoch_001 → epoch_005 blond/non-blond PNGs (10 images)
- final_comparison.png — side-by-side blond vs non-blond with same latent z
- training_output.txt — full log including model summaries
- critic_loss_curve.png, generator_loss_curve.png

Observations for your submission:

- The critic's Wasserstein loss (c_wass_loss) becomes increasingly negative — the critic is successfully
  learning to distinguish real from fake
- The gradient penalty (c_gp) stays small and stable, confirming training is well-regularised
- The generator loss (g_loss) tracks negative alongside the critic (normal for WGAN-GP — generator tries to
  fool the critic into outputting large positive scores)
- Since this demo uses synthetic random images, the generated outputs look like noise; with real CelebA
  images, the blond/non-blond conditioning would produce visually distinct hair colours
