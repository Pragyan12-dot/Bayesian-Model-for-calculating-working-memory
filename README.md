# Bayesian-Model-for-calculating-working-memory
Bayesian Model for calculating working memory
A hierarchical Bayesian implementation of the K–A–G model for estimating individual differences in visual working memory using R and JAGS.

Overview

This project implements a hierarchical Bayesian K–A–G model to investigate individual differences in visual working memory performance.

The model estimates three participant-specific latent parameters:

K (Storage Capacity): Represents the number of items that can be maintained in working memory.
A (Availability): Represents the probability of successfully accessing or detecting available information.
G (Guessing): Represents the baseline probability of a correct response through guessing.

A hierarchical structure is used to model participant-level parameters as arising from shared population-level distributions.

The implementation includes Bayesian inference using Markov Chain Monte Carlo (MCMC), convergence diagnostics, effective sample size estimation, and prior and posterior predictive checks.
