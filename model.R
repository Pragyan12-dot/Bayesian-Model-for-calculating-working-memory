# -----------------------------------------------------------------

# 1. Load Packages --------------------------------------------------
library(rjags)
library(coda)
library(ggplot2)

# 2. Generate Fake Data --------------------------------------------
set.seed(123)
S <- c(5, 5, 5, 5)                 # set sizes
C <- c(0, 1, 2, 5)                 # changed objects
N <- rep(60, 4)                    # trials per condition

# five synthetic participants
data <- matrix(c(10, 42, 45, 55,
                 5,  37, 45, 59,
                 3,  46, 48, 56,
                 3,  32, 42, 50,
                 1,  38, 49, 53),
               nrow = 5, byrow = TRUE)

# 3. Bundle Data for JAGS ------------------------------------------
jags_data <- list(
  S = S,
  C = C,
  N_trials = N,
  y = data,
  N_participants = nrow(data),
  N_trialtypes = length(S),
  K_max = max(S),
  idx1 = S - C + 1,
  idx2 = S,
  maxS = max(S)
)

# 4. JAGS Model -----------------------------------------------------
model_string <- "
model {
  # ---------- participant level ----------
  for (i in 1:N_participants) {
    K[i] ~ dnorm(mu_K, tau_K) T(0, K_max)
    A[i] ~ dbeta(alpha_A, beta_A)
    G[i] ~ dbeta(alpha_G, beta_G)

    for (t in 1:N_trialtypes) {

      # compute product term for detection probability
      for (j in 1:maxS) {
        log_term[i,t,j] <- step(j - idx1[t]) * step(idx2[t] - j) *
                           log(max(1.0E-9 , (j - K[i]) / j))
      }
      logProd[i,t]  <- sum(log_term[i,t,1:maxS])
      prod_term[i,t] <- exp(logProd[i,t])

      D[i,t] <- equals(C[t], 0) * 0 +
                step(C[t] - (S[t] - K[i])) * 1 +
                (1 - equals(C[t], 0)) *
                (1 - step(C[t] - (S[t] - K[i]))) * (1 - prod_term[i,t])

      p[i,t] <- G[i] + (1 - G[i]) * A[i] * D[i,t]
      y[i,t] ~ dbin(p[i,t], N_trials[t])
    }
  }

  # ---------- hyper‑priors (psychotic profile) ----------
  mu_K    ~ dnorm(2.0, 1)                 # mean 2, sd 1, truncated implicitly by K T(0,5)
  sigma_K ~ dunif(0.1, 2)
  tau_K   <- pow(sigma_K, -2)

  alpha_A ~ dgamma(3, 1)                  # Beta mean 0.5
  beta_A  ~ dgamma(3, 1)

  alpha_G ~ dgamma(2, 1)                  # Beta mean 0.2
  beta_G  ~ dgamma(8, 1)
}
"

# 5. Compile & Run MCMC --------------------------------------------
model_file <- tempfile(fileext = ".bug")
writeLines(model_string, model_file)

jags_model <- jags.model(model_file, data = jags_data,
                         n.chains = 3, n.adapt = 1000)

update(jags_model, n.iter = 2000)  # burn‑in

samples <- coda.samples(jags_model,
                        variable.names = c("K","A","G","mu_K","sigma_K",
                                           "alpha_A","beta_A","alpha_G","beta_G"),
                        n.iter = 6000, thin = 3)

# 6. MCMC Diagnostics ----------------------------------------------
print(summary(samples))
cat("\nGelman‑Rubin R‑hat:\n"); print(gelman.diag(samples))
cat("\nEffective sample sizes:\n"); print(effectiveSize(samples))

# Optional trace / ACF
# plot(samples); acfplot(samples[,1:6])

# 7. Prior Predictive Check (consistent with priors) ----------------
set.seed(42)
N_participants <- 5

# draw hyper‑params from SAME priors
mu_K    <- rnorm(1, 2.0, 1)
sigma_K <- runif(1, 0.1, 2)
tau_K   <- 1 / sigma_K^2

alpha_A <- rgamma(1, 3, 1)
beta_A  <- rgamma(1, 3, 1)

alpha_G <- rgamma(1, 2, 1)
beta_G  <- rgamma(1, 8, 1)

# participant‑level draws
K_prior <- pmin(pmax(rnorm(N_participants, mu_K, sqrt(1/tau_K)), 0), max(S))
A_prior <- rbeta(N_participants, alpha_A, beta_A)
G_prior <- rbeta(N_participants, alpha_G, beta_G)

# compute predicted probabilities
pred_p_prior <- matrix(NA, nrow=N_participants, ncol=length(S))
for (i in 1:N_participants) {
  for (t in 1:length(S)) {
    if (C[t]==0) {
      D <- 0
    } else {
      idx <- (S[t]-C[t]+1):S[t]
      prod_term <- prod((idx - K_prior[i]) / idx)
      D <- ifelse(C[t] > S[t]-K_prior[i], 1, 1-prod_term)
    }
    pred_p_prior[i,t] <- G_prior[i] + (1 - G_prior[i]) * A_prior[i] * D
  }
}
colnames(pred_p_prior) <- paste0("C=", C)
cat("\n📊 Prior Predictive Check (new priors):\n")
print(round(pred_p_prior, 2))

# 8. Posterior Predictive Check ------------------------------------
post_mat <- as.matrix(samples)
K_hat <- colMeans(post_mat[, grep("^K\\[", colnames(post_mat))])
A_hat <- colMeans(post_mat[, grep("^A\\[", colnames(post_mat))])
G_hat <- colMeans(post_mat[, grep("^G\\[", colnames(post_mat))])

pred_p_post <- matrix(NA, nrow=nrow(data), ncol=length(S))
for (i in 1:nrow(data)) {
  for (t in 1:length(S)) {
    if (C[t]==0) {
      D <- 0
    } else {
      idx <- (S[t]-C[t]+1):S[t]
      prod_term <- prod((idx - K_hat[i]) / idx)
      D <- ifelse(C[t] > S[t]-K_hat[i], 1, 1-prod_term)
    }
    pred_p_post[i,t] <- G_hat[i] + (1 - G_hat[i]) * A_hat[i] * D
  }
}

obs_prop <- data / matrix(N, nrow=nrow(data), ncol=length(N), byrow=TRUE)

cat("\n📊 Posterior Predictive Check (Observed vs. Predicted):\n")
for (i in 1:nrow(data)) {
  cat(sprintf("\nParticipant %d:\n", i))
  for (t in 1:length(S)) {
    cat(sprintf("  C=%d: Obs=%.2f, Pred=%.2f\n",
                C[t], obs_prop[i,t], pred_p_post[i,t]))
  }
}

# quick scatter plot
plot(obs_prop, pred_p_post,
     xlab="Observed Proportion", ylab="Predicted Proportion",
     main="Posterior Predictive Check", col="blue", pch=19)
abline(0,1,col="red", lty=2)
