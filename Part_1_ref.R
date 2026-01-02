library(rjags)
library(coda)
library(ggplot2)
library(loo)       # for WAIC (model comparison, ref:DIC and model comparison)
library(bayesplot) # for PPC (posterior predictive checks, ref: Predictive approach to GOF testing)
library(ggmcmc)    # for caterpillar plots (convergence diagnostics, ref: Convergence diagnostics in CODA)
library(MCMCvis)   # for additional summaries and traces (convergence diagnostics)
library(dplyr)     # for data summarization (e.g., nobs per subject)

set.seed(123)

dat <- readRDS("/Users/alejandrina/Desktop/Bayesian_analysis/GSPS.RData")
dat$working <- as.integer(dat$working == 1)
dat$id_num <- as.numeric(factor(dat$id))
N <- nrow(dat)  # Total observations
J <- length(unique(dat$id_num))  # Number of subjects

# Compute nobs: number of observations per subject (for shrinkage B, ref:Repeated measurements over time)
nobs <- dat %>% group_by(id_num) %>% summarise(n = n()) %>% pull(n)

# Covariates for Models 2 & 3 (no intercept column)
X <- model.matrix(~ female + age + educ + married + hhkids +
                    handdum + hhninc + public, data = dat)[, -1]
K <- ncol(X)  # Number of predictors (for variable selection in ridge/lasso, ref:LASSO prior for variable selection)

############################################################
# 1. MODEL 1: Random intercept only (no covariates)
# Ref: Chapter 6 Hierarchical models; Gaussian hierarchical models (adapted to logistic)
############################################################
cat("model {
  # Likelihood - Level 1 (ref: Page 235 - Level 1 in hierarchical models)
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])  # For WAIC/DIC (ref:DIC for model comparison)
  }
  
  # Distribution of random effects - Level 2 (ref:Level 2 in hierarchical models)
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  
  # Priors (ref: Noninformative priors; Priors for parameters)
  alpha ~ dnorm(0, 0.0001)  # Vague normal
  sigma_b ~ dunif(0, 100)   # Uniform for SD
  tau_b <- pow(sigma_b, -2)
  
  # Shrinkage factor B
  for (j in 1:J) {
    B[j] <- tau_b / (tau_b + nobs[j])  # Approximate, since logistic variance ~1
  }
  
  # Intra-class correlation r 
  r <- pow(sigma_b, 2) / (pow(sigma_b, 2) + 3.2899)  # pi^2/3 ≈ 3.29
  
  # Predictions for PPC (ref: Posterior predictive check with T(y) vs T(~y))
  for (i in 1:N) {
    predict[i] ~ dbern(p[i])
  }
  mpredict <- mean(predict[])  # Mean prediction
  
  # New subject prediction 
  b_new ~ dnorm(0, tau_b)
  for (k in 1:nnew) {
    logit(p_new[k]) <- alpha + b_new  # Hypothetical new obs (no time/covariates in Model 1)
    predict_new[k] ~ dbern(p_new[k])
  }
  mpredict_new <- mean(predict_new[])
}", file = "model1.jags")

# Data list (add nobs, nnew, Nsubj)
data_m1 <- list(working = dat$working, id_num = dat$id_num, N = N, J = J, nobs = nobs, nnew = 10)

# Initial values (varied across chains, ref: Scripts 1,5,6)
inits_m1 <- list(
  list(alpha = 0, sigma_b = 1),
  list(alpha = 1, sigma_b = 10),
  list(alpha = -1, sigma_b = 5)
)

jags_m1 <- jags.model("model1.jags", data = data_m1, inits = inits_m1, n.chains = 3)
update(jags_m1, 1000)  # Burn-in 

# Monitor expanded parameters (ref: Bayes theorem for posterior; Convergence diagnostics)
samples_m1 <- coda.samples(jags_m1,
                           variable.names = c("alpha", "sigma_b", "r", "B", "mpredict", "mpredict_new", "loglik"),
                           n.iter = 30000, thin = 3)  # Thinning (ref: Thinning in computations)

# Summaries and diagnostics (ref: Convergence diagnostics: traceplots, autocorr, Geweke)
summary(samples_m1)
plot(samples_m1[, c("alpha", "sigma_b")])  # Trace and density (ref: Trace plots)
gelman.diag(samples_m1, multivariate = FALSE)  # Univariate BGR only (ref: Formal diagnostics)
geweke.diag(samples_m1)  # Geweke (ref: Page 411)
autocorr.plot(samples_m1[, "alpha"], lag.max = 30)  # Autocorr
effectiveSize(samples_m1)  # Effective sample size 

# Ergodic mean plot
runmean <- function(x) {
  nlen <- length(x)
  mean_iteration <- numeric(nlen)
  mean_iteration[1] <- x[1]
  for(j in 2:nlen) mean_iteration[j] <- mean(x[1:j])
  return(mean_iteration)
}
plot(1:nrow(samples_m1[[1]]), runmean(samples_m1[[1]][,"alpha"]), type="l", xlab="Iterations", ylab="Ergodic mean alpha")

# QQ-plot
alpha_samples <- samples_m1[[1]][,"alpha"]
half1 <- alpha_samples[1:(length(alpha_samples)/2)]
half2 <- alpha_samples[(length(alpha_samples)/2 + 1):length(alpha_samples)]
qqplot(half1, half2, xlab="First half alpha", ylab="Second half alpha")

# ggmcmc caterpillar 
out_ggs1 <- ggs(samples_m1)
ggs_caterpillar(out_ggs1, family = "^B", sort = TRUE)  # For shrinkage B

# MCMCvis 
MCMCsummary(samples_m1, params = c("alpha", "sigma_b", "r"), round = 3, HPD = TRUE)
MCMCtrace(samples_m1, params = c("alpha", "sigma_b"), pdf = FALSE, ind = TRUE)

# bayesplot 
mcmc_areas(samples_m1, pars = vars(alpha, sigma_b), prob = 0.8)
mcmc_intervals(samples_m1, pars = vars(alpha, sigma_b), prob = 0.8)

############################################################
# 2. MODEL 2: Bayesian Ridge
# Ref: Ridge/LASSO priors for variable selection; hierarchical prior for shrinkage
############################################################
cat("model {
  # Likelihood - Level 1
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(beta[], X[i,]) + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])
  }
  
  # Ridge prior for betas (ref: Priors for variable selection; common tau_beta)
  for (k in 1:K) {
    beta[k] ~ dnorm(0, tau_beta)
  }
  
  # Level 2 random effects
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  
  # Priors
  alpha ~ dnorm(0, 0.0001)
  sd_beta ~ dunif(0, 100)  # Uniform for ridge SD
  tau_beta <- pow(sd_beta, -2)
  sigma_b ~ dunif(0, 100)
  tau_b <- pow(sigma_b, -2)
  
  # Shrinkage B
  for (j in 1:J) {
    B[j] <- tau_b / (tau_b + nobs[j])
  }
  
  # ICC r
  r <- pow(sigma_b, 2) / (pow(sigma_b, 2) + 3.2899)
  
  # Predictions
  for (i in 1:N) {
    predict[i] ~ dbern(p[i])
  }
  mpredict <- mean(predict[])
  
  # New prediction
  b_new ~ dnorm(0, tau_b)
  for (k in 1:nnew) {
    logit(p_new[k]) <- alpha + b_new
    predict_new[k] ~ dbern(p_new[k])
  }
  mpredict_new <- mean(predict_new[])
}", file = "model2.jags")

data_m2 <- list(working = dat$working, X = X, id_num = dat$id_num,
                N = N, J = J, K = K, nobs = nobs, nnew = 10)

inits_m2 <- list(
  list(alpha = 0, sd_beta = 1, sigma_b = 1, beta = rep(0, K)),
  list(alpha = 1, sd_beta = 10, sigma_b = 10, beta = rep(0, K)),
  list(alpha = -1, sd_beta = 5, sigma_b = 5, beta = rep(0, K))
)

jags_m2 <- jags.model("model2.jags", data = data_m2, inits = inits_m2, n.chains = 3)
update(jags_m2, 1000)
samples_m2 <- coda.samples(jags_m2,
                           variable.names = c("alpha", "beta", "sd_beta", "sigma_b", "r", "B", "mpredict", "mpredict_new", "loglik"),
                           n.iter = 30000, thin = 3)

# Diagnostics 
summary(samples_m2)
plot(samples_m2[, c("alpha", "sd_beta", "sigma_b")])
gelman.diag(samples_m2,multivariate = FALSE)
geweke.diag(samples_m2)
autocorr.plot(samples_m2[, "alpha"], lag.max = 30)
effectiveSize(samples_m2)
plot(1:nrow(samples_m2[[1]]), runmean(samples_m2[[1]][,"alpha"]), type="l", xlab="Iterations", ylab="Ergodic mean alpha")
alpha_samples <- samples_m2[[1]][,"alpha"]
half1 <- alpha_samples[1:(length(alpha_samples)/2)]
half2 <- alpha_samples[(length(alpha_samples)/2 + 1):length(alpha_samples)]
qqplot(half1, half2, xlab="First half alpha", ylab="Second half alpha")
out_ggs2 <- ggs(samples_m2)
ggs_caterpillar(out_ggs2, family = "^beta", sort = TRUE)
MCMCsummary(samples_m2, params = c("alpha", "beta", "sd_beta", "sigma_b", "r"), round = 3, HPD = TRUE)
MCMCtrace(samples_m2, params = "beta", pdf = FALSE, ind = TRUE)
mcmc_areas(samples_m2, pars = vars(starts_with("beta")), prob = 0.8)
mcmc_intervals(samples_m2, pars = vars(starts_with("beta")), prob = 0.8)

############################################################
# 3. MODEL 3: Bayesian Lasso
# Ref: LASSO prior (normal-exponential mixture for variable selection)
############################################################
cat("model {
  # Likelihood - Level 1
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(beta[], X[i,]) + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])
  }
  
  # Lasso prior for betas (ref: LASSO prior via gamma on tau_k)
  for (k in 1:K) {
    beta[k] ~ dnorm(0, tau_k[k])
    tau_k[k] ~ dgamma(0.5, 0.5 * lambda^2)
  }
  
  # Level 2
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  
  # Priors
  alpha ~ dnorm(0, 0.0001)
  lambda ~ dgamma(1, 0.1)  # For lasso shrinkage (ref:Priors directing variable search)
  sigma_b ~ dunif(0, 100)
  tau_b <- pow(sigma_b, -2)
  
  # Shrinkage B
  for (j in 1:J) {
    B[j] <- tau_b / (tau_b + nobs[j])
  }
  
  # ICC r
  r <- pow(sigma_b, 2) / (pow(sigma_b, 2) + 3.2899)
  
  # Predictions
  for (i in 1:N) {
    predict[i] ~ dbern(p[i])
  }
  mpredict <- mean(predict[])
  
  # New prediction
  b_new ~ dnorm(0, tau_b)
  for (k in 1:nnew) {
    logit(p_new[k]) <- alpha + b_new
    predict_new[k] ~ dbern(p_new[k])
  }
  mpredict_new <- mean(predict_new[])
}", file = "model3.jags")

jags_m3 <- jags.model("model3.jags", data = data_m2, inits = inits_m2, n.chains = 3)  # Reuse inits_m2
update(jags_m3, 1000)
samples_m3 <- coda.samples(jags_m3,
                           variable.names = c("alpha", "beta", "lambda", "sigma_b", "r", "B", "mpredict", "mpredict_new", "loglik"),
                           n.iter = 30000, thin = 3)

# Diagnostics
summary(samples_m3)
plot(samples_m3[, c("alpha", "lambda", "sigma_b")])
gelman.diag(samples_m3,multivariate = FALSE)
geweke.diag(samples_m3)
autocorr.plot(samples_m3[, "alpha"], lag.max = 30)
effectiveSize(samples_m3)
plot(1:nrow(samples_m3[[1]]), runmean(samples_m3[[1]][,"alpha"]), type="l", xlab="Iterations", ylab="Ergodic mean alpha")
alpha_samples <- samples_m3[[1]][,"alpha"]
half1 <- alpha_samples[1:(length(alpha_samples)/2)]
half2 <- alpha_samples[(length(alpha_samples)/2 + 1):length(alpha_samples)]
qqplot(half1, half2, xlab="First half alpha", ylab="Second half alpha")
out_ggs3 <- ggs(samples_m3)
ggs_caterpillar(out_ggs3, family = "^beta", sort = TRUE)
MCMCsummary(samples_m3, params = c("alpha", "beta", "lambda", "sigma_b", "r"), round = 3, HPD = TRUE)
MCMCtrace(samples_m3, params = "beta", pdf = FALSE, ind = TRUE)
mcmc_areas(samples_m3, pars = vars(starts_with("beta")), prob = 0.8)
mcmc_intervals(samples_m3, pars = vars(starts_with("beta")), prob = 0.8)

############################################################
# 4. MODEL COMPARISON: DIC and WAIC
# Ref: DIC for hierarchical models; WAIC as advanced comparison 
############################################################
dic1 <- dic.samples(jags_m1, n.iter = 10000)
dic2 <- dic.samples(jags_m2, n.iter = 10000)
dic3 <- dic.samples(jags_m3, n.iter = 10000)
print(dic1)
print(dic2)
print(dic3)

# WAIC
loglik1 <- do.call(rbind, lapply(samples_m1, function(chain) as.matrix(chain[, grep("loglik", colnames(chain))])))
loglik2 <- do.call(rbind, lapply(samples_m2, function(chain) as.matrix(chain[, grep("loglik", colnames(chain))])))
loglik3 <- do.call(rbind, lapply(samples_m3, function(chain) as.matrix(chain[, grep("loglik", colnames(chain))])))
waic1 <- waic(loglik1)
waic2 <- waic(loglik2)
waic3 <- waic(loglik3)
print(waic1)
print(waic2)
print(waic3)
loo_compare(waic1, waic2, waic3)

# PPC visualization for Model 1 (ref: Compare T(y) vs T(~y))
par(mfrow=c(1,2))
densplot(samples_m1[,"mpredict"])
abline(v = mean(dat$working), col="red")  # Observed mean
densplot(samples_m1[,"mpredict_new"])
abline(v = mean(samples_m1[[1]][,"alpha"]), col="red")  # Approximate

############################################################
# 5. POSTERIOR PREDICTIVE CHECK for Model 1
# Ref: Page 382 - PPC with distribution and discrepancies
############################################################
cat("model {
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + b[id_num[i]]
    yrep[i] ~ dbern(p[i])
  }
  for (j in 1:J) { b[j] ~ dnorm(0, tau_b) }
  alpha ~ dnorm(0, 0.0001)
  sigma_b ~ dunif(0, 100)
  tau_b <- pow(sigma_b, -2)
}", file = "ppc1.jags")

jags_ppc <- jags.model("ppc1.jags", data = data_m1, n.chains = 3)
update(jags_ppc, 1000)
ppc_samples <- coda.samples(jags_ppc, "yrep", n.iter = 5000)
yrep_mat <- as.matrix(ppc_samples)

# Distribution overlay
ppc_dens_overlay(dat$working, yrep_mat[sample(nrow(yrep_mat), 50), ]) +
  labs(title = "PPC: Observed vs Replicated (Model 1)")

# Test statistics
ppc_stat(dat$working, yrep_mat, stat = "mean") + labs(title = "Mean")
ppc_stat(dat$working, yrep_mat, stat = "sd")   + labs(title = "SD")

T_skew <- function(y) {
  (sum((y - mean(y))^3) / length(y)) / (sum((y - mean(y))^2) / length(y))^(3/2)
}
ppc_stat(dat$working, yrep_mat, stat = T_skew) + labs(title = "Skewness")

# Densplot of predicted mean
densplot(samples_m1[,"mpredict"])
abline(v = mean(dat$working), col = "red", lwd = 2)