library(rjags)
library(coda)
library(ggplot2)
library(loo) # for WAIC
library(bayesplot) # for PPC
set.seed(123)
dat <- readRDS("GSPS.RData")
dat$working <- as.integer(dat$working == 1)
dat$id_num <- as.numeric(factor(dat$id))
N <- nrow(dat)
J <- length(unique(dat$id_num))
# Covariates for Models 2 & 3 (no intercept column)
X <- model.matrix(~ female + age + educ + married + hhkids +
                    handdum + hhninc + public, data = dat)[, -1]
K <- ncol(X)
############################################################
# 1. MODEL 1: Random intercept only (no covariates)
############################################################
cat("model {
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])
  }
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  alpha ~ dnorm(0, 0.01)
  sigma_b ~ dunif(0, 10)
  tau_b <- pow(sigma_b, -2)
}", file = "model1.jags")
data_m1 <- list(working = dat$working, id_num = dat$id_num, N = N, J = J)
jags_m1 <- jags.model("model1.jags", data = data_m1, n.chains = 3)
update(jags_m1, 2000)
samples_m1 <- coda.samples(jags_m1,
                           variable.names = c("alpha", "sigma_b", "deviance", "loglik"),
                           n.iter = 20000, thin = 5)
summary(samples_m1)
plot(samples_m1[, c("alpha", "sigma_b", "deviance")])
gelman.diag(samples_m1, multivariate = FALSE)
effectiveSize(samples_m1)
############################################################
# 2. MODEL 2: Bayesian Ridge
############################################################
cat("model {
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(beta[], X[i,]) + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])
  }
  for (k in 1:K) {
    beta[k] ~ dnorm(0, tau_beta)
  }
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  alpha ~ dnorm(0, 0.01)
  tau_beta <- pow(sd_beta, -2)
  sd_beta ~ dunif(0, 10)
  sigma_b ~ dunif(0, 10)
  tau_b <- pow(sigma_b, -2)
}", file = "model2.jags")
data_m2 <- list(working = dat$working, X = X, id_num = dat$id_num,
                N = N, J = J, K = K)
jags_m2 <- jags.model("model2.jags", data = data_m2, n.chains = 3)
update(jags_m2, 2000)
samples_m2 <- coda.samples(jags_m2,
                           variable.names = c("alpha", "beta", "sd_beta", "sigma_b", "deviance", "loglik"),
                           n.iter = 20000, thin = 5)
summary(samples_m2)
gelman.diag(samples_m2, multivariate = FALSE)
############################################################
# 3. MODEL 3: Bayesian Lasso
############################################################
cat("model {
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + inprod(beta[], X[i,]) + b[id_num[i]]
    loglik[i] <- logdensity.bern(working[i], p[i])
  }
  for (k in 1:K) {
    beta[k] ~ dnorm(0, tau_k[k])
    tau_k[k] ~ dgamma(0.5, 0.5 * lambda^2)   # Fixed: no 'rate=', use scale
  }
  for (j in 1:J) {
    b[j] ~ dnorm(0, tau_b)
  }
  alpha ~ dnorm(0, 0.01)
  lambda ~ dgamma(1, 0.1)
  sigma_b ~ dunif(0, 10)
  tau_b <- pow(sigma_b, -2)
}", file = "model3.jags")

jags_m3 <- jags.model("model3.jags", data = data_m2, n.chains = 3)
update(jags_m3, 2000)
samples_m3 <- coda.samples(jags_m3,
                           variable.names = c("alpha", "beta", "lambda", "sigma_b", "deviance", "loglik"),
                           n.iter = 20000, thin = 5)
summary(samples_m3)
gelman.diag(samples_m3, multivariate = FALSE)
############################################################
# 4. MODEL COMPARISON: DIC and WAIC
############################################################
dic1 <- dic.samples(jags_m1, n.iter = 10000)
dic2 <- dic.samples(jags_m2, n.iter = 10000)
dic3 <- dic.samples(jags_m3, n.iter = 10000)
print(dic1)
print(dic2)
print(dic3)
# WAIC - Manual extraction for coda objects
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
############################################################
# 5. POSTERIOR PREDICTIVE CHECK for Model 1
#    - Distribution overlay
#    - Test statistics: mean, SD, skewness, maximum
############################################################

cat("model {
  for (i in 1:N) {
    working[i] ~ dbern(p[i])
    logit(p[i]) <- alpha + b[id_num[i]]
    yrep[i] ~ dbern(p[i])
  }
  for (j in 1:J) { 
    b[j] ~ dnorm(0, tau_b) 
  }
  alpha ~ dnorm(0, 0.01)
  sigma_b ~ dunif(0, 10)
  tau_b <- pow(sigma_b, -2)
}", file = "ppc1.jags")

jags_ppc <- jags.model("ppc1.jags", data = data_m1, n.chains = 3)
update(jags_ppc, 1000)
ppc_samples <- coda.samples(jags_ppc, variable.names = "yrep", n.iter = 5000)  # more samples for smoother plots
yrep_mat <- as.matrix(ppc_samples)  # rows = posterior draws, cols = observations

# 1. Overlay of observed and replicated distributions (density)
bayesplot::ppc_dens_overlay(dat$working, yrep_mat[sample(nrow(yrep_mat), 50), ]) +
  labs(title = "Posterior Predictive Distribution vs Observed (Model 1)",
       subtitle = "50 replicated datasets overlaid")

# 2. Test statistics
# Mean employment rate
bayesplot::ppc_stat(dat$working, yrep_mat, stat = "mean") +
  labs(title = "Mean Employment Rate")

# Standard deviation
bayesplot::ppc_stat(dat$working, yrep_mat, stat = "sd") +
  labs(title = "Standard Deviation of Employment Status")

# Skewness (custom function needed for bayesplot)
T_skew <- function(y) {
  n <- length(y)
  mean_y <- mean(y)
  (sum((y - mean_y)^3) / n) / (sum((y - mean_y)^2) / n)^(3/2)
}

bayesplot::ppc_stat(dat$working, yrep_mat, stat = T_skew) +
  labs(title = "Skewness of Employment Indicator")
