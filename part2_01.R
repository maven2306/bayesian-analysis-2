### 1 ###

# The joint posterior distribution

set.seed(2025)

# data
y <- c(4, 1, 5, 14, 3, 19, 7, 6)
t <- c(95, 16, 63, 126, 6, 32, 16, 19)
n <- length(y)

# hyperparameters (shape-rate)
alpha <- 1.8
gamma <- 0.01
delta <- 1

# MCMC settings
M <- 35000
burn <- 5000

lambda <- rep(0.05, n)  # init >0
beta   <- 1             # init >0

L <- matrix(NA_real_, nrow = M, ncol = n)
B <- numeric(M)

for (m in 1:M) {
  # lambda_i | beta, D  ~ Gamma(alpha + y_i, beta + t_i)
  lambda <- rgamma(n, shape = alpha + y, rate = beta + t)
  
  # beta | lambda, D ~ Gamma(gamma + n*alpha, delta + sum(lambda))
  beta <- rgamma(1, shape = gamma + n * alpha, rate = delta + sum(lambda))
  
  L[m, ] <- lambda
  B[m] <- beta
}

# keep post-burnin draws
L_post <- L[(burn + 1):M, , drop = FALSE]
B_post <- B[(burn + 1):M]

# quick summaries
ci <- function(x) quantile(x, c(0.025, 0.5, 0.975))
c(mean = mean(B_post), ci(B_post))
apply(L_post, 2, function(x) c(mean = mean(x), ci(x)))
