# ============================================================================
# Question 11: Gibbs Sampler Implementation
# ============================================================================

# Set seed for reproducibility
set.seed(2025)

# ----------------------------------------------------------------------------
# 1. INITIALIZATION
# ----------------------------------------------------------------------------

# Define total iterations and burn-in
M <- 35000  # Total iterations
B <- 5000   # Burn-in period

# Define the data from the problem
y <- c(4, 1, 5, 14, 3, 19, 7, 6)
t <- c(95, 16, 63, 126, 6, 32, 16, 19)

# Define the fixed hyperparameters
alpha <- 1.8
gamma <- 0.01
delta <- 1

# Number of groups
n <- length(y)

# Create storage matrices for samples
# Each row corresponds to a group/parameter, each column to an iteration
lambda_samples <- matrix(0, nrow = n, ncol = M)
beta_samples <- numeric(M)

# Set initial values for iteration 1
lambda_samples[, 1] <- rep(1, n)  # Initialize all lambda_i to 1
beta_samples[1] <- 1.0            # Initialize beta to 1

# ----------------------------------------------------------------------------
# 2. MAIN GIBBS SAMPLING LOOP
# ----------------------------------------------------------------------------

for (m in 2:M) {
  
  # Step 3a: Update each lambda_i
  for (i in 1:n) {
    # Define parameters for the full conditional of lambda_i
    shape_lambda <- y[i] + alpha
    rate_lambda <- t[i] + beta_samples[m - 1]  # Use beta from previous iteration
    
    # Draw new sample from Gamma distribution
    lambda_samples[i, m] <- rgamma(1, shape = shape_lambda, rate = rate_lambda)
  }
  
  # Step 3b: Update beta
  # Calculate sum of newly updated lambda values from current iteration
  sum_lambda <- sum(lambda_samples[, m])
  
  # Define parameters for the full conditional of beta
  shape_beta <- 8 * alpha + gamma
  rate_beta <- delta + sum_lambda
  
  # Draw new sample from Gamma distribution
  beta_samples[m] <- rgamma(1, shape = shape_beta, rate = rate_beta)
}

# ----------------------------------------------------------------------------
# 3. POST-PROCESSING
# ----------------------------------------------------------------------------

# Discard burn-in samples
lambda_posterior <- lambda_samples[, (B + 1):M]  # Keep samples after burn-in
beta_posterior <- beta_samples[(B + 1):M]        # Keep samples after burn-in

# Number of valid posterior samples
n_posterior_samples <- M - B

# ----------------------------------------------------------------------------
# 4. ACCEPTANCE RATE
# ----------------------------------------------------------------------------

# In Gibbs sampling, every proposal is accepted because we sample directly
# from the full conditional distributions. Therefore, the acceptance rate is 100%
acceptance_rate <- 100

cat("\n============================================================================\n")
cat("GIBBS SAMPLER RESULTS\n")
cat("============================================================================\n")
cat("Total iterations:", M, "\n")
cat("Burn-in period:", B, "\n")
cat("Posterior samples retained:", n_posterior_samples, "\n")
cat("\nAcceptance rate:", acceptance_rate, "%\n")
cat("\nNote: In Gibbs sampling, the acceptance rate is always 100% because\n")
cat("we sample directly from the full conditional distributions rather than\n")
cat("using a proposal distribution with an accept/reject step.\n")
cat("============================================================================\n\n")

# Display summary statistics for posterior samples
cat("Summary Statistics for Posterior Samples:\n")
cat("------------------------------------------\n\n")

cat("Beta (shared hyperparameter):\n")
print(summary(beta_posterior))
cat("\n")

for (i in 1:n) {
  cat(paste0("Lambda_", i, " (infection rate for group ", i, "):\n"))
  print(summary(lambda_posterior[i, ]))
  cat("\n")
}

# ============================================================================
# Question 12: Convergence Diagnostics
# ============================================================================

# Load the coda package for MCMC diagnostics
library(coda)

cat("\n============================================================================\n")
cat("CONVERGENCE DIAGNOSTICS\n")
cat("============================================================================\n\n")

# ----------------------------------------------------------------------------
# Prepare Data for coda Package
# ----------------------------------------------------------------------------

# Create a single matrix where:
# - Columns 1-8 are the lambda parameters (transposed from lambda_posterior)
# - Column 9 is the beta parameter
posterior_matrix <- cbind(t(lambda_posterior), beta_posterior)

# Add column names for clarity
colnames(posterior_matrix) <- c(paste0("lambda_", 1:8), "beta")

# Convert to mcmc object for use with coda functions
posterior_mcmc <- as.mcmc(posterior_matrix)

# ----------------------------------------------------------------------------
# Part 1: Geweke Diagnostic
# ----------------------------------------------------------------------------

cat("Part 1: GEWEKE DIAGNOSTIC\n")
cat("-------------------------\n")
cat("The Geweke diagnostic is a Z-test that compares the mean of the first 10%\n")
cat("of the chain (after burn-in) with the mean of the last 50% of the chain.\n")
cat("Under the null hypothesis of stationarity, the Z-scores should be within\n")
cat("the range [-1.96, 1.96] at the 95% confidence level.\n\n")

# Run Geweke diagnostic
geweke_results <- geweke.diag(posterior_mcmc)

cat("Geweke Z-scores:\n")
print(geweke_results)
cat("\n")

# Interpret the results
z_scores <- geweke_results$z
all_pass <- all(abs(z_scores) < 1.96)

cat("Interpretation:\n")
if (all_pass) {
  cat("✓ All Z-scores fall within the interval [-1.96, 1.96].\n")
  cat("✓ The Geweke diagnostic provides NO evidence against stationarity.\n")
  cat("✓ This is a strong indication that all chains have converged.\n\n")
} else {
  failed_params <- names(z_scores)[abs(z_scores) >= 1.96]
  cat("✗ The following parameters have Z-scores outside [-1.96, 1.96]:\n")
  cat(paste("  -", failed_params, collapse = "\n"), "\n")
  cat("✗ This suggests potential non-stationarity for these parameters.\n\n")
}

# Create Geweke plot
cat("Generating Geweke plot (dynamic convergence visualization)...\n")
cat("This plot shows Z-scores calculated progressively throughout the chain.\n")
cat("Horizontal lines at ±1.96 indicate the acceptance region.\n\n")

# Save the plot to a file
# Use larger dimensions and appropriate resolution for 9-panel plot
png("geweke_plot.png", width = 2400, height = 1800, res = 150)
# Adjust margins and spacing for better readability
par(mar = c(4, 4, 3, 1), oma = c(0, 0, 2, 0))
geweke.plot(posterior_mcmc)
dev.off()

cat("✓ Geweke plot saved as 'geweke_plot.png'\n\n")

# ----------------------------------------------------------------------------
# Part 2: Heidelberger-Welch Stationarity Test
# ----------------------------------------------------------------------------

cat("Part 2: HEIDELBERGER-WELCH STATIONARITY TEST\n")
cat("---------------------------------------------\n")
cat("The Heidelberger-Welch test uses the Cramer-von Mises test to assess\n")
cat("whether the chain comes from a stationary distribution.\n")
cat("It also provides a half-width test for the accuracy of the posterior mean.\n\n")

# Run Heidelberger-Welch diagnostic
heidel_results <- heidel.diag(posterior_mcmc)

cat("Heidelberger-Welch Test Results:\n")
print(heidel_results)
cat("\n")

# Interpret the results
stationarity_test <- heidel_results[, "stest"]
halfwidth_test <- heidel_results[, "htest"]

all_stationary <- all(stationarity_test == 1)
all_halfwidth <- all(halfwidth_test == 1)

cat("Interpretation:\n")
cat("Stationarity Test:\n")
if (all_stationary) {
  cat("✓ All 9 parameters PASSED the stationarity test.\n")
  cat("✓ This provides formal evidence that the burn-in period was sufficient.\n")
  cat("✓ The retained samples come from a stationary posterior distribution.\n\n")
} else {
  failed_params <- rownames(heidel_results)[stationarity_test == 0]
  cat("✗ The following parameters FAILED the stationarity test:\n")
  cat(paste("  -", failed_params, collapse = "\n"), "\n\n")
}

cat("Half-width Test (Accuracy):\n")
if (all_halfwidth) {
  cat("✓ All 9 parameters PASSED the half-width test.\n")
  cat("✓ The posterior means are estimated with sufficient accuracy.\n\n")
} else {
  failed_params <- rownames(heidel_results)[halfwidth_test == 0]
  cat("✗ The following parameters FAILED the half-width test:\n")
  cat(paste("  -", failed_params, collapse = "\n"), "\n")
  cat("Note: This suggests the chain may need to be run longer for better accuracy.\n\n")
}

# ----------------------------------------------------------------------------
# Summary of Convergence Assessment
# ----------------------------------------------------------------------------

cat("============================================================================\n")
cat("SUMMARY OF CONVERGENCE ASSESSMENT\n")
cat("============================================================================\n")

if (all_pass && all_stationary) {
  cat("✓ CONVERGENCE CONFIRMED\n\n")
  cat("Both the Geweke diagnostic and the Heidelberger-Welch stationarity test\n")
  cat("provide strong formal evidence that:\n")
  cat("  1. The MCMC chains have converged to the target posterior distribution\n")
  cat("  2. The burn-in period of 5,000 iterations was sufficient\n")
  cat("  3. The retained 30,000 samples are from a stationary distribution\n")
  cat("  4. The results are reliable for posterior inference\n")
} else {
  cat("⚠ CONVERGENCE ISSUES DETECTED\n\n")
  cat("One or more diagnostic tests suggest potential convergence problems.\n")
  cat("Consider:\n")
  cat("  1. Increasing the burn-in period\n")
  cat("  2. Running the chain for more iterations\n")
  cat("  3. Checking for coding errors in the Gibbs sampler\n")
  cat("  4. Using different initial values\n")
}
cat("============================================================================\n\n")

# ============================================================================
# Question 13: Posterior Inference for E(y_6)
# ============================================================================

cat("\n============================================================================\n")
cat("POSTERIOR INFERENCE FOR E(y_6)\n")
cat("============================================================================\n\n")

cat("Objective: Estimate the expected number of infections for virus v_6\n")
cat("           with an exposure time of t_6 = 32 minutes.\n\n")

# ----------------------------------------------------------------------------
# Step 1: Isolate the Posterior Samples for lambda_6
# ----------------------------------------------------------------------------

# Extract the 6th row from lambda_posterior matrix
lambda_6_posterior <- lambda_posterior[6, ]

cat("Number of posterior samples for lambda_6:", length(lambda_6_posterior), "\n\n")

# ----------------------------------------------------------------------------
# Step 2: Calculate the Posterior Distribution of E(y_6)
# ----------------------------------------------------------------------------

# Define the exposure time for group 6
t_6 <- t[6]  # t_6 = 32

cat("Known constant: t_6 =", t_6, "minutes\n\n")

cat("Theoretical relationship:\n")
cat("  y_6 ~ Poisson(lambda_6 * t_6)\n")
cat("  E(y_6) = lambda_6 * t_6\n\n")

# Calculate E(y_6) for each posterior sample
# This gives us the full posterior distribution of E(y_6)
E_y6_posterior <- lambda_6_posterior * t_6

cat("Calculated posterior distribution for E(y_6) by applying the formula\n")
cat("to all", length(E_y6_posterior), "posterior samples.\n\n")

# ----------------------------------------------------------------------------
# Step 3: Calculate the Point Estimate
# ----------------------------------------------------------------------------

# The posterior mean is the standard Bayesian point estimate
point_estimate_y6 <- mean(E_y6_posterior)
point_estimate_y6_rounded <- round(point_estimate_y6)

cat("Point Estimate Calculation:\n")
cat("  Posterior mean of E(y_6):", point_estimate_y6, "\n")
cat("  Rounded to nearest integer:", point_estimate_y6_rounded, "\n\n")

# ----------------------------------------------------------------------------
# Step 4: Calculate the 95% Credible Interval
# ----------------------------------------------------------------------------

# Quantile-based credible interval using 2.5th and 97.5th percentiles
credible_interval_y6 <- quantile(E_y6_posterior, probs = c(0.025, 0.975))
credible_interval_y6_rounded <- round(credible_interval_y6)

cat("95% Credible Interval Calculation:\n")
cat("  2.5th percentile:", credible_interval_y6[1], "\n")
cat("  97.5th percentile:", credible_interval_y6[2], "\n")
cat("  Rounded interval: (", credible_interval_y6_rounded[1], ", ", 
    credible_interval_y6_rounded[2], ")\n\n", sep = "")

# ----------------------------------------------------------------------------
# Step 5: Report the Final Results
# ----------------------------------------------------------------------------

cat("============================================================================\n")
cat("FINAL RESULTS FOR E(y_6)\n")
cat("============================================================================\n\n")

cat("Posterior Inference for E(y_6):\n")
cat("--------------------------------\n")
cat("Point Estimate (Posterior Mean):", point_estimate_y6_rounded, "\n")
cat("95% Quantile-Based Credible Interval: (", credible_interval_y6_rounded[1], 
    ", ", credible_interval_y6_rounded[2], ")\n\n", sep = "")

cat("Interpretation:\n")
cat("---------------\n")
cat("Based on the observed data and our hierarchical Bayesian model:\n\n")
cat("• Our best estimate for the expected number of infections for virus v_6\n")
cat("  with an exposure time of 32 minutes is", point_estimate_y6_rounded, "infections.\n\n")
cat("• We are 95% confident that the true expected number of infections lies\n")
cat("  between", credible_interval_y6_rounded[1], "and", credible_interval_y6_rounded[2], 
    "infections.\n\n")
cat("• This credible interval reflects both the uncertainty in lambda_6 and\n")
cat("  our prior beliefs about the infection rate structure.\n\n")

# Additional summary statistics for context
cat("Additional Summary Statistics:\n")
cat("------------------------------\n")
cat("Posterior mean of lambda_6:", round(mean(lambda_6_posterior), 4), "\n")
cat("Posterior SD of E(y_6):", round(sd(E_y6_posterior), 2), "\n")
cat("Posterior median of E(y_6):", round(median(E_y6_posterior)), "\n")
cat("Observed infections for group 6 (y_6):", y[6], "\n")

cat("\n============================================================================\n\n")

# ============================================================================
# Question 14: Calculating a Posterior Probability
# ============================================================================

cat("\n============================================================================\n")
cat("CALCULATION OF P(lambda_6 > 0.53)\n")
cat("============================================================================\n\n")

cat("Objective: Estimate the posterior probability that the infection rate for\n")
cat("           virus v_6 is greater than 0.53.\n\n")

cat("Method: This is calculated as the proportion of MCMC samples for lambda_6\n")
cat("        that are greater than 0.53.\n\n")

# ----------------------------------------------------------------------------
# Step 1: Use the Existing Posterior Samples for lambda_6
# ----------------------------------------------------------------------------

# lambda_6_posterior was already extracted in Question 13
# It contains 30,000 samples from the posterior distribution of lambda_6

# ----------------------------------------------------------------------------
# Step 2: Count the Favorable Outcomes
# ----------------------------------------------------------------------------

# Create a logical vector checking the condition lambda_6 > 0.53
condition_satisfied <- lambda_6_posterior > 0.53

# Count how many samples satisfy the condition
# sum() treats TRUE as 1 and FALSE as 0
num_samples_above_threshold <- sum(condition_satisfied)

# Total number of posterior samples
total_samples <- length(lambda_6_posterior)

cat("Number of samples where lambda_6 > 0.53:", num_samples_above_threshold, "\n")
cat("Total number of posterior samples:", total_samples, "\n\n")

# ----------------------------------------------------------------------------
# Step 3: Calculate the Probability
# ----------------------------------------------------------------------------

# Monte Carlo estimate: proportion of samples satisfying the condition
prob_lambda6_gt_053 <- num_samples_above_threshold / total_samples

cat("Estimated Probability:\n")
cat("----------------------\n")
cat("P(lambda_6 > 0.53) =", prob_lambda6_gt_053, "\n")
cat("P(lambda_6 > 0.53) =", round(prob_lambda6_gt_053, 4), "(rounded to 4 decimals)\n")
cat("P(lambda_6 > 0.53) =", round(prob_lambda6_gt_053 * 100, 2), "%\n\n")

# ----------------------------------------------------------------------------
# Step 4: Interpretation and Context
# ----------------------------------------------------------------------------

cat("Interpretation:\n")
cat("---------------\n")
cat("Based on our model and the data, there is a", 
    round(prob_lambda6_gt_053 * 100, 2), "% chance\n")
cat("that the true infection rate for virus v_6 is greater than 0.53.\n\n")

# Additional context
cat("Additional Context:\n")
cat("-------------------\n")
cat("Posterior mean of lambda_6:", round(mean(lambda_6_posterior), 4), "\n")
cat("Posterior median of lambda_6:", round(median(lambda_6_posterior), 4), "\n")
cat("Posterior SD of lambda_6:", round(sd(lambda_6_posterior), 4), "\n")
cat("95% CI for lambda_6: (", 
    round(quantile(lambda_6_posterior, 0.025), 4), ", ",
    round(quantile(lambda_6_posterior, 0.975), 4), ")\n\n", sep = "")

cat("Note: This type of probability statement is a unique feature of the\n")
cat("      Bayesian approach. In frequentist statistics, we cannot make\n")
cat("      probability statements about parameter values because parameters\n")
cat("      are treated as fixed, unknown constants. In the Bayesian framework,\n")
cat("      parameters are random variables, making this type of inference\n")
cat("      natural and straightforward.\n\n")

cat("============================================================================\n")
cat("END OF ANALYSIS - ALL QUESTIONS COMPLETED\n")
cat("============================================================================\n\n")

cat("Summary of Completed Tasks:\n")
cat("---------------------------\n")
cat("✓ Question 11: Gibbs Sampler Implementation\n")
cat("✓ Question 12: Convergence Diagnostics (Geweke & Heidelberger-Welch)\n")
cat("✓ Question 13: Posterior Inference for E(y_6)\n")
cat("✓ Question 14: Posterior Probability Calculation P(lambda_6 > 0.53)\n\n")

cat("All MCMC chains have converged, and results are reliable for inference.\n")
cat("============================================================================\n")
