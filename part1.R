library("rjags")
library("coda")
library("readr")
library("runjags")

gsps <- readRDS("\\\\Mac/iCloud/Biostatystyka/Hasselt/2.7 Bayesian Inference I & II/Part II/Project/Bayesian/GSPS.RData")

# • id: person - identification number
# • female: female = 1; male = 0
# • year: calendar year of the observation
# • age: age in years
# • hsat: health satisfaction, coded 0 (low) - 10 (high)
# • handdum: handicapped = 1; otherwise = 0
# • handper: degree of handicap in percent (0 - 100)
# • hhninc: household nominal monthly net income in German marks / 1000
# • hhkids: children under age 16 in the household = 1; otherwise = 0
# • educ: years of schooling
# • married: married = 1; otherwise = 0
# • working: employed = 1; otherwise = 0
# • docvis: number of doctor visits in last three months
# • hospvis: number of hospital visits in last calendar year
# • public: insured in public health insurance = 1; otherwise = 0


#########################################################################################################
#                                          PART 1:                                                      #
#                 Bayesian analyses with MCMC inference in NIMBLE or jags                               #
#########################################################################################################

# Model 1: 
#   - logistic mixed effects model to investigate the employment status over time
#   - Use a random intercept to account for the individual heterogeneity. 
#   - No other covariates have to be taken into account in this model. 
#   - Describe the model that you use (likelihood and prior specification). 
#   - Implement the Bayesian model, give appropriate inference summaries and interpret results.

# https://bayesball.github.io/BOOK/bayesian-multiple-regression-and-logistic-models.html#bayesian-logistic-regression

# https://www.di.fc.ul.pt/~jpn/r/bugs/part2.html#glm-with-a-bernoulli-logistic-regression

# Specification of model

cat("model{
  
  # level 1
  for(i in 1:N) {	
    y[i] ~ dbern(p[i])
    logit(p[i]) <- beta[1] + beta[2]*year[i] + b0[id[i]]
    
  }
  
  # level 2
  for(j in 1:Z) {	
    b0[j] ~ dnorm(0,taub0)
  }
  
  # priors (random effects)
  sigmab0  ~ dunif(0,100)
  sigma2b0 <- pow(sigmab0, 2)
  taub0    <- pow(sigma2b0, -1)
  
  # priors (fixed effects)
  for (k in 1:2){
    beta[k]  ~ dnorm(0.0,1.0E-4)
  }
  
}", file="model.txt")



# Prepare data:
y <- as.integer(gsps$working)
id_fac <- factor(gsps$id)
id <- as.integer(id_fac)
Z <- nlevels(id_fac)     
N <- length(y)          
year <- as.numeric(scale(gsps$year, center = TRUE, scale = FALSE))


my.data <- list(N=N, Z=Z, year=year, y=y, id=id)

# Initial parameters (taken 1:1 from the course materials)

my.inits <- list(
  list(beta=c(0,0), sigmab0=1),
  list(beta=c(0,0), sigmab0=10),
  list(beta=c(0,0), sigmab0=5)
)


# Specify parameters to monitor

parameters <- c("beta","sigmab0")

## Running JAGS:

jags <- jags.model(file="model.txt",
                   data = my.data,
                   inits = my.inits,
                   n.chains = 3)

update(jags,1000) # burn-in period
model.sim <- coda.samples(model = jags,
                          variable.names = parameters,
                          n.iter=30000, 
                          thin=3)



# Convert osteo.sim into mcmc.list for processing with CODA package

model.mcmc <- as.mcmc.list(model.sim)

# Produce general summary of obtained MCMC sampling

summary(model.mcmc)

# Trace plots from Gibbs sampler

par(mfrow=c(1,2))
traceplot(model.mcmc)

# BGR diagnostic (target: < 1.1)

gelman.plot(model.mcmc,ask=FALSE)
gelman.diag(model.mcmc)

# Geweke diagnostic

geweke.diag(model.mcmc)
geweke.plot(model.mcmc,ask=FALSE)

# additional?
effectiveSize(model.mcmc)
plot(model.mcmc)

# Inference summary
m <- as.matrix(model.mcmc)

# effect of the year (per id after centering): beta[2] 
beta_year <- m[, "beta[2]"]

c(mean=mean(beta_year),
  quantile(beta_year, c(0.025,0.5,0.975)))

# OR per 1 year
OR <- exp(beta_year)
quantile(OR, c(0.025,0.5,0.975))