stopifnot(require(MCMCpack))
stopifnot(require(rstan))

data(Nethvote)
Nethvote$class <- as.factor(round(pmax(0, Nethvote$class)))
Nethvote$income <- as.factor(round(pmax(0, pmin(6, Nethvote$income))))
Nethvote$educ <- as.factor(round(Nethvote$educ))
Nethvote$age <- as.factor(Nethvote$age)

X <- model.matrix(~ relig + class + income  + educ + age + urban, data = Nethvote)
Z <- model.matrix(~ (distCDA + distD66 + distPvdA + distVVD) : vote - 1, data = Nethvote)
Z <- matrix(0, nrow(Z), 0)
data_list <- list(J = nlevels(Nethvote$vote), N = nrow(Nethvote), K = ncol(X), Q = ncol(Z),
                  P = ncol(Z), Z = Z, X = X, y = as.integer(Nethvote$vote), eta = 1 + ncol(X) / 2)
fit <- stan("sMNP.stan", data = data_list, init_r = 0.1, chains = 1,
            pars = c("gamma", "beta", "beta_J", "eff_indep", "Sigma"), test_grad = TRUE)
library(parallel)
stan_posterior <- sflist2stanfit(mclapply(1:8, mc.cores = detectCores(), FUN = function(chain) {
  stan(fit = fit, data = data_list, chains = 1, refresh = 10 * (chain == 1), 
       seed = 12345, chain_id = chain, pars = c("gamma", "beta", "beta_J", "eff_indep", "Sigma"))
}))

means <- get_posterior_mean(stan_posterior)[,9]
U_gap <- matrix(means[grepl("^utility_gap", names(means))], ncol = 3, byrow = TRUE)
U_best <- -rowSums(U_gap) / 4
U <- t(sapply(1:nrow(Nethvote), FUN = function(i) {
  mark <- as.integer(Nethvote$vote[i])
  u <- U_best[i] + c(head(U_gap[i,], mark - 1), 0, tail(U_gap[i,], 4 - mark))
  return(u)
}))
plot(1:nrow(U), ylim = range(U), type = "n", xlab = "Person ID", ylab = "Utility")
invisible(sapply(1:ncol(U), FUN  = function(j) points(U[,j], pch = 20, col = j)))

