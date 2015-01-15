data {
  int<lower=3> J;                      # choices
  int<lower=1> N;                      # observations
  int<lower=0> K;                      # unit-specific predictors
  int<lower=0> P;                      # choice-specific predictors
  int<lower=0> Q;                      # number of free choice-specific coefficients
  
  matrix[N,K]  X;                      # predictor matrix for unit-specific variables
  matrix[N,P]  Z;                      # predictor matrix for choice-specific variables for first J - 1 choices
  int<lower=1,upper=J> y[N];           # observed choice outcomes
  real<lower=0> eta;                   # shape parameter, typically 1.0 but possibly greater
}
transformed data {
  matrix[N,P] ZZ;
  int<lower=0> PoverJm1;
  PoverJm1 <- P / (J - 1);
  for(p in 1:P) {                      # center choice-specific variables
    real colmean;
    colmean <- mean(col(Z, p));
    for(i in 1:N) ZZ[i,p] <- Z[i,p] - colmean;
  }
}
parameters {
  vector<upper=0>[J-1] utility_gap[N]; # utility differences for non-best alternatives relative to best alternative
  cholesky_factor_corr[J-1] L;         # Cholesky factor of error correlation matrix for first J - 1 choices
  simplex[J-1] scale;                  # parameters which enforce that the trace of the error covariance matrix is J

  # all the above parameters are nuisances; everything below (including in generated quantities) is interesting
  vector[Q] gamma;                     # coefficients for choice-specific variables
  real beta[J-1,K];                    # coefficients for unit-specific variables on first J - 1 choices  
}
model {
  matrix[N,J-1] U;                     # utility, excluding the Jth category
  matrix[N,J-1] mu;                    # linear predictor of utility, excluding the Jth category
  vector[J-1] sds;                     # provisional standard deviations of the first J - 1 errors
  for(i in 1:N) {                      # construct utility such that it sums to zero for each unit
    real utility_best;
    utility_best <- -sum(utility_gap[i]) / J;
    for(j in 1:(y[i] - 1))   U[i,j]    <- utility_best + utility_gap[i,j];
    if(y[i] < J)             U[i,y[i]] <- utility_best;
    for(j in (y[i]+1):(J-1)) U[i,j]    <- utility_best + utility_gap[i,j-1];
  }
  
  # deal with error standard deviations enforcing the implicit constraint on the trace of Sigma (below)
  for(j in 1:(J-1)) sds[j] <- sqrt(scale[j]);
  {
    matrix[J-1,J-1] constrained_L;
    real ss;
    constrained_L <- diag_pre_multiply(sds, L);
    ss <- 1;
    for(j in 1:(J-1)) {
      ss <- ss + square(sum(sub_col(constrained_L, j, j, J - 1 - j)));
    }
    for(j in 1:(J-1)) sds[j] <- 1.0 / (sds[j] * sqrt(J / ss));
  }
  
  # set up the linear predictors
  if(P > 0) {
    vector[P] gamma_temp;
    int pos;
    pos <- 1;
    for(q in 1:Q) for(p in 1:P) {
      gamma_temp[pos] <- gamma[q];
      pos <- pos + 1;
    }
    if(K > 0) mu <- X * to_matrix(beta)' + rep_matrix(ZZ * gamma_temp, J - 1);
    else mu <- rep_matrix(ZZ * gamma_temp, J - 1);
  }
  else mu <- X * to_matrix(beta)';

  # likelihood
  increment_log_prob(-0.5 * sum(columns_dot_self(mdivide_left_tri_low(L, 
                     diag_pre_multiply(sds, (U - mu)'))))); # kernel
  increment_log_prob(-N * sum(log(diagonal(L)))); # determinant
  increment_log_prob(-N * sum(log(sds)));         # determinant

  /* suggested priors */
  if(K > 0) to_array_1d(beta) ~ normal(0,1);
  if(P > 0) to_vector(gamma)  ~ normal(0,1);
  L ~ lkj_corr_cholesky(eta);
  // implicit; scale ~ uniform on simplex
}
generated quantities {
  vector[K] beta_J;                    # coefficients for the Jth choice implied by the sum-to-zero constraint
  real eff_indep;                      # how correlated are the errors?
  matrix[J,J] Sigma;                   # error covariance matrix implied by the sum-to-zero constraint
  
  for(k in 1:K) beta_J[k] <- -sum(col(to_matrix(beta),k));
  eff_indep <- exp(2 * mean(log(diagonal(L))));
  {
    matrix[J-1,J-1] constrained_L;
    row_vector[J-1] constraints;
    vector[J-1] sds;
    real ss;
    for(j in 1:(J-1)) sds[j] <- sqrt(scale[j]);
    constrained_L <- diag_pre_multiply(sds, L);
    
    for(j in 1:(J-1)) constraints[j] <- -sum(col(constrained_L, j));
    ss <- 1 + dot_self(constraints);
    Sigma <- multiply_lower_tri_self_transpose(append_row(constrained_L, constraints)) * J / ss;
  }
}



