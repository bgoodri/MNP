data {
  int<lower=3> J;                      // number of choices
  int<lower=1> N;                      // number of observations
  int<lower=0> K;                      // number of unit-specific predictors
  int<lower=0> Q;                      // number of choice-specific coefficients
  
  matrix[N,K]   X;                     // predictor matrix for unit-specific variables
  matrix[N,J-1] Z[Q];                  // predictor matrices for choice-specific variables, excluding Jth choice
  /* Note that the N by J matrices must have a mean of zero for each observation but we only use J - 1 columns */
  
  int<lower=1,upper=J> y[N];           // observed choice outcomes
  real<lower=0> eta;                   // shape parameter for LKJ distribution
}
parameters {
  vector<upper=0>[J-1] utility_gap[N]; // utility differences for non-best choices relative to best choice
  cholesky_factor_corr[J-1] L;         // Cholesky factor of error correlation matrix for first J - 1 choices
  simplex[J-1] scale;                  // parameters which enforce that trace(Sigma) = J, see below

  /* all the above parameters are nuisances; everything below (including beta_J and Sigma) is interesting */
  real gamma[Q];                       // coefficients for choice-specific variables
  real beta[J-1,K];                    // coefficients for unit-specific variables on first J - 1 choices  
}
model {
  matrix[N,J-1] U;                     // utility for the first J - 1 choices
  matrix[N,J-1] mu;                    // linear predictor of U
  vector[J-1] sds;                     // provisional standard deviations of the first J - 1 errors
  
  /* construct utility such that it sums to zero for each unit (Jacobian determinant is constant / ignored) */
  for(i in 1:N) {
    real utility_best;
    utility_best <- -sum(utility_gap[i]) / J;
    for(j in 1:(y[i] - 1))   U[i,j]    <- utility_best + utility_gap[i,j];
    if(y[i] < J)             U[i,y[i]] <- utility_best;
    for(j in (y[i]+1):(J-1)) U[i,j]    <- utility_best + utility_gap[i,j-1];
  }
  
  /* deal with error standard deviations enforcing the implicit constraint that trace(Sigma) = J, see below */
  for(j in 1:(J-1)) sds[j] <- sqrt(scale[j]);
  {
    matrix[J-1,J-1] cov_L;
    real ss;
    cov_L <- diag_pre_multiply(sds, L);
    ss <- 1;             // sum-of-squares of cov_L currently
    for(j in 1:(J-1)) {  // increment ss with sum-of-squares for implicit Jth row of Cholesky factor
      ss <- ss + square(sum(sub_col(cov_L, j, j, J - 1 - j)));
    }
    for(j in 1:(J-1)) sds[j] <- 1.0 / (sds[j] * sqrt(J / ss)); // rescaled sds, now in reciprocal form
  }
  
  /* set up the linear predictors */
  if(Q > 0) { // case where there are some choice-specific predictors
    if(K > 0) mu <- X * to_matrix(beta)';
    else mu <- rep_matrix(0.0, N, J - 1);
    for(q in 1:Q) mu <- mu + Z[q] * gamma[q];
  }
  else mu <- X * to_matrix(beta)'; // case where there are no choice-specific predictors

  /* multivariate normal likelihood */
  increment_log_prob(-0.5 * sum(columns_dot_self(mdivide_left_tri_low(L, 
                     diag_pre_multiply(sds, (U - mu)'))))); // kernel
  increment_log_prob(-N * sum(log(diagonal(L)))); // determinant of correlation submatrix
  increment_log_prob( N * sum(log(sds)));         // determinant of diagonal matrix of standard deviations
                                                  // sds are in reciprocal form so add rather than subtract
                                                  
  /* suggested priors */
  if(K > 0) to_array_1d(beta) ~ normal(0,1);
  if(Q > 0) gamma  ~ normal(0,1);
  L ~ lkj_corr_cholesky(eta);
  // implicit; scale ~ uniform on simplex
}
generated quantities {
  vector[K] beta_J;                    // coefficients for the Jth choice implied by the sum-to-zero constraint
  real eff_indep;                      // measure of error uncorrelatedness for first J-1 choices
  matrix[J,J] Sigma;                   // error covariance matrix implied by the sum-to-zero constraint
  
  for(k in 1:K) beta_J[k] <- -sum(col(to_matrix(beta),k));
  eff_indep <- exp(2 * mean(log(diagonal(L))));
  { /* proceed as in the model {} block */
    matrix[J-1,J-1] cov_L;
    row_vector[J-1] constraints;
    vector[J-1] sds;
    real ss;
    for(j in 1:(J-1)) sds[j] <- sqrt(scale[j]);
    cov_L <- diag_pre_multiply(sds, L);
    
    /* jth element of the implicit Jth row of Cholesky factor is the negative sum of the jth column */
    for(j in 1:(J-1)) constraints[j] <- -sum(sub_col(cov_L, j, j, J - 1 - j));
    ss <- 1 + dot_self(constraints);
    Sigma <- multiply_lower_tri_self_transpose(append_row(cov_L, constraints)) * J / ss; // trace(Sigma) = J
  }
}
