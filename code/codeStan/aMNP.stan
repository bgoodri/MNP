data {
  int<lower=3> J;                      // number of choices
  int<lower=1> N;                      // number of observations
  int<lower=0> K;                      // number of unit-specific predictors
  int<lower=0> Q;                      // number of choice-specific coefficients
  int<lower=0> P;                      // number of choice-specific coefficients on "fly-away" choice

  row_vector[K] X[N];                  // predictor matrix for unit-specific variables
  row_vector[P] Z[N];                  // predictor matrix for unit-specific variables on "fly-away" choice

  int<lower=1,upper=J> y[N];           // observed choice outcomes
  real<lower=0> eta;                   // shape parameter for LKJ distribution
}
parameters {
  vector[N] utility_best;              // utility of the best alternative
  vector<upper=0>[J-1] utility_gap[N]; // utility differences for non-best alternatives
  // all the above parameters are nuisances; everything below is somewhat interesting  
  matrix[K,J-1] beta;                  // coefficients relative on non "fly-away" choices
  vector[P] lambda;                    // coefficients for the "fly-away choice"
  cholesky_factor_corr[J-1] L;         // Cholesky factor of error correlation matrix
  vector<lower=0>[J-2] free_sds;       // free error standard deviations
}
model {
  vector[J] U[N];      // utility
  row_vector[J] mu[N]; // linear predictors
  matrix[J,J] full_L;  // full Cholesky factor of correlation matrix
  vector[J] sds;       // standard deviations
  
  /* set up standard deviations with identification restrictions */
  sds[1] <- 1.0;      
  sds[2] <- 1.0;
  for(j in 3:J) sds[j] <- free_sds[j-2];
  
  /* set up Cholesky factor so that the first column has all zeros except for the first element */
  full_L <- append_col(rep_vector(0.0, J),
            append_row(rep_row_vector(0.0, J - 1), L));
  full_L[1,1] <- 1;

  /* construct utility for each individual */
  for(i in 1:N) {
    for(j in 1:(y[i] - 1)) U[i][j] <- utility_best[i] + utility_gap[i][j];
    U[i][y[i]] <- utility_best[i];
    for(j in (y[i] + 1):J) U[i][j] <- utility_best[i] + utility_gap[i][j-1];
    mu[i] <- append_col(Z[i] * P, X[i] * beta);
  }
  
  /* likelihood; you can ignore warnings about Jacobians because U just involves + and - */
  U ~ multi_normal_cholesky(mu, diag_pre_multiply(sds, full_L));

  /* suggested priors */
  to_vector(beta) ~ normal(0,1);
  L ~ lkj_corr_cholesky(eta);
  free_sds ~ exponential(1);
}

