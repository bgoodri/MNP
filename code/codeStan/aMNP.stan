data {
  int<lower=3> J;                      // number of choices
  int<lower=1> N;                      // number of observations
  int<lower=0> K;                      // number of unit-specific predictors
  // int<lower=0> Q;                      // number of choice-specific coefficients
  int<lower=0> P;                      // number of choice-specific coefficients on "fly-away" choice

  row_vector[K] X[N];                  // predictor matrix for unit-specific variables
  row_vector[P] Z[N];                  // predictor matrix for unit-specific variables on "fly-away" choice

  int<lower=1,upper=J> y[N];           // observed choice outcomes
  
  matrix[J-1,J-1] edges;
}
parameters {
  vector[N] utility_best;              // utility of the best alternative
  vector<upper=0>[J-1] utility_gap[N]; // utility differences for non-best alternatives
  // all the above parameters are nuisances; everything below is somewhat interesting  
  matrix[K,J-1] beta;                  // coefficients relative on non "fly-away" choices
  matrix[P,1] lambda;                  // coefficients for the "fly-away choice"
  real<lower=0> eta2;
  real<lower=0> rho2;
  real<lower=0> sigma2;
}
model {
  vector[J] U[N];      // utility
  row_vector[J] mu[N]; // linear predictors
  matrix[J,J] Sigma;
  
  for(i in 2:(J-1)) {
    Sigma[1,i] <- 0;
    Sigma[i,1] <- 0;
    for(j in (i+1):J) {
      Sigma[i,j] <- eta2 * exp(-rho2 * edges[i-1,j-1]);
      Sigma[j,i] <- Sigma[i,j];
    }
  }
  Sigma[1,J] <- 0;
  Sigma[J,1] <- 0;
  Sigma[1,1] <- 1;
  Sigma[2,2] <- 1;
  for(j in 3:J) Sigma[j,j] <- eta2 + sigma2;

  /* construct utility for each individual */
  for(i in 1:N) {
    for(j in 1:(y[i] - 1)) U[i][j] <- utility_best[i] + utility_gap[i][j];
    U[i][y[i]] <- utility_best[i];
    for(j in (y[i] + 1):J) U[i][j] <- utility_best[i] + utility_gap[i][j-1];
    mu[i] <- append_col(Z[i] * lambda, X[i] * beta);
  }
  
  /* likelihood; you can ignore warnings about Jacobians because U just involves + and - */
  U ~ multi_normal(mu, Sigma);

  /* suggested priors */
  to_vector(beta) ~ normal(0,1);
  to_vector(lambda) ~ normal(0,1);
  eta2 ~ exponential(1);
  rho2 ~ exponential(1);
  sigma2 ~ exponential(1);
}

