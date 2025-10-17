// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// Helper: safe convergence check (L2 norm)
inline double l2_diff(const arma::vec& a, const arma::vec& b) {
  return arma::norm(a - b, 2);
}

// [[Rcpp::export]]
List cpp_poisson_irls(const arma::mat& X,
                      const arma::vec& y,
                      int maxit = 25,
                      double tol = 1e-8,
                      bool verbose = false) {
  // Poisson with log link (canonical)
  // X: n x p design matrix (include intercept column if desired)
  // y: n vector of nonnegative counts
  const int n = X.n_rows;
  const int p = X.n_cols;
  
  arma::vec beta = arma::zeros<arma::vec>(p);      // initial coefficients
  arma::vec eta(n), mu(n), z(n), w(n);
  
  arma::mat XtWX(p, p);
  arma::vec XtWz(p);
  
  bool converged = false;
  int iter;
  for (iter = 0; iter < maxit; ++iter) {
    // eta = X * beta
    eta = X * beta;
    
    // mu = exp(eta)  (inverse link)
    mu = arma::exp(eta);
    
    // weights and working response for Poisson (log link)
    // For canonical log link: g'(mu) = 1/mu, V(mu) = mu
    // w = 1 / (V(mu) * (g'(mu))^2) = mu
    // z = eta + (y - mu) * g'(mu) = eta + (y - mu)/mu
    w = mu; // n-vector
    // avoid zero weights
    for (int i = 0; i < n; ++i) if (w[i] <= 0) w[i] = 1e-12;
    z = eta + (y - mu) / mu;
    
    // Compute XtWX and XtWz
    XtWX.zeros();
    XtWz.zeros();
    
    // Efficient formation: for each column j, k
    // XtWX = X^T * diag(w) * X
    // XtWz = X^T * (w % z)
    arma::vec wz = w % z; // elementwise
    for (int j = 0; j < p; ++j) {
      for (int k = j; k < p; ++k) {
        double val = arma::dot(X.col(j) % w, X.col(k));
        XtWX(j, k) = val;
        if (k != j) XtWX(k, j) = val;
      }
      XtWz[j] = arma::dot(X.col(j), wz);
    }
    
    // Solve linear system (XtWX) beta_new = XtWz
    arma::vec beta_new;
    bool solved = arma::solve(beta_new, XtWX, XtWz, arma::solve_opts::fast + arma::solve_opts::likely_sympd);
    if (!solved) { // fallback to more robust solve
      beta_new = arma::solve(XtWX + 1e-8 * arma::eye<arma::mat>(p,p), XtWz);
    }
    
    double diff = l2_diff(beta_new, beta);
    beta = beta_new;
    
    if (verbose) Rcpp::Rcout << "iter=" << iter+1 << " diff=" << diff << "\n";
    if (diff < tol) { converged = true; ++iter; break; }
  }
  
  // final outputs
  eta = X * beta;
  mu = arma::exp(eta);
  
  // deviance for Poisson: 2 * sum( y * log(y/mu) - (y - mu) ) with convention y*log(y/mu)=0 if y==0
  double dev = 0.0;
  for (int i = 0; i < n; ++i) {
    if (y[i] == 0) {
      dev += - (y[i] - mu[i]);
    } else {
      dev += 2.0 * ( y[i] * std::log( y[i] / mu[i] ) - (y[i] - mu[i]) )/2.0 * 2.0; // simplified, keep numeric stable
      // simpler stable form:
      dev += 2.0 * ( y[i] * std::log( y[i] / mu[i] ) - (y[i] - mu[i]) );
    }
  }
  
  return List::create(
    Named("coefficients") = wrap(beta),
    Named("fitted.values") = wrap(mu),
    Named("linear.predictors") = wrap(eta),
    Named("deviance") = dev,
    Named("iterations") = iter,
    Named("converged") = converged
  );
}


// [[Rcpp::export]]
List cpp_matched_pair_binomial_irls(const arma::mat& DeltaX,
                                    const arma::vec& ya,
                                    const arma::vec& yb,
                                    int maxit = 25,
                                    double tol = 1e-8,
                                    bool verbose = false) {
  // Fits conditional Binomial model from Poisson splitting:
  // For each pair i: t_i = ya_i + yb_i, success = ya_i
  // p_i = tau(DeltaX_i %*% beta) where tau is logistic
  // This fits a binomial GLM with logit link using IRLS on counts.
  const int n = DeltaX.n_rows;
  const int p = DeltaX.n_cols;
  
  arma::vec t = ya + yb;
  // check non-negative integers
  for (int i = 0; i < n; ++i) if (t[i] <= 0) t[i] = 0; // pairs where t==0 contribute no info
  
  arma::vec beta = arma::zeros<arma::vec>(p);
  arma::vec eta(n), mu(n), z(n), w(n);
  
  arma::mat XtWX(p, p);
  arma::vec XtWz(p);
  
  bool converged = false;
  int iter;
  for (iter = 0; iter < maxit; ++iter) {
    // linear predictor and mean (logit)
    eta = DeltaX * beta;
    // mu = logistic(eta)
    mu = 1.0 / (1.0 + arma::exp(-eta));
    
    // weights: w_i = t_i * mu_i * (1 - mu_i)
    w = t % (mu % (1.0 - mu));
    for (int i = 0; i < n; ++i) if (w[i] <= 0) w[i] = 1e-12;
    
    // working response: z = eta + ( (y/t) - mu ) / (mu*(1-mu))
    arma::vec yprop(n);
    for (int i = 0; i < n; ++i) {
      if (t[i] > 0) yprop[i] = ya[i] / t[i];
      else yprop[i] = mu[i]; // no info
    }
    z = eta + (yprop - mu) / (mu % (1 - mu));
    
    // form XtWX, XtWz
    XtWX.zeros();
    XtWz.zeros();
    arma::vec wz = w % z;
    
    for (int j = 0; j < p; ++j) {
      for (int k = j; k < p; ++k) {
        double val = arma::dot(DeltaX.col(j) % w, DeltaX.col(k));
        XtWX(j, k) = val;
        if (k != j) XtWX(k, j) = val;
      }
      XtWz[j] = arma::dot(DeltaX.col(j), wz);
    }
    
    // solve
    arma::vec beta_new;
    bool solved = arma::solve(beta_new, XtWX, XtWz, arma::solve_opts::fast + arma::solve_opts::likely_sympd);
    if (!solved) {
      beta_new = arma::solve(XtWX + 1e-8 * arma::eye<arma::mat>(p,p), XtWz);
    }
    
    double diff = l2_diff(beta_new, beta);
    beta = beta_new;
    
    if (verbose) Rcpp::Rcout << "iter=" << iter+1 << " diff=" << diff << "\n";
    if (diff < tol) { converged = true; ++iter; break; }
  }
  
  eta = DeltaX * beta;
  mu = 1.0 / (1.0 + arma::exp(-eta));
  
  // binomial deviance: 2 * sum( t_i * ( yprop_i * log(yprop_i/mu_i) + (1-yprop_i)*log((1-yprop_i)/(1-mu_i)) ) )
  double dev = 0.0;
  for (int i = 0; i < n; ++i) {
    double yi = (t[i] > 0) ? (ya[i]/t[i]) : mu[i];
    if (t[i] > 0) {
      if (ya[i] == 0) {
        dev += 2.0 * t[i] * ( (1-yi) * std::log((1-yi)/(1-mu[i])) );
      } else if (ya[i] == t[i]) {
        dev += 2.0 * t[i] * ( yi * std::log( yi / mu[i] ) );
      } else {
        dev += 2.0 * t[i] * ( yi * std::log( yi / mu[i] ) + (1-yi) * std::log((1-yi)/(1-mu[i])) );
      }
    }
  }
  
  return List::create(
    Named("coefficients") = wrap(beta),
    Named("fitted.probabilities") = wrap(mu),
    Named("linear.predictors") = wrap(eta),
    Named("deviance") = dev,
    Named("iterations") = iter,
    Named("converged") = converged
  );
}

