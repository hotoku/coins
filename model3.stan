functions {
  row_vector count_coins(int change){
    int n500 = change / 500;
    int c2 = change % n500;
    int n100 = c2 / 100;
    int c3 = c2 % 100;
    int n50 = c3 / 50;
    int c4 = c3 % 50;
    int n10 = c4 / 10;
    int c5 = c4 % 10;
    int n5 = c5 / 5;
    int c6 = c5 % 5;
    int n1 = c6;
    return [n1, n5, n10, n50, n100, n500];
  }
  
  row_vector count_coins_rng(int num_put){
    int changes[num_put];
    vector[999] prob_change = rep_vector(1.0 / 999, num_put);
    matrix[num_put, 6] counts;
    row_vector[num_put] ones = rep_row_vector(1, num_put);
    for(i in 1:num_put){
      changes[i] = categorical_rng(prob_change);
      counts[i,] = count_coins(changes[i]);
    }
    return ones * counts;
  }
  
  real weight(int change){
    row_vector[6] count = count_coins(change);
    vector[6] weight = [1, 3.75, 4.5, 4, 4.8, 7]';
    return count * weight;
  }
}

data {
  real weight_total;
  int Nmin;
  int Nmax;
}

transformed data {
  int numN = Nmax - Nmin + 1;
  int numTerm = (Nmax + Nmin) * numN / 2;
}

parameters {
  real<lower=0> beta;
  simplex[numN] theta;  
}

model {
  real W;
  vector[numTerm] loglikelihoods;
  int pos;
  pos = 1;
  for(N in Nmin:Nmax){
    for(i in 1:N){
      for(Ai in 1:999){
        W = weight(Ai);
        loglikelihoods[pos] = 
        gamma_lpdf(weight_total | beta * W, beta) - N * log(999) + log(theta[i]);
        pos += 1;
      }
    }
  }
  target += log_sum_exp(loglikelihoods);
}

generated quantities {
  int N_ = categorical_rng(theta);
  int N = N_ + Nmin - 1;
}
