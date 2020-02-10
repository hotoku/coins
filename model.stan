functions {
  row_vector count_coins(real change){
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
}

data {
  real weight_total;
  int num_put;
}

transformed data {
  vector[6] weight = [1, 3.75, 4.5, 4, 4.8, 7]';
  vector[999] theta = softmax(rep_vector(1, 999));
}

parameters {
  // real<lower=0> l_put;
  real beta;
  int<lower=1, upper=999> change[num_put];
  real a1;
  real b1;
}

model {
  matrix[num_put, 6] num_coins;
  real weight_exp;

  change ~ uniform()
  for(i in 1:num_put){
    num_coins[i,] = count_coins(floor(change[i]));
  }
  weight_exp = sum(num_coins * weight);
  weight_total ~ gamma(weight_exp*beta, beta);
}

