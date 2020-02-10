data {
  real weight_total;
  int num_put;  
}

parameters {
  real<lower=0, upper=1> p;
}

model {
  vector[2] theta = [p, 1-p]';
  num_put ~ categorical(theta);
}
