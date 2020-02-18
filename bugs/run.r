rm(list=ls(all=T))

library(rstan)
library(tidyverse)
library(logging)

iter <- 10000
warmup <- 2000
chains <- 4

stan_file <- "model3.stan"
model_file <- "model.rds"

should_make <- function(src, prd){
  if(prd %>% file.exists() %>% all %>% `!`){
    return(TRUE)
  }
    
  src_newest <- src %>% file.mtime %>% max()   
  prd_oldest <- prd %>% file.mtime %>% min()
  src_newest >= prd_oldest
}

compile <- function(stan_file, model_file){
  
  if(should_make(stan_file, model_file)){
    loginfo("compile")  
    ret <- stan_model(stan_file)
    saveRDS(ret, model_file)
  } else {
    loginfo("not compiled. reading cache")  
    ret <- readRDS(model_file)
  }
  ret
}

sampling <- function(model, data, conf){
  args <- list(
    model=model,
    data=data,
    conf=conf
  )
  cache <- sprintf("sampling_%s.rds", digest::digest(args))
  loginfo(sprintf("sampling: cache=%s", cache))
  if(file.exists(cache)){
    loginfo("not sampled: reading cache")
    ret <- readRDS(cache)
  } else {
    loginfo("sampling")
    ret <- rstan::sampling(
      model, data = data, verbose = TRUE,
      iter = conf$iter, warmup = conf$warmup, chains = conf$chains,
      cores = 4)
    saveRDS(ret, cache)
  }
  ret
}

data <- list(
  weight_total = 10 * 1000,
  Nmin = 250,
  Nmax = 400
)
model <- compile(stan_file, model_file)
fit <- sampling(model, data, list(
  iter=iter, warmup=warmup, chains=chains  
))