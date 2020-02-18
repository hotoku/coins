library(BRugs)
library(MASS)

model_file <- "model4.txt"
p <- rep(1/999, 999)
n_chain = 1
n_iter = 1e4
burnin = 0.1

modelCheck(model_file)
data_file <- bugsData(list(p=p, digits = 10))
modelData(data_file)
modelCompile(n_chain)
modelGenInits()
modelUpdate(n_iter * burnin)
samplesSet(c("change", "n_500"))
modelUpdate(n_iter)

change <- samplesSample("change")
n_500 <- samplesSample("n_500")

truehist(n_500)

