# Loading necessasry libraries
library(rpart)
library(caret)
library(nnet)
library(kernlab)
library(randomForest)
library(adabag)


# Loading data
load("D:/Study/S.MachineLearning/Project/backpain.RData")
backpain_data <- dat

# Ensuring factors
is.factor(backpain_data$PainDiagnosis)
backpain_data$PainDiagnosis = as.factor(backpain_data$PainDiagnosis)
backpain_data$SurityRating = as.factor(backpain_data$SurityRating)

# Summary statistics
summary(backpain_data)


# Creating matrix of size 100 x 8
res<-matrix(NA,100,8)
res <- as.data.frame(res)
colnames(res) <- c("Logistic", "R.Forest", "SVM", "C. Tree", "Bagging", "Boosting", "Chosen", "Test")


# Setting iteration number
iterlim <- 100
for (iter in 1:iterlim)
{
  # Arround 62 % of the data as training data (since bootstraping)
  # Sample 19% of the data as validation
  # Let the remaining 19% data be test data
  
  cat("Iteration: ", iter)
  
  # Bootstrapping and test, validation, train split
  N <- nrow(backpain_data)
  indtrain <- sample(1:N, replace=TRUE)
  indtrain <- sort(indtrain)
  diff <- setdiff(1:N,indtrain)
  indvalid <- sample(diff,size=0.5*length(diff))
  indvalid <- sort(indvalid)
  indtest <- setdiff(1:N, union(indtrain,indvalid))
  
  ############# LOGISTIC REGRESSION #############
  fit.logistic <- multinom(PainDiagnosis~., data=backpain_data, subset=indtrain)
  pred.logistic <- predict(fit.logistic, newdata=backpain_data, type="class")
  tab.logistic <- table(backpain_data$PainDiagnosis[indvalid],pred.logistic[indvalid])
  acc.logistic <- sum(diag(tab.logistic))/sum(tab.logistic)
  res[iter,1] <- acc.logistic

  
  ################ RANDOM FOREST ################
  fit.random<-randomForest(PainDiagnosis~.,data=backpain_data[indtrain,])
  pred.random<-predict(fit.random,type="class",newdata=backpain_data[indvalid,])
  tab.random<-table(backpain_data[indvalid,]$PainDiagnosis, pred.random)
  acc.random <- sum(diag(tab.random))/sum(tab.random)
  res[iter,2] <- acc.random
  
  ##################### SVM #####################
  fit.svm <- ksvm(PainDiagnosis~., data=backpain_data[indtrain,])
  pred.svm <-  predict(fit.svm, backpain_data[indvalid,], type="response") 
  tab.svm <- table(backpain_data[indvalid,]$PainDiagnosis, pred.svm)
  acc.svm <- sum(diag(tab.svm))/sum(tab.svm)
  res[iter,3] <- acc.svm
  
  ############# CLASSIFICATION TREE ##############
  fit.tree = rpart(PainDiagnosis~., data=backpain_data[indtrain,])
  pred.tree <-  predict(fit.tree, backpain_data[indvalid,], type="class") 
  tab.tree <- table(backpain_data[indvalid,]$PainDiagnosis, pred.tree)
  acc.tree <- sum(diag(tab.tree))/sum(tab.tree)
  res[iter,4] <- acc.tree
  
  
  ################### BAGGING ###################
  fit.bag<-bagging(PainDiagnosis~.,data=backpain_data[indtrain,])
  pred.bag<-predict(fit.bag,type="class",newdata=backpain_data[indvalid,])
  acc.bag<-1-pred.bag$error
  res[iter,5] <- acc.bag
  
  ################### BOOSTING ##################
  fit.boost<-boosting(PainDiagnosis~.,data=backpain_data[indtrain,])
  pred.boost<-predict(fit.boost, type="class",newdata=backpain_data[indvalid,])
  acc.boost <- 1 - pred.boost$error
  res[iter,6] = acc.boost 
  
  
  # Taking best model and saving it to results
  max_method = names(res[which.max(res[iter,1:6])])
  type_s = "class"
  
  if (max_method == colnames(res)[1]){
    best_model = fit.logistic
    name = "Logistic"
  } else if (max_method == colnames(res)[2]){
    best_model = fit.svm
    type_s = "response"
    name = "Random Forest"
  } else if (max_method == colnames(res)[3]){
    best_model = fit.tree
    name = "SVM"
  } else if (max_method == colnames(res)[4]){
    best_model = fit.random
    name = "Classification Tree"
  } else if (max_method == colnames(res)[5]){
    best_model = fit.bag
    name = "Bagging"
  } else{
    best_model = fit.boost
    name = "Boosting"
  }

  # Predicting with best model
  pred.best<-predict(best_model,type=type_s,newdata=backpain_data[indtest,])
  
  if (name == "Boosting" || name == "Bagging" ){
     acc.best <- 1 - pred.best$error
  }
  else{
    tab.best<-table(backpain_data[indtest,]$PainDiagnosis, pred.best)
    acc.best <- sum(diag(tab.best))/sum(tab.best)
  }
  
  # storing results
  res[iter,7] = name
  res[iter,8] = acc.best
  
} 

res$Chosen <- as.factor(res$Chosen)
summary(res)

############## ANALYSING RESULTS ################

reslt <- res
# Plotting the reults obtained
maxbarcol <- 1+(table(reslt$Chosen)==max(table(reslt$Chosen)))
bar=barplot(table(reslt$Chosen), main="Frequency plot (100 Iter.)", 
        xlab = "Algorithms", ylab="No: of ocurrence", 
        col=c("black","red")[maxbarcol], names.arg = "", ylim = c(0, 60))
text(bar[,1], -3.7, srt = 60, adj= 1, xpd = TRUE, 
     labels =c("Bagging", "Boosting", "C. Tree", "Logistic", "R.Forest", "SVM") , cex=0.7)
text(x = bar, y = table(reslt$Chosen), label = table(reslt$Chosen), pos = 3, cex = 0.8, col = "red")


# Test data statistics
tapply(reslt$Test, reslt$Chosen, max) # maxvalues
tapply(reslt$Test, reslt$Chosen, min) # minvalues
tapply(reslt$Test, reslt$Chosen, mode) # meanvalues

# Variable importnace plot
imp=varImp(fit.random)
# Plot
varImpPlot(fit.random, main = "Variable importance")


