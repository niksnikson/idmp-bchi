library(tidyverse)
library(olsrr)
library(purrr)
library(modelr)
library(e1071)
library(caTools) 
library(randomForest)
library(caret) 
library(data.table)
library(mlr)
library(xgboost)
# load data
data <- read.csv("../Data/heart.csv")

data <- data.frame(data, check.names=FALSE, stringsAsFactors=FALSE)

head(data)

#format col name
names(data)[1] <- 'age'

#set seed
set.seed(1)
# change categorical var to character

#data$target[data$target == '0'] <- 'No'

#data$target[data$target == '1'] <- 'Yes'

#data$sex[data$sex == '1'] <- 'M'

#data$sex[data$sex == '0'] <- 'F'

data$target <- as.character(data$target)

data$sex <- as.character(data$sex)

data$fbs <- as.character(data$fbs)

data$cp <- as.character(data$cp)

data$restecg <- as.character(data$restecg)

data$exang <- as.character(data$exang)

data$slope <- as.character(data$slope)

data$ca <- as.character(data$ca)

data$thal <- as.character(data$thal)



# model selection

model <- lm(target ~ age + sex + cp + trestbps + chol + fbs + restecg 
            + thalach + exang + oldpeak + slope + ca + thal, data=data)

k <- ols_step_best_subset(model)
plot(k)

model1 <- lm(target ~ age + cp + trestbps + chol + fbs + restecg 
            + thalach + exang + oldpeak + slope + ca + thal, data=data)

k1 <- ols_step_best_subset(model1)
k1
plot(k1)

# 5-cross validation
data_cv <- crossv_kfold(data, 5)

# fit model

data_cv <- data_cv %>%
  mutate(fit = map(train, lm(target ~ cp + slope + ca + thal, data = .)))

data_cv
# plot
ggplot(data, aes(x = target, y = age)) + geom_boxplot()

ggplot(data, aes(x = target, y = trestbps)) + geom_boxplot()

ggplot(data, aes(x = target, y = chol)) + geom_boxplot()

ggplot(data, aes(x = target, y = thalach)) + geom_boxplot()

ggplot(data, aes(x = target, y = oldpeak)) + geom_boxplot()

ggplot(data, aes(x = target, fill = fbs)) + geom_bar()

ggplot(data, aes(x = target, fill = cp)) + geom_bar()

ggplot(data, aes(x = target, fill = restecg)) + geom_bar()

ggplot(data, aes(x = target, fill = exang)) + geom_bar()

ggplot(data, aes(x = target, fill = slope)) + geom_bar()

ggplot(data, aes(x = target, fill = ca)) + geom_bar()

ggplot(data, aes(x = target, fill = thal)) + geom_bar()

ggplot(data, aes(x = target, fill = sex)) + geom_bar()

## SVM

#subset
data_s = select(data,c('cp', 'thalach', 'exang', 'oldpeak','target'))

#factor
#data_s$cp = factor(data_s$cp, levels = c(0, 1, 2, 3)) 

#data_s$exang = factor(data_s$exang, levels = c(0, 1)) 

data_s$target = factor(data_s$target, levels = c(0, 1))


# Splitting the dataset into the Training set and Test set 
split = sample.split(data_s$target, SplitRatio = 0.8) 

training_set = subset(data_s, split == TRUE) 
test_set = subset(data_s, split == FALSE) 


# split train test
#data_s <- resample_partition(data_s, c(train = 0.8, test = 0.2))

#data_s$train <- as_tibble(data_s$train)

#data_s$test <- as_tibble(data_s$test)

# Feature Scaling 
training_set[-5] = scale(training_set[-5]) 
test_set[-5] = scale(test_set[-5]) 

# Fitting SVM to the Training set 
classifier = svm(formula = target ~ ., data = training_set, 
                 type = 'C-classification', kernel = 'linear') 
classifier

y_pred = predict(classifier, newdata = test_set[-5]) 

y_pred

# Making the Confusion Matrix 
cm = table(test_set[, 5], y_pred) 
cm


# Random forest

# transform data
data <- transform(
  data,
  age=as.integer(age),
  sex=as.factor(sex),
  cp=as.factor(cp),
  trestbps=as.integer(trestbps),
  chol=as.integer(chol),
  fbs=as.factor(fbs),
  restecg=as.factor(restecg),
  thalach=as.integer(thalach),
  exang=as.factor(exang),
  oldpeak=as.numeric(oldpeak),
  slope=as.factor(slope),
  ca=as.factor(ca),
  thal=as.factor(thal),
  target=as.factor(target)
)

# check class
sapply(data, class)

#summary
summary(data)

#check null
data[ data == "?"] <- NA
colSums(is.na(data))

# split
sample = sample.split(data$target, SplitRatio = .8)
train = subset(data, sample == TRUE)
test  = subset(data, sample == FALSE)
dim(train)
dim(test)

# initialise randomforest
rf <- randomForest(
  target ~ .,
  data=train
)

#predict
pred = predict(rf, newdata=test[-14])

#confusion matrix
cm = table(test[,14], pred)
cm

confusionMatrix(cm)

classifier = randomForest(x = train[-14],
                          y = train$target,
                          ntree = 500, random_state = 0)
plot(classifier)

#------------------------------------------------------------------------#
# randonm forest using 4 vars
data_s = select(data,c('cp', 'thalach', 'exang', 'oldpeak','target'))

# split
sample = sample.split(data_s$target, SplitRatio = .8)
train1 = subset(data_s, sample == TRUE)
test1  = subset(data_s, sample == FALSE)
dim(train1)
dim(test1)

# initialise randomforest
rf1 <- randomForest(
  target ~ .,
  data=train1
)

#predict
pred1 = predict(rf1, newdata=test1[-5])

#confusion matrix
cm1 = table(test1[,5], pred1)
cm1

confusionMatrix(cm1)

###

classifier1 = randomForest(x = train1[-5],
                          y = train1$target,
                          ntree = 500, random_state = 0)
plot(classifier1)


#----------------------------------------------------#

#xgboost
split = sample.split(data$target, SplitRatio = 0.8) 

train = subset(data, split == TRUE) 
test = subset(data, split == FALSE) 

#convert data frame to data table
setDT(train) 
setDT(test)

#using one hot encoding 
labels <- train$target 
ts_label <- test$target
new_tr <- model.matrix(~.+0,data = train[,-c("target"),with=F]) 
new_ts <- model.matrix(~.+0,data = test[,-c("target"),with=F])

#convert factor to numeric 
labels <- as.numeric(labels)+2
ts_label <- as.numeric(ts_label)+2

#preparing matrix 
dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

#default parameters
params <- list(booster = "gbtree", objective = "binary:logistic", 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, 
                 print_every_n = 10, early_stop_round = 20, maximize = F)

min(xgbcv$test.error.mean)


xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 79, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, early_stop_round = 10, 
                   maximize = F , eval_metric = "error")

#model prediction
xgbpred <- predict (xgb1,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)

#confusion matrix
confusionMatrix (xgbpred, ts_label)

xgbpred <- as.factor(xgbpred)
ts_label <- as.factor(ts_label)

#view variable importance plot
mat <- xgb.importance (feature_names = colnames(new_tr),model = xgb1)
xgb.plot.importance (importance_matrix = mat[1:13]) 


#------------------------------------------------------------------#
#Feature selection using random forest

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(data[,1:13], data[,14], sizes=c(1:13), rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))
