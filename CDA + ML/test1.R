library(tidyverse)
library(olsrr)
library(purrr)
library(modelr)
library(e1071)
library(caTools) 
library(randomForest)
library(caret) 
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
rf <- randomForest(
  target ~ .,
  data=train1
)

#predict
pred1 = predict(rf, newdata=test1[-5])

#confusion matrix
cm1 = table(test1[,5], pred1)
cm1

confusionMatrix(cm1)
