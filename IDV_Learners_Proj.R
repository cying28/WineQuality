##Capstone: IDV Learners - Wine Quality Prediction

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(hrbrthemes)) install.packages("hrbrthemes")
if(!require(corrplot)) install.packages("corrplot")
if(!require(factoextra)) install.packages("factoextra")
if(!require(class)) install.packages("class")
if(!require(e1071)) install.packages("e1071")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")

###########################
#1. Data Preprocessing (Load, Clean, Partition)
#########################
# Load Red Wine Data 
red_wine <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", red_wine)
r_wine <- read.csv(file=red_wine, sep=";")
head(r_wine)
dim(r_wine) #1599, 12

# Load White Wine Data 
white_wine <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", white_wine)
w_wine <- read.csv(file=white_wine, sep=";")
head(w_wine)
dim(w_wine) #4898, 12

# Selected white wine data as the dataset for this project
# Total # of white wine data > # of red wine data

# Data Cleaning
sapply(w_wine, function(x) sum(is.na(x))) #No NA values for the combined dataset

# Data Partition (10% Testing)
set.seed(1, sample.kind="Rounding") #Version used: R 3.6.2
test_index <- createDataPartition(y = w_wine$quality, times = 1, p = 0.1, list = FALSE)
train <- w_wine[-test_index,]
test <- w_wine[test_index,]

###########################
#2. Data Exploration and Visualization
#########################
# Check testing percentage is correct (approx = 10%)
dim(test)[1]/(dim(train)[1]+dim(test)[1])

# Overview/Summary of Data
head(train)
summary(train)

# Histogram Overview of Each Variable
par(mfrow=c(2,3))
for(i in 1:6) {
  hist(train[,i], main=names(train)[i], xlab="Values", col="orange")
}
for(i in 7:12) {
  hist(train[,i], main=names(train)[i], xlab="Values", col="orange")
}

# Correlation Summary
# Try to find out possible models for this dataset
cor_val<-cor(train) #No direct correlation between other factors and quality but alcohol has highest correlation of 0.4383
cor_val
# Correlation Plot
corrplot(round(cor_val,4), type="upper", order="hclust", tl.col="black", tl.srt =45)
#Find the darkest color dots and graph

# Find variables of highest correlation and graph them 
# Residual sugar & density 0.8396
ggpar(mar=c(1,3))
p1 <- ggplot(train, aes(x=residual.sugar, y=density)) +
  geom_point() +
  geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE) 
p1
# Alcohol & density -0.7780
p2 <- ggplot(train, aes(x=alcohol, y=density)) +
  geom_point() +
  geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE) 
p2
# Free sulfur dioxide & total.sulfur.dioxide 0.6162
p3 <- ggplot(train, aes(x=free.sulfur.dioxide, y=total.sulfur.dioxide)) +
  geom_point() +
  geom_smooth(method=lm , color="red", fill="#69b3a2", se=TRUE) +
p3

# Since there are quite a few variables involved, decide whether PCA is needed
# PCA & Visualization
pca<- prcomp(train, center=TRUE, scale.=TRUE)
summary(pca) #Summary PCA
fviz_eig(pca) #Visualized PCA
# PCA is disgarded since contribution of each PC is not a lot


#####################################################

#Loss Function (Evaluation Formula)
RMSE <- function(tru_ratings, pred_ratings){
  sqrt(mean((tru_ratings - pred_ratings)^2))
}

# Model 1: Naive Model (RMSE=0.90455)
#Starting RMSE 
mu_hat <- mean(train$quality)
RMSE_1<-RMSE(test$quality, mu_hat) 
rmse_table<- data.table(Method="Naive", Type="Mean", RMSE=RMSE_1)
rmse_table %>%knitr::kable()

# Model 2: Optimized Knn (RMSE=2.1806196)
set.seed(1)
train_knn<- train(quality ~ ., method = "knn", 
                  data = train,
                  tuneGrid = data.frame(k = seq(3, 51, 2)))
train_knn #use k=49
X_tr_knn=train[,1:11]
Y_tr_knn=train$quality
X_ts_knn=test[,1:11]
Y_ts_knn=test$quality
ts_knn=test[,1:11]
mod_2 <- knn(train=scale(X_tr_knn), test=scale(X_ts_knn), cl=Y_tr_knn, k=c(train_knn$bestTune), prob=TRUE)
RMSE_2 <- RMSE(test$quality, c(mod_2)) #2.154
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="Knn",
                                   Type="Classification",
                                   RMSE = RMSE_2))
rmse_table %>%knitr::kable()


# Model 3: Regression Tree (RMSE=0.7525170)
par(mfrow=c(1,1))
fit_a<- rpart(quality ~ ., data = train, 
              method="anova")
rpart.plot(fit_a)
predict_a <-predict(fit_a, newdata=test)

fit_b<- rpart(quality ~ ., data = train, 
              method="anova",
              control=list(cp=0, xval=10)) #10-fold cross validation
predict_b <-predict(fit_b, newdata=test)
plotcp(fit_b)
abline(v=7, lty="dashed",col = "red")
RMSE_3 <- RMSE(test$quality, predict_b)
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="Regression Tree",
                                   Type="Classification",
                                   RMSE = RMSE_3))
rmse_table %>%knitr::kable()


# Model 4: Linear Regression (Alcohol) (RMSE=0.8244660)
fit <- lm(quality~alcohol, data=train)
mod_4<- fit$coef[1]+fit$coef[2]*test$alcohol
RMSE_4<- RMSE(test$quality, mod_4) #0.82447
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="LM_Alcohol",
                                   Type="Regression",
                                   RMSE = RMSE_4))
rmse_table %>%knitr::kable()

# Model 5: Linear Regression (Alcohol+Density) (RMSE=0.8247146)
fit_2 <- lm(quality~alcohol+density, data=train)
mod_5 <-fit_2$coef[1]+fit_2$coef[2]*test$alcohol+fit_2$coef[3]*test$density
RMSE_5 <- RMSE(test$quality, mod_5) #0.82471
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="LM_Alcohol+Density",
                                   Type="Regression",
                                   RMSE = RMSE_5))
rmse_table %>%knitr::kable()

# Model 6: SVR (RMSE=0.7042508)
mod_6 <- svm(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol, data=train, cost=5, epsilon=0.1) 
predict_6<- predict(mod_6, newdata=test)
RMSE_6 <- RMSE(test$quality, predict_6) #0.70425
rmse_table <- bind_rows(rmse_table, 
                        data_frame(Method="SVR",
                                   Type="Regression",
                                   RMSE = RMSE_6))
rmse_table %>%knitr::kable()


