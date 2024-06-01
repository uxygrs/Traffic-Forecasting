# PIPING AND UTILITY
library(dplyr)

# WORKING WITH TIME DATA-TYPE
library(lubridate) 
library(xts)

# PLOTTING TIME SERIES
library(timetk)
library(ggplot2)
library(gghighlight)

# MOVING AVERAGE
library(zoo)

# FORECASTING
library(forecast)
library(tseries)

# TESTING COEFFICIENTS
library(lmtest)

# XGBOOST
library(xgboost)
library(caret)

# Reading traffic flow data and showing 'head' of the data
traffic <- read.csv("./my_traffic.csv")
head(traffic)

# PREPROCESSING
# Removing ID because it's irrelevant
traffic <- traffic[, -1]
traffic$DateTime <- as_datetime(traffic$DateTime)
traffic$Junction <- as.factor(traffic$Junction)

# EXPLORATORY DATA ANALYSIS
# Plotting 4 plots
traffic %>%
  group_by(Junction) %>%
  plot_time_series(DateTime, Vehicles,      
                   .color_var = year(DateTime),
                   # Facet Formatting
                   .facet_ncol = 2, 
                   .facet_scales = "fixed",
                   .interactive = F,
                   .smooth = F)

# Box Plot for each junction
ggplot(traffic, aes(x = Vehicles)) + 
  geom_boxplot()+
  facet_grid(. ~ Junction)+
  theme_minimal()+ coord_flip()

# Plotting hostogram for each junction
ggplot(traffic, aes(x = Vehicles)) + 
  geom_histogram()+
  facet_grid(. ~ Junction)+
  theme_minimal()

# Smoothing of data to see a clear trend for each junction
smoothing <- function(x) {
  dt <- smooth_vec(x, span = 0.75, degree = 2)
  dt <- data.frame(dt)
  colnames(dt)<-"Smooth"
  return(dt)
}

Junc1 <- traffic %>% filter(Junction == 1) %>% select(DateTime,Vehicles)
Junc2 <- traffic %>% filter(Junction == 2) %>% select(DateTime,Vehicles)
Junc3 <- traffic %>% filter(Junction == 3) %>% select(DateTime,Vehicles)
Junc4 <- traffic %>% filter(Junction == 4) %>% select(DateTime,Vehicles)
Junc5 <- traffic %>% filter(Junction == 5) %>% select(DateTime,Vehicles)
Junc6 <- traffic %>% filter(Junction == 6) %>% select(DateTime,Vehicles)
Junc7 <- traffic %>% filter(Junction == 7) %>% select(DateTime,Vehicles)
Junc8 <- traffic %>% filter(Junction == 8) %>% select(DateTime,Vehicles)
Junc9 <- traffic %>% filter(Junction == 9) %>% select(DateTime,Vehicles)
Junc10 <- traffic %>% filter(Junction == 10) %>% select(DateTime,Vehicles)
Junc11 <- traffic %>% filter(Junction == 11) %>% select(DateTime,Vehicles)

Junc1S <- smoothing(Junc1$Vehicles)
Junc2S <- smoothing(Junc2$Vehicles)
Junc3S <- smoothing(Junc3$Vehicles)
Junc4S <- smoothing(Junc4$Vehicles)
Junc5S <- smoothing(Junc5$Vehicles)
Junc6S <- smoothing(Junc6$Vehicles)
Junc7S <- smoothing(Junc7$Vehicles)
Junc8S <- smoothing(Junc8$Vehicles)
Junc9S <- smoothing(Junc9$Vehicles)
Junc10S <- smoothing(Junc10$Vehicles)
Junc11S <- smoothing(Junc11$Vehicles)

traffic.S <- rbind(Junc1S,Junc2S,Junc3S,Junc4S,Junc5S,Junc6S,Junc7S,Junc8S,Junc9S,Junc10S,Junc11S)
traffic.S <- cbind(traffic,traffic.S)

ggplot(traffic.S, aes(x = DateTime, y = Smooth)) + 
  geom_line(aes(color = Junction), linewidth = 1) +
  scale_color_manual(values = c("red", "blue","green","purple", "cyan", "brown", "magenta", "black", "pink", "yellow", "gray")) +
  theme_minimal()

Junc1.mod <- Junc1 %>%
  dplyr::mutate(.,hours =  hour(DateTime),
                days =  day(DateTime),
                months = month(DateTime),
                years =  year(DateTime))

#avg line
ggplot(data=Junc1.mod, aes(x = hours, y = Vehicles)) + 
  geom_bar(stat = "identity",show.legend = FALSE) +
  theme_minimal() + 
  gghighlight(hours == '19', use_direct_label = F) +
  geom_hline(yintercept = nrow(dt)/24,color="cyan3")

time_index <- seq(from = as.POSIXct("2000-03-17 07:00:00"), 
                  to = as.POSIXct("2005-10-28 14:00:00"), by = "hour")
Junc1.ts <- ts(Junc1$Vehicles, frequency = 24*30*3 ,start = time_index)

trIndex <- ceiling(0.8 * nrow(Junc1))
train <- subset(Junc1.ts, end = trIndex)
test <- subset(Junc1.ts, start = trIndex+1)

train.dec<-decompose(train)
plot(train.dec)

RMSE<- function(fit,obs){
  err <- sqrt(sum((obs-fit)^2/length(obs)))
  return(err)
}

adf.test(train)

kpss.test(train, null="Trend")

ndiffs(train)

auto.arima(train,trace = T,seasonal = F)

(arima512 <- arima(train, order = c(5,1,2)))

coeftest(arima512)

(arima212 <- arima(train, order = c(2,1,2)))

coeftest(arima212)

Box.test(arima512$residuals,type = "Ljung")

Box.test(arima212$residuals,type = "Ljung") #arima212 might be worse than arima512

train.xgb <- Junc1.mod[1:trIndex,]
test.xgb <- Junc1.mod[-(1:trIndex),]

train_Dmatrix <- train.xgb %>%
  dplyr::select(hours,days,months,years) %>%
  as.matrix() 

test_Dmatrix <- test.xgb %>%
  dplyr::select(hours,days,months,years) %>%
  as.matrix() 

target <- train.xgb$Vehicles

xgb_grid <-expand.grid(
  list(
    nrounds = 100,
    max_depth = 10, # maximum depth of a tree
    colsample_bytree = seq(0.5), # subsample ratio of columns when construction each tree
    eta = 0.1, # learning rate
    gamma = 0, # minimum loss reduction
    min_child_weight = 1,  # minimum sum of instance weight (hessian) needed ina child
    subsample = 1 # subsample ratio of the training instances
  ))

xgb_model <- train(
  train_Dmatrix, target,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  nthread = 1
)

actual <- test

forecas.arima212 <- forecast(arima212, h=length(test)) 
autoplot(forecas.arima212) + autolayer(actual)  + autolayer(forecas.arima212$mean)

forecast.arima212.rmse.train <- sqrt(sum(arima212$residuals^2/length(train)))
forecast.arima212.rmse.test <- RMSE(fit = forecas.arima212$mean, obs = test)

forecas.arima512 <- forecast(arima512, h=length(test)) 
autoplot(forecas.arima512) + autolayer(actual) + autolayer(forecas.arima512$mean)

(forecast.arima512.rmse.train <- sqrt(sum(arima512$residuals^2/length(train))))

(forecast.arima512.rmse.test <- RMSE(fit = forecas.arima512$mean, obs = test))

xgb_pred <- predict(xgb_model, newdata = test_Dmatrix)
xgb_pred.rmse.test <- RMSE(xgb_pred, test.xgb$Vehicles)

pred.xgb.dt <- data.frame(DateTime=test.xgb$DateTime,Vehicles=xgb_pred,label="pred")
train.xgb.dt <- data.frame(DateTime=train.xgb$DateTime,Vehicles=train.xgb$Vehicles,label="train")
test.xgb.dt <- data.frame(DateTime=test.xgb$DateTime,Vehicles=test.xgb$Vehicles,label="actual")

plot.xgb <- rbind(train.xgb.dt,test.xgb.dt,pred.xgb.dt)

ggplot(plot.xgb, aes(x = DateTime, y = Vehicles)) + 
  geom_line(aes(color = label), size = 1) +
  scale_color_manual(values = c("tomato", "turquoise", "gray")) +
  theme_minimal()+
  ylim(-100, 200)

RMSE.DT <- data.frame(Method=c("Arima 212","Arima 512","XGBOOST"),
                      RMSE.TRAIN=c(forecast.arima212.rmse.train,forecast.arima512.rmse.train,xgb_model$results$RMSE),RMSE.TEST=c(forecast.arima212.rmse.test,forecast.arima512.rmse.test,xgb_pred.rmse.test))
RMSE.DT

train.full_Dmatrix <- Junc1.mod %>%
  dplyr::select(hours,days,months,years) %>%
  as.matrix()

xgb_model_pred <- train(
  train.full_Dmatrix, Junc1.mod$Vehicles,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  nthread = 1
)

n.forecast <- 24*30*4
DateTime.na <- seq(ymd_hms("2005-10-28 14:00:00"), by = "hour", length.out = n.forecast)
Vehicles.na <- rep(NA, n.forecast)
DT.na <- data.frame(DateTime=DateTime.na,Vehicles=Vehicles.na)

DT.na.mod <- DT.na %>%
  dplyr::mutate(.,hours =  hour(DateTime),
                days =  day(DateTime),
                months = month(DateTime),
                years =  year(DateTime))

pred.n_Dmatrix <- DT.na.mod %>%
  dplyr::select(hours,days,months,years) %>%
  as.matrix()

xgb.full_pred <- xgb_model_pred %>% predict(pred.n_Dmatrix)

train.full <- data.frame(DateTime=Junc1.mod$DateTime,Vehicles=Junc1.mod$Vehicles,label="train")
pred.n <- data.frame(DateTime=DT.na.mod$DateTime,Vehicles=xgb.full_pred,label="pred")

plot.n <- rbind(train.full,pred.n)

ggplot(plot.n, aes(x = DateTime, y = Vehicles)) + 
  geom_line(aes(color = label), size = 1) +
  scale_color_manual(values = c("turquoise", "gray")) +
  theme_minimal()+
  ylim(-100, 200)

