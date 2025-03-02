## Predicting Bank Loan Approval and Disapproval

# importing the two data set into the r-studio IDE
Profile <- read.csv("C:/Users/Downloads/Telegram Desktop/predictive r/archive bank/Data 1 (Demographics).csv")
Profile
View(Profile)
Construct <- read.csv("C:/Users/Downloads/Telegram Desktop/predictive r/archive bank/Data 2 (Constucts).csv")
Construct
View(Construct)

library(tidyverse)
library(rpart)
library(rpart.plot)
library(tidymodels)


# Now merging the two data sets
# Merge data sets on a common column (assuming 'id' is the common column)
Combinded_data <- inner_join(Profile, Construct, by = 'ID')
Combinded_data
View(Combinded_data)

str(Combinded_data)


                            # Data cleaning
# checking for missing values
sum(is.na(Combinded_data))
# checking for duplicated columns and rows
sum(duplicated(Combinded_data))

# Having an overview of the combined data
glimpse(Combinded_data)
str(Combinded_data)
summary(Combinded_data)


                         # Exploratory Data Analysis
                             #Feature Selection
# Create a subset of the initial data, taking the most relevant columns
data <- Combinded_data%>%
  select(Age, Experience,Income,Family,CCAvg,
         Education,Personal.Loan,CreditCard,
         Online,CD.Account)
View(data)


set.seed(100)

# an overview of the data
glimpse(data)



# Factorizing the feature selected columns of the data

data <- data %>%
  mutate(
    Personal.Loan = factor(Personal.Loan, 
                           levels = c("0", "1"),
                           labels = c("Not Approved", "Approved")),
    Age = as.numeric(Age),
    Age = cut(Age, breaks= c(22,30,40,50,60,70),
              labels = c("22-30", "31-40", "41-50", "51-60", "61-70")),
    Income = as.numeric(Income),
    Income = cut(Income, breaks = c(7, 24, 100, 150, 200, 230),
                 labels = c("Poor", "MiddleClass", "UpperMiddleClass", "Rich", "SuperRich")),
    CreditCard = factor(CreditCard, levels = c("0", "1"),
                        labels = c("Not available", "Available")),
    Online = factor(Online, levels = c("0", "1"),
                    labels = c("Not Technically Inclined", "Technically Inclined")),
    Education = factor(Education, levels = c("1","2","3"),
                       labels = c ("Diploma","Bachelors Degree", "Masters Degree")),
    Experience = as.numeric(Experience),
    Experience = cut(
      Experience,
      breaks = c(-1, 5, 10, 15, 20, 30, 35, 45, 50),
      labels = c("-1-5", "6-10", "11-15", "16-20", "21-30", "31-35", "36-45", "46-50")),
    CD.Account = factor(CD.Account, levels= c("0","1"),
                        labels= c("Not a member", "Member")),
    Family = factor(Family, levels = c("1","2","3","4"),
                    labels =c("Small-Size", "Normal-Size", "Okay-Size", "Big-Size")),
    CCAvg = as.numeric(CCAvg),
    CCAvg = cut(CCAvg, breaks = c(0, 2, 3.5, 5, 7, 10),
                labels= c("0.5-2", "2-3.5", "3.5-5", "5-7", "7-10"))
  )|>glimpse()

View(data)


# Visualization box Plot and Correlation 
library(ggplot2)
library(reshape2)
library(corrplot)

                           # some visualizations of the data
# Plot Box plot for Income by Personal Loan with colors
ggplot(data, aes(x = Personal.Loan, y = as.numeric(Income), fill = Personal.Loan)) +
  geom_boxplot() +
  scale_fill_manual(values = c("Not Approved" = "lightblue", "Approved" = "orange")) +
  labs(title = "Box Plot of Income by Personal Loan Status",
       x = "Personal Loan Status",
       y = "Income") +
  theme_minimal()


# EDA: Creating scatter plot matrix to visualize pairwise relationships
library(GGally)

# Beautified pairs plot
ggpairs(data, columns = c("Age", "Experience","Income","Family","CCAvg","Education",
                          "Personal.Loan","CreditCard","Online","CD.Account"), 
        lower = list(continuous = "points"), diag = list(continuous = "density"), 
        upper = list(continuous = "cor", combo = "box"))

pairs(data)


                              # Feature Engineering
                           # Data Partitioning into training and testing
# Nb Mostly training data takes 80% or 75% of your data
#          DATA PREPROCESSING
bank_sec_A <- initial_split(data, prop = 0.75,
                            strata = Personal.Loan)


# creating the testing and training data set
# train data set
bank_train_A <- bank_sec_A%>%
  training()
glimpse(bank_train_A)
nrow(bank_train_A)

# test data set
bank_test_A <- bank_sec_A%>%
  testing()
glimpse(bank_test_A)
nrow(bank_test_A)

                     # now choosing the specified model to predict
                    # First model (Logistics Model)
Logistic_Model <- logistic_reg()%>%
  set_engine('glm')%>%
  set_mode('classification')


# Model Fitting
log_fit <- Logistic_Model%>%
  fit(Personal.Loan ~.,
      data = bank_train_A)


# predict outcome categories (Actual result achieved)
# predicting the test data based on the initial train data partitioned
class_preds <- log_fit %>%
  predict(new_data = bank_test_A,
          type = 'class')
class_preds



# Predicting the estimated probability thresholds (Predicted Probability)
probability_preds <- log_fit %>%
  predict(new_data = bank_test_A,
          type = 'prob')
probability_preds


# combine the probability threshold and class to see classification
# comparing results to know the results of both
classifier <- probability_preds%>%
  mutate(result = class_preds)
view(classifier)


# combining the results (actual vs predicted)
bank_results_1 <- bank_test_A%>%
  select(Personal.Loan)%>%
  bind_cols(class_preds, probability_preds)
bank_results_1
View(bank_results_1)


                           # now performing the confusing matrix
# model Performance checking and reducing errors and visualizing
conf_mat(bank_results_1,
         truth = Personal.Loan,
         estimate = .pred_class)%>%
  autoplot(type = 'heatmap')


                         # model accuracy
accuracy(bank_results_1,
         truth = Personal.Loan,
         estimate = .pred_class)

# predicting the sensitivity
sens(bank_results_1,
     truth = Personal.Loan,
     estimate = .pred_class)

# predicting the no sensitivity
spec(bank_results_1,
     truth = Personal.Loan,
     estimate = .pred_class)


              # Displaying the curve whether the outcome is good or poor
# calculate area under curve (AUC)
bank_results_1%>%
  roc_curve(truth = Personal.Loan, .pred_Approved)%>%
  autoplot()


# plot receiver operating curve
bank_results_1%>%
  roc_auc(truth = Personal.Loan, .pred_Approved)

 # We had a poor outcome since majority of the participant were denied approval





                               # model Two
                      # Second Model (linear regression)
bank_sec_B <- initial_split(Combinded_data, prop = 0.75,
                            strata = Personal.Loan)


bank_train_B <- bank_sec_B |>
  training()


bank_test_B <- bank_sec_B |>
  testing()

#Initialize Linear Regression Object
linear_model <- linear_reg()%>%
  set_engine('lm')%>%
  set_mode('regression')

#Train the model with training data
# Convert the factor column to character and then to numeric
Combinded_data$Personal.Loan <- as.numeric(as.character(
  Combinded_data$Personal.Loan))


# Check the conversion
str(Combinded_data)
glimpse(Combinded_data)

lm_fit <- linear_model%>%
  fit(Personal.Loan ~ ., 
      data= bank_train_B)

lm_fit  

#Predict Selling Price 
bank_predictions <- predict(lm_fit,
                            new_data = bank_test_B)

bank_predictions



#Combine test data with predictions
bank_test_results <- bank_test_B%>%
  select(Personal.Loan, Age,Income,Family,
         Education, Securities.Account,CCAvg)%>%
  bind_cols(bank_predictions)

View(bank_test_results)  

#Model Evaluation
#Common Metrics
#Root Mean Square Error (RMSE)
#R- Square Metric (RSQ)


#Calculate the RMSE Metric
bank_test_results%>%
  rmse(truth = Personal.Loan, estimate = .pred)


#Determining the R-Squared Metric
bank_test_results%>%
  rsq(truth = Personal.Loan, estimate = .pred)

#Lets create an R squared plot to view model performance

ggplot(bank_test_results,
       aes(x=Personal.Loan, y = .pred))+
  geom_point(alpha = 0.5)+
  geom_abline(color = 'blue', linetype = 2)+
  coord_obs_pred()+
  labs(x = 'Actual Bank Decision',
       y = 'Predicted Bank Decision')



                # MODEL THREE
#          DATA PREPROCESSING
bank_sec_A <- initial_split(data, prop = 0.75,
                            strata = Personal.Loan)


# creating the testing and training data set
# train data set
bank_train_A <- bank_sec_A%>%
  training()
glimpse(bank_train_A)
nrow(bank_train_A)

# test data set
bank_test_A <- bank_sec_A%>%
  testing()
glimpse(bank_test_A)
nrow(bank_test_A)

hrtree <- rpart(Personal.Loan ~ Age + Experience + Income + Family + CCAvg +
                Education + CreditCard +
                Online + CD.Account,
                data = bank_train_A)

hrtree
?rpart.plot
rpart.plot(hrtree, type = 0, extra = 0)
rpart.plot(hrtree, type = 0, extra = 104)


# model specification
dt_model <- decision_tree()%>%
  set_engine('rpart')%>%
  set_mode('classification')

# model fitting
dt_fit <- dt_model%>%
  fit(Personal.Loan ~ .,
      data = bank_train_A)

# predict outcome categories
dt_class_pred <- dt_fit %>%
  predict(new_data = bank_test_A,
          type = 'class')
dt_class_pred


# estimated probability thresholds
dt_prob_pred <- dt_fit %>%
  predict(new_data = bank_test_A,
          type = 'prob')
dt_prob_pred


# combining the results (actual vs predicted)
dt_results <- bank_test_A%>%
  select(Personal.Loan)%>%
  bind_cols(dt_class_pred,dt_prob_pred)
dt_results

# model Performance 
conf_mat(dt_results,
         truth = Personal.Loan,
         estimate = .pred_class)


# model accuracy
accuracy(dt_results,
         truth = Personal.Loan,
         estimate = .pred_class)

# model sensitivity
sens(dt_results,
     truth = Personal.Loan,
     estimate = .pred_class)

# predicting the no sensitivity
spec(dt_results,
     truth = Personal.Loan,
     estimate = .pred_class)

# calculate area under curve (AUC)
dt_results%>%
  roc_auc(truth = Personal.Loan, .pred_Approved)


# plot receiver operating curve
dt_results%>%
  roc_curve(truth = Personal.Loan, .pred_Approved)%>%
  autoplot()
















