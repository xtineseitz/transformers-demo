################################################################################
##########################  R version and Misc. Notes ##########################
################################################################################

## R Version: 4.3.3
## RStudio Version: 2024.12.1+563 "Kousa Dogwood" for Windows

################################################################################
################################  Reset Button  ################################
################################################################################

## reset workspace
remove(list = ls())

## update global display preferences
options(scipen=999)                             #discourages scientific notation
options(max.print=1000000)                      #increases output in the console


################################################################################
##############################  Install Libraries  #############################
################################################################################

## Run the following code once, if needed:

#install.packages("text")
#install.packages("rJava")
#install.packages("reticulate")
#install.packages("tidyverse")
#install.packages("psych")

## For an extended installation guide for the text package, please visit:
#     https://www.rtext.org/articles/huggingface_in_r_extended_installation_guide.html
#     [[note to self: the above is outdated; find resources elsewhere...]]

## Note: The text package may require additional setup through the reticulate package.
##       See package documentation for the most up-to-date installation guidance.


################################################################################
###############################  Load Libraries  ###############################
################################################################################

## Load the following libraries into R:
library(text)
library(reticulate)
#library(rJava)
library(tidyverse)
library(psych)

## Set a random seed so results are reproducible across runs:
set.seed(1234)


################################################################################
###############################  Import Data  ##################################
################################################################################

## Import the 2019 SIOP machine learning competition data:
master_data <- read.csv("C:\\Users\\xtine\\Desktop\\Transformers\\Materials\\2019_siop_ml_comp_data.csv")


## Alternate import code for selecting the file manually: 
# master_data = read.csv(file.choose())


################################################################################
###############################  Data Checks  ##################################
################################################################################

## Inspect the dataset structure, variable names, and variable types
glimpse(master_data)

## Note: For data splits, "Train" is used to fit the mode, "Dev" is for 
#        immediate evaluation, and "Test" is for final out-of-sample performance.

## Count number of rows in each data split (Train/Development/Test)
table(master_data$Dataset)

## Examine the proportion of cases in each split
prop.table(table(master_data$Dataset))

## Check for missing values across all columns
colSums(is.na(master_data))

## Check for empty strings in the focal open-ended item
sum(master_data$open_ended_1 == "", na.rm = TRUE)               #should return 0

## Trim leading/trailing whitespace from focal open-ended item
master_data <- master_data %>%
  mutate(open_ended_1 = trimws(open_ended_1))

## Inspect the range and distribution of key numeric variables
summary(master_data$E_Scale_score)
summary(master_data$A_Scale_score)

## Convert dataset split to a factor variable
master_data$Dataset <- factor(master_data$Dataset)
class(master_data$Dataset)                               #verify class == factor

## Inspect distribution of response length (in characters)
nchar(master_data$open_ended_1) %>%
  summary()

## Visualize distribution of response lengths
hist(nchar(master_data$open_ended_1),
     breaks = 30,
     main = "Distribution of Response Lengths",
     xlab = "Number of Characters")

## Check for duplicate respondent IDs
sum(duplicated(master_data$Respondent_ID))


################################################################################
###############################  Split the Data  ###############################
################################################################################

## Next, we'll split the data into two subsets: training data and test data.
##    We'll also remove rows with missing values on the text prompt, the target
##    outcome (Extraversion), or the predictor (Agreeableness).
##    This ensures the training set only contains complete cases.

## Create a subset using only cases in the "Train" split
data_train <- master_data %>%
  filter(Dataset == "Train") %>%
  filter(!is.na(open_ended_1),                    #remove rows with missing text
         !is.na(E_Scale_score),
         !is.na(A_Scale_score))

## Create a subset that only includes cases in either "Dev" or "Test"
data_test <- master_data %>%
  filter(Dataset %in% c("Dev","Test")) %>%
  filter(!is.na(open_ended_1),
         !is.na(E_Scale_score),
         !is.na(A_Scale_score))

##### SMALLER SUBSET FOR DEMO PURPOSES ONLY (remove to keep all training data) #####
data_train <- master_data %>% slice_sample(n = 35)
data_test  <- master_data %>% slice_sample(n = 15)


################################################################################
###########################  Create Text Embeddings  ###########################
################################################################################

## Next, we'll convert open-ended responses into numeric text embeddings using a
##    pretrained transformer model from HuggingFace.

## Generate embeddings
train_embeddings <- textEmbed(  
  texts = data_train$open_ended_1,                    #select raw text responses
  model = "bert-base-uncased",                        #specify weights/tokenizer
  aggregation_from_layers_to_tokens = "concatenate",
  aggregation_from_tokens_to_texts = "mean",
  aggregation_from_tokens_to_word_types = "mean",
  dim_names = FALSE)

## View token-level, text-level, and word-type embeddings
View(train_embeddings)

## Extracts text-level embeddings that will be used in the regression model (one row/response)
train_embeddings$texts


################################################################################
###############################  Train Text Model  #############################
################################################################################

## Train a predictive model using the text embeddings as inputs to predict extraversion 
##    scores. Nested cross-validation estimates performance and reduces the risk of overfitting.

## Train the model
text_model <- textTrain(
  x = train_embeddings$texts$texts,               #predictor matrix (text-level)
  y = data_train$E_Scale_score,                                #outcome variable
  outside_folds = 3,                                         #reduce overfitting
  inside_folds = 5,                                        #tune hyperparameters
  cv_method = "cv_folds")                                      #specifies k-fold

## Inspect model results
text_model$results

## Optionally: Fit a second model that includes a structured numeric predictor,
##    demonstrating how text-based features can be integrated with standard scale scores. 

text_model_appended <- textTrain(
  x = train_embeddings$texts$texts,
  y = data_train$E_Scale_score,
  x_append = data.frame(A_Scale_score = data_train$A_Scale_score),    #specify predictor
  outside_folds = 3,
  inside_folds = 5,
  cv_method = "cv_folds")

## Inspect model results
text_model_appended$results


################################################################################
###############################  Baseline Model  ###############################
################################################################################

## Create a comparison point for the transformer-based model.

## Fit linear regression model
baseline_model <- lm(E_Scale_score ~ A_Scale_score, data = data_train)
summary(baseline_model)

## Generate predicted extraversion scores for the test set
baseline_preds <- predict(baseline_model, newdata = data_test)


################################################################################
###############################  Test Set Embeddings  ##########################
################################################################################

## Convert unseen test responses into embeddings
test_embeddings <- textEmbed(
  texts = data_test$open_ended_1,                     #select raw text responses
  model = "bert-base-uncased",                        #specify weights/tokenizer
  aggregation_from_layers_to_tokens = "concatenate",
  aggregation_from_tokens_to_texts = "mean",
  aggregation_from_tokens_to_word_types = "mean")

## Apply trained text model to the test set to generate out-of-sample predictions *
text_preds <- textPredict(
  text_model,                                             #specify trained model
  test_embeddings$texts$texts)                      #text-level embedding matrix

## Optional: Apply to the appended model instead *
text_preds_appended <- textPredict(
  text_model_appended,                                    #specify trained model
  test_embeddings$texts$texts,                      #text-level embedding matrix
  x_append = data.frame(A_Scale_score = data_test$A_Scale_score))  #include numeric predictor values


################################################################################
###############################  Evaluation  ###################################
################################################################################

## Extract predicted extraversion scores from the text-only model
pred <- text_preds$word_embeddings__ypred

## Extract observed extraversion scores from the test set
actual <- data_test$E_Scale_score

## Model performance can be evaluated using several common metrics, such as 
#     correlation, RMSE, and MAE.

## Compute correlation between predicted and actual scores
cor(pred, actual, use = "complete.obs")

## Compute RMSE (lower = better)
sqrt(mean((pred - actual)^2, na.rm = TRUE))

## Compute MAE (lower = better)
mean(abs(pred - actual), na.rm = TRUE)

## Compare baseline model performance -- does text add value?
cor(baseline_preds, actual, use = "complete.obs") 

## Optional: Evaluate the appended model (embeddings + agreeableness)
pred_appended <- text_preds_appended$word_embeddings__ypred

## Compute correlation, RMSE, and MAE for the appended model
cor(pred_appended, actual, use = "complete.obs")
sqrt(mean((pred_appended - actual)^2, na.rm = TRUE))
mean(abs(pred_appended - actual), na.rm = TRUE)


################################################################################
###############################  Visualization  ################################
################################################################################

## Create a dataframe for plotting predicted versus observed scores (each row = one observation)
eval_df <- data.frame(
  actual = actual,
  predicted = pred)

## Plot predicted scores against actual scores
ggplot(eval_df, aes(x = actual, y = predicted)) +
  geom_point(alpha = .6) +
  geom_smooth(method = "lm", se = FALSE) +         #add best fit regression line
  labs(
    title = "Predicted vs. Actual Extraversion Scores",
    x = "Actual Score",
    y = "Predicted Score"
  ) +
  theme_minimal()

