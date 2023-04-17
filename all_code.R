# This script is used to make the models for calcium oxalate vs calcium phosphate vs uric acid stone types

##### Load in libraries #####
library(caret) # For ML methods
library(gbm) # For gbm
library(rBayesianOptimization) # For Bayesian optimization
library(MLeval) # For ROC-AUC calculation and plot
library(ggplot2) # For graphing
library(fastshap) # For SHAP values
library(shapviz) # For graphing SHAP values
library(tidyr) # For data wrangling

###### Step 1: Load data and clean it up ######
# Load data
all <- read.table("~/SCIENCE/Project_ML_urinary_parameters/data/all_2/all_2_v5.txt", header = TRUE, sep = "\t", check.names = FALSE, quote = "", stringsAsFactors = FALSE, row.names = 1)

# Remove dates
all$DATE_OF_24_HR_URINE_COLLECTION <- NULL
all$DATE_OF_BLOOD_SPECIMEN <- NULL
all$DATE_OF_URINE_SPECIMEN <- NULL
all$DATE_OF_VISIT <- NULL
all$DATE_OF_VISIT_M <- NULL

# Fix factors
all$SEX <- as.factor(all$SEX)
all$STONE_TYPE <- as.factor(all$STONE_TYPE)

# Check factors
str(all)


# check proportions
prop.table(table(all$STONE_TYPE))
#       CaOx        CaP      UA    
# 0.6894737 0.1328947 0.1776316 

# How many in each group?
table(all$STONE_TYPE)
# CaOx  CaP   UA 
# 524  101  135

###### Step 2: Split data training and testing ######

# Split data training and testing
set.seed(2382)
trainIndex <- createDataPartition(all$STONE_TYPE, p = 0.80, 
                                  list = FALSE, 
                                  # number of partitions
                                  times = 1)

# Make training data
all_train <- all[ trainIndex, ]
# Make testing data
all_test  <- all[-trainIndex, ]


# Confirm proportions are ok
#prop.table(table(all$STONE_TYPE))
#prop.table(table(all_train$STONE_TYPE))
#prop.table(table(all_test$STONE_TYPE))


###### Step 3: Pre-process training data: impute and change binary to factors ######
# Define pre-processing values
set.seed(2382)
preProcVals <- preProcess(all_train, method = "bagImpute")

# Apply pre-processing values to training data
all_train_imputed <- predict(preProcVals, all_train)

# Round binary variables
all_train_imputed$U_LEUKOCYTES <- round(all_train_imputed$U_LEUKOCYTES, digits = 0)
all_train_imputed$U_PROTEIN <- round(all_train_imputed$U_PROTEIN, digits = 0)
all_train_imputed$U_GLUCOSE <- round(all_train_imputed$U_GLUCOSE, digits = 0)
all_train_imputed$U_KETONES <- round(all_train_imputed$U_KETONES, digits = 0)
all_train_imputed$U_BLOOD <- round(all_train_imputed$U_BLOOD, digits = 0)
all_train_imputed$U_NITRITE <- round(all_train_imputed$U_NITRITE, digits = 0)

all_train_imputed$GOUT <- round(all_train_imputed$GOUT, digits = 0)
all_train_imputed$IBD <- round(all_train_imputed$IBD, digits = 0)
all_train_imputed$UTI <- round(all_train_imputed$UTI, digits = 0)
all_train_imputed$HYPERTENSION <- round(all_train_imputed$HYPERTENSION, digits = 0)
all_train_imputed$CARDIAC <- round(all_train_imputed$CARDIAC, digits = 0)
all_train_imputed$STROKE <- round(all_train_imputed$STROKE, digits = 0)
all_train_imputed$DIABETES <- round(all_train_imputed$DIABETES, digits = 0)
all_train_imputed$SARCOIDOSIS <- round(all_train_imputed$SARCOIDOSIS, digits = 0)
all_train_imputed$MEDULLARY_SPONGE_KIDNEY <- round(all_train_imputed$MEDULLARY_SPONGE_KIDNEY, digits = 0)


# Change binary values to factors
all_train_imputed$U_LEUKOCYTES <- factor(ifelse(all_train_imputed$U_LEUKOCYTES == 0, yes="NEG", no="POS"))
all_train_imputed$U_PROTEIN <- factor(ifelse(all_train_imputed$U_PROTEIN == 0, yes="NEG", no="POS"))
all_train_imputed$U_GLUCOSE <- factor(ifelse(all_train_imputed$U_GLUCOSE == 0, yes="NEG", no="POS"))
all_train_imputed$U_KETONES <- factor(ifelse(all_train_imputed$U_KETONES == 0, yes="NEG", no="POS"))
all_train_imputed$U_BLOOD <- factor(ifelse(all_train_imputed$U_BLOOD == 0, yes="NEG", no="POS"))
all_train_imputed$U_NITRITE <- factor(ifelse(all_train_imputed$U_NITRITE == 0, yes="NEG", no="POS"))

all_train_imputed$GOUT <- factor(ifelse(all_train_imputed$GOUT == 0, yes="NEG", no="POS"))
all_train_imputed$IBD <- factor(ifelse(all_train_imputed$IBD == 0, yes="NEG", no="POS"))
all_train_imputed$UTI <- factor(ifelse(all_train_imputed$UTI == 0, yes="NEG", no="POS"))
all_train_imputed$HYPERTENSION <- factor(ifelse(all_train_imputed$HYPERTENSION == 0, yes="NEG", no="POS"))
all_train_imputed$CARDIAC <- factor(ifelse(all_train_imputed$CARDIAC == 0, yes="NEG", no="POS"))
all_train_imputed$STROKE <- factor(ifelse(all_train_imputed$STROKE == 0, yes="NEG", no="POS"))
all_train_imputed$DIABETES <- factor(ifelse(all_train_imputed$DIABETES == 0, yes="NEG", no="POS"))
all_train_imputed$SARCOIDOSIS <- factor(ifelse(all_train_imputed$SARCOIDOSIS == 0, yes="NEG", no="POS"))
all_train_imputed$MEDULLARY_SPONGE_KIDNEY <- factor(ifelse(all_train_imputed$MEDULLARY_SPONGE_KIDNEY == 0, yes="NEG", no="POS"))



# Add in BMI
all_train_imputed$BMI <- NA 
all_train_imputed$BMI <- (all_train_imputed$WEIGHT)/(all_train_imputed$HEIGHT/100)^2

# Remove HEIGHT and WEIGHT
all_train_imputed$WEIGHT <- NULL
all_train_imputed$HEIGHT <- NULL


###### Step 4: Pre-process testing data: impute and change binary to factors ######

# Apply pre-processing values to training data
all_test_imputed <- predict(preProcVals, all_test)

# Round binary variables
all_test_imputed$U_LEUKOCYTES <- round(all_test_imputed$U_LEUKOCYTES, digits = 0)
all_test_imputed$U_PROTEIN <- round(all_test_imputed$U_PROTEIN, digits = 0)
all_test_imputed$U_GLUCOSE <- round(all_test_imputed$U_GLUCOSE, digits = 0)
all_test_imputed$U_KETONES <- round(all_test_imputed$U_KETONES, digits = 0)
all_test_imputed$U_BLOOD <- round(all_test_imputed$U_BLOOD, digits = 0)
all_test_imputed$U_NITRITE <- round(all_test_imputed$U_NITRITE, digits = 0)

all_test_imputed$GOUT <- round(all_test_imputed$GOUT, digits = 0)
all_test_imputed$IBD <- round(all_test_imputed$IBD, digits = 0)
all_test_imputed$UTI <- round(all_test_imputed$UTI, digits = 0)
all_test_imputed$HYPERTENSION <- round(all_test_imputed$HYPERTENSION, digits = 0)
all_test_imputed$CARDIAC <- round(all_test_imputed$CARDIAC, digits = 0)
all_test_imputed$STROKE <- round(all_test_imputed$STROKE, digits = 0)
all_test_imputed$DIABETES <- round(all_test_imputed$DIABETES, digits = 0)
all_test_imputed$SARCOIDOSIS <- round(all_test_imputed$SARCOIDOSIS, digits = 0)
all_test_imputed$MEDULLARY_SPONGE_KIDNEY <- round(all_test_imputed$MEDULLARY_SPONGE_KIDNEY, digits = 0)

# Change binary values to factors
all_test_imputed$U_LEUKOCYTES <- factor(ifelse(all_test_imputed$U_LEUKOCYTES == 0, yes="NEG", no="POS"))
all_test_imputed$U_PROTEIN <- factor(ifelse(all_test_imputed$U_PROTEIN == 0, yes="NEG", no="POS"))
all_test_imputed$U_GLUCOSE <- factor(ifelse(all_test_imputed$U_GLUCOSE == 0, yes="NEG", no="POS"))
all_test_imputed$U_KETONES <- factor(ifelse(all_test_imputed$U_KETONES == 0, yes="NEG", no="POS"))
all_test_imputed$U_BLOOD <- factor(ifelse(all_test_imputed$U_BLOOD == 0, yes="NEG", no="POS"))
all_test_imputed$U_NITRITE <- factor(ifelse(all_test_imputed$U_NITRITE == 0, yes="NEG", no="POS"))

all_test_imputed$GOUT <- factor(ifelse(all_test_imputed$GOUT == 0, yes="NEG", no="POS"))
all_test_imputed$IBD <- factor(ifelse(all_test_imputed$IBD == 0, yes="NEG", no="POS"))
all_test_imputed$UTI <- factor(ifelse(all_test_imputed$UTI == 0, yes="NEG", no="POS"))
all_test_imputed$HYPERTENSION <- factor(ifelse(all_test_imputed$HYPERTENSION == 0, yes="NEG", no="POS"))
all_test_imputed$CARDIAC <- factor(ifelse(all_test_imputed$CARDIAC == 0, yes="NEG", no="POS"))
all_test_imputed$STROKE <- factor(ifelse(all_test_imputed$STROKE == 0, yes="NEG", no="POS"))
all_test_imputed$DIABETES <- factor(ifelse(all_test_imputed$DIABETES == 0, yes="NEG", no="POS"))
all_test_imputed$SARCOIDOSIS <- factor(ifelse(all_test_imputed$SARCOIDOSIS == 0, yes="NEG", no="POS"))
all_test_imputed$MEDULLARY_SPONGE_KIDNEY <- factor(ifelse(all_test_imputed$MEDULLARY_SPONGE_KIDNEY == 0, yes="NEG", no="POS"))


# Add in BMI
all_test_imputed$BMI <- NA 
all_test_imputed$BMI <- (all_test_imputed$WEIGHT)/(all_test_imputed$HEIGHT/100)^2

# Remove HEIGHT and WEIGHT
all_test_imputed$WEIGHT <- NULL
all_test_imputed$HEIGHT <- NULL


###### Step 5: Bayesian optimization for hyperparameter tuning ######

## Define the resampling method
ctrl_up <- trainControl(method = "repeatedcv", 
                        repeats = 3,
                        number = 10,
                        classProbs = TRUE,
                        sampling = "up",
                        allowParallel = TRUE,
                        savePredictions = "final")


## Use this function to optimize the model.
gbm_fit_bayes <- function(n.trees, interaction.depth, shrinkage, n.minobsinnode) {
  txt <- capture.output(
    mod <- train(STONE_TYPE ~ ., data = all_train_imputed,
                 method = "gbm",
                 metric = "Kappa",
                 trControl = ctrl_up,
                 tuneGrid = data.frame(n.trees = n.trees, interaction.depth = interaction.depth, shrinkage = shrinkage, n.minobsinnode = n.minobsinnode))
  )
  list(Score = getTrainPerf(mod)[, "TrainKappa"], Pred = 0)
}


## Run the optimization
set.seed(2382)
system.time(
  gbm_BO_search <- BayesianOptimization(gbm_fit_bayes,
                                        # Define parameter bounds
                                        bounds = list(n.trees = c(50L, 2000L),
                                                      interaction.depth = c(1L, 5L),
                                                      shrinkage = c(0.001, 0.1),
                                                      n.minobsinnode = c(1L, 20L)
                                        ),
                                        init_points = 10, 
                                        n_iter = 20,
                                        acq = "ucb", 
                                        kappa =  2.576, # Default
                                        eps = 0.0, # Default
                                        verbose = FALSE)
)

# Output: time 5600 sec (158 min)
# Round = 27	n.trees = 915.0000	interaction.depth = 4.0000	shrinkage = 0.009055568	n.minobsinnode = 20.0000	Value = 0.3055781
# Warning: SARCOIDOSISPOS has no variation



BO_tuning <- data.frame(n.trees = gbm_BO_search$Best_Par["n.trees"],
                        interaction.depth = gbm_BO_search$Best_Par["interaction.depth"],
                        shrinkage = gbm_BO_search$Best_Par["shrinkage"],
                        n.minobsinnode = gbm_BO_search$Best_Par["n.minobsinnode"])

# Save output
write.table(BO_tuning, "C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/all_2/all_BO_parms.txt")

# Make gbm model with the BO hyperparameters
set.seed(2382)
gbm_model <- train(STONE_TYPE ~ ., data = all_train_imputed,
                   method = "gbm",
                   tuneGrid = data.frame(n.trees = gbm_BO_search$Best_Par["n.trees"],
                                         interaction.depth = gbm_BO_search$Best_Par["interaction.depth"],
                                         shrinkage = gbm_BO_search$Best_Par["shrinkage"],
                                         n.minobsinnode = gbm_BO_search$Best_Par["n.minobsinnode"]),
                   metric = "Kappa",
                   trControl = ctrl_up,
                   verbose = FALSE)


###### Step 6: Make multinom model ######
set.seed(2382)
system.time(
multinom_model <- train(STONE_TYPE ~ ., data = all_train_imputed,
                   method = "multinom",
                   metric = "Kappa",
                   trControl = ctrl_up)
)

# Output time: <1 minute

###### Step 7: Compare models and collect model statistics ######

# Save model predictions on testing data
gbm_pred <- postResample(predict(gbm_model, all_test_imputed), all_test_imputed$STONE_TYPE)
multinom_pred <- postResample(predict(multinom_model, all_test_imputed), all_test_imputed$STONE_TYPE)

gbm_pred
multinom_pred

#  Accuracy     Kappa
# 0.6622517 0.3023193  gbm
# 0.5894040 0.3295617  MLR



# Generate confusion matrices and model assessment values
gbm_conf <- confusionMatrix(reference = all_test_imputed$STONE_TYPE, data = predict(gbm_model, all_test_imputed), mode = "everything")
multinom_conf <- confusionMatrix(reference = all_test_imputed$STONE_TYPE, data = predict(multinom_model, all_test_imputed), mode = "everything")

# save confusion matrix info
cat(capture.output(print(gbm_conf), file="C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/all_2/data/gbm_BO_conf_mat.txt"))

cat(capture.output(print(multinom_conf), file="C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/all_2/data/multinom_BO_conf_mat.txt"))


###### Step 8: AUC-ROC values and plot ######

# Calculate AUC-ROC values
all_AUCROC <- as.data.frame(evalm(gbm_model)$roc[["data"]])

# AUC-ROC
# 0.78 (GBM)
# 0.79 (MLR)

# Plot AUC-ROC for GBM
all_gbm_AUCROC <- ggplot() + 
  geom_line(data = all_AUCROC, aes(x = FPR, y = SENS, colour = "GBM (0.78)"),linewidth = 1) + 
  scale_colour_manual("AUC-ROC", breaks = c("GBM (0.78)"), values = c("#F8766D")) +
  labs(title = NULL, x = "False Positive Rate", y = "Trus Positive Rate") +
  geom_abline(colour = "grey50") +
  theme_classic(base_line_size = 0) + 
  theme(panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
        legend.title = element_text(size = 9, face = "bold"),
        legend.text = element_text(size = 9),
        axis.text.x = element_text(size = 9, colour = "black"),
        legend.background = element_blank(),
        axis.text.y = element_text(size = 9, colour = "black"),
        axis.ticks = element_line(colour = "black"),
        axis.title.x = element_text(size = 9, colour = "black"),
        axis.title.y = element_text(size = 9, colour = "black"),
        legend.position = c(0.7, 0.2),
        plot.background = element_blank())

# Save it
ggsave(all_gbm_AUCROC, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/all_2/figure/all_AUCROC.pdf", width = 42, height = 42, units = "mm", scale = 1.5, useDingbats=FALSE)


###### Step 9: Variable importance and approximate SHAP values ######

# Make function for each class
pfun_caox <- function(object, newdata) {
  caret::predict.train(object,
                       newdata = newdata,
                       type = "prob")[,1]
  
}

pfun_cap <- function(object, newdata) {
  caret::predict.train(object,
                       newdata = newdata,
                       type = "prob")[,2]
  
}

pfun_ua <- function(object, newdata) {
  caret::predict.train(object,
                       newdata = newdata,
                       type = "prob")[,3]
  
}

# Generate SHAP values for each classification
# CaOx
set.seed(2382)
system.time(
  fastshap_caox <- explain(gbm_model, X = all_train_imputed[,-3], pred_wrapper = pfun_caox, nsim = 500)
)
# Output time: ~80 minutes

# CaP
set.seed(2382)
system.time(
  fastshap_cap <- explain(gbm_model, X = all_train_imputed[,-3], pred_wrapper = pfun_cap, nsim = 500)
)
# Output time: ~75 minutes

# UA
set.seed(2382)
system.time(
  fastshap_ua <- explain(gbm_model, X = all_train_imputed[,-3], pred_wrapper = pfun_ua, nsim = 500)
)
# Output time: ~80 minutes


# Make shapvis object
fastshap_vis_caox <- shapviz(fastshap_caox, X = all_train_imputed[,-3])
fastshap_vis_cap <- shapviz(fastshap_cap, X = all_train_imputed[,-3])
fastshap_vis_ua <- shapviz(fastshap_ua, X = all_train_imputed[,-3])




# Define function to fix columns
col_fix<- function(x){
  ifelse(colnames(x) == "SEX", "Sex",
  ifelse(colnames(x) == "AGE", "Age",
  #
  ifelse(colnames(x) == "U_VOLUME", "24H Urine Volume",
  ifelse(colnames(x) == "U_SODIUM", "24H Urine Sodium",
  ifelse(colnames(x) == "U_CREATININE", "24H Urine Creatinine",
  ifelse(colnames(x) == "U_PHOSPHATE", "24H Urine Phosphate",
  ifelse(colnames(x) == "U_URATE", "24H Urine Urate",
  ifelse(colnames(x) == "U_CALCIUM", "24H Urine Calcium",
  ifelse(colnames(x) == "U_UREA", "24H Urine Urea",
  ifelse(colnames(x) == "U_OXALATE", "24H Urine Oxalate",
  ifelse(colnames(x) == "U_CITRATE", "24H Urine Citrate",
  #
  ifelse(colnames(x) == "U_LEUKOCYTES", "Urine Leukocytes",
  ifelse(colnames(x) == "U_PH", "Urine pH",
  ifelse(colnames(x) == "U_PROTEIN", "Urine Protein",
  ifelse(colnames(x) == "U_GLUCOSE", "Urine Glucose",
  ifelse(colnames(x) == "U_KETONES", "Urine Ketones",
  ifelse(colnames(x) == "U_BLOOD", "Urine Blood",
  ifelse(colnames(x) == "U_NITRITE", "Urine Nitrite", 
  #  
  ifelse(colnames(x) == "B_SODIUM", "Blood Sodium",
  ifelse(colnames(x) == "B_POTASSIUM", "Blood Potassium",
  ifelse(colnames(x) == "B_CHLORIDE", "Blood Chloride",
  ifelse(colnames(x) == "B_BICARBONATE", "Blood Bicarbonate",
  ifelse(colnames(x) == "B_UREA", "Blood Urea",
  ifelse(colnames(x) == "B_CREATININE", "Blood Creatinine",
  ifelse(colnames(x) == "B_TOTAL_CALCIUM", "Blood Total Calcium",  
  ifelse(colnames(x) == "B_PHOSPHATE", "Blood Phosphate",
  ifelse(colnames(x) == "B_URATE", "Blood Urate",
  ifelse(colnames(x) == "B_PARATHYROID", "Blood PTH",
  ifelse(colnames(x) == "B_VITAMIN_DH25", "Blood 25(OH)D",  
  #
  ifelse(colnames(x) == "GOUT", "Gout",
  ifelse(colnames(x) == "IBD", "IBD",
  ifelse(colnames(x) == "HYPERTENSION", "Hypertension",
  ifelse(colnames(x) == "CARDIAC", "Cardiac",
  ifelse(colnames(x) == "STROKE", "Stroke",
  ifelse(colnames(x) == "DIABETES", "Diabetes",
  ifelse(colnames(x) == "SARCOIDOSIS", "Sarcoidosis",
  ifelse(colnames(x) == "CYSTINURIA", "Cystinuria",
  ifelse(colnames(x) == "MEDULLARY_SPONGE_KIDNEY", "Medullary Sponge Kidney", 
  #
  ifelse(colnames(x) == "BMI", "BMI", colnames(x))))))))))))))))))))))))))))))))))))))))
}

# Apply new column name function to fastshap object
colnames(fastshap_vis_caox[["X"]]) <- col_fix(fastshap_vis_caox[["X"]])
colnames(fastshap_vis_caox[["S"]]) <- col_fix(fastshap_vis_caox[["S"]])

colnames(fastshap_vis_cap[["X"]]) <- col_fix(fastshap_vis_cap[["X"]])
colnames(fastshap_vis_cap[["S"]]) <- col_fix(fastshap_vis_cap[["S"]])

colnames(fastshap_vis_ua[["X"]]) <- col_fix(fastshap_vis_ua[["X"]])
colnames(fastshap_vis_ua[["S"]]) <- col_fix(fastshap_vis_ua[["S"]])


# Making global importance plot
mean_fastshap_imp <- as.data.frame(rbind(colMeans(abs(fastshap_imp_caox)), colMeans(abs(fastshap_imp_cap)), colMeans(abs(fastshap_imp_ua))))

# Define stone type
mean_fastshap_imp$"Stone_Type" <- c("Calcium Oxalate", "Calcium Phosphate", "Uric Acid")

# Fix titles
mean_fastshap_imp_long <- pivot_longer(mean_fastshap_imp,
                    cols = c("Sex", "Age", "24H Urine Volume", "24H Urine Sodium", "24H Urine Creatinine", "24H Urine Phosphate", "24H Urine Urate", "24H Urine Calcium", "24H Urine Urea", "24H Urine Oxalate", "24H Urine Citrate", "Urine Leukocytes", "Urine pH", "Urine Protein", "Urine Glucose", "Urine Ketones", "Urine Blood", "Urine Nitrite", "Blood Sodium", "Blood Potassium", "Blood Chloride", "Blood Bicarbonate", "Blood Urea", "Blood Creatinine", "Blood Total Calcium", "Blood Phosphate", "Blood Urate", "Blood PTH", "Blood 25(OH)D", "Gout", "IBD", "UTI", "Hypertension", "Cardiac", "Stroke", "Diabetes", "Sarcoidosis", "Medullary Sponge Kidney", "BMI"), 
                    names_to = "Predictor", values_to = "SHAP")


# SHAP global
fastshap_imp_all <- ggplot() +
  geom_bar(data = mean_fastshap_imp_long, aes(fill = Stone_Type, y = SHAP, x = reorder(Predictor, SHAP)), position = position_stack(reverse = TRUE), stat="identity", width = 0.75) + 
  coord_flip() +
  scale_fill_manual(values = c("#5B8ADC", "#F33E93", "#FFD757")) + #5CBD6E #CE6495 #648ACB
  scale_y_continuous(expand = c(0,0)) +
  labs(title = "Global Feature Importance", x = NULL, y = "mean(|SHAP value|)", fill = "Stone Type") +
  theme_classic() +
  theme(legend.text = element_text(size = 9),
        legend.title = element_text(size = 9),
        legend.background = element_blank(),
        legend.position = c(0.65, 0.075),
        legend.key.size = unit(0.75,"line"),
        axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9, colour = "black"),
        axis.ticks = element_line(colour = "black"),
        plot.title = element_text(size = 10, hjust = 0.5, colour = "black"),
        plot.background = element_blank())
# Save output
ggsave(fastshap_imp_all, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/all_2/figure/all_gbm_global_varimp.pdf", width = 60, height = 90, units = "mm", scale = 1.5, useDingbats=FALSE)

###### Step 10: Save R Image ######

save.image(file='~/SCIENCE/Project_ML_urinary_parameters/data/all_2/data/all_2_data.RData')
