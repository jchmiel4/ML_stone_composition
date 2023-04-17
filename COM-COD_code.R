# This script is used to make the models for COM vs COD stone types

##### Load in libraries #####
library(caret) # For ML methods
library(gbm) # For gbm
library(rBayesianOptimization) # For Bayesian optimization
library(MLeval) # For ROC-AUC calculation and plot
library(ggplot2) # For graphing
library(fastshap) # For SHAP values
library(shapviz) # For graphing SHAP values

##### Step 1: Load data and clean it up #####
# Load data
COM_COD <- read.table("~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/COM-COD-v5.txt", header = TRUE, sep = "\t", check.names = FALSE, quote = "", stringsAsFactors = FALSE, row.names = 1)

# Remove dates
COM_COD$DATE_OF_24_HR_URINE_COLLECTION <- NULL
COM_COD$DATE_OF_BLOOD_SPECIMEN <- NULL
COM_COD$DATE_OF_URINE_SPECIMEN <- NULL
COM_COD$DATE_OF_VISIT <- NULL
COM_COD$DATE_OF_VISIT_M <- NULL

# Fix factors
COM_COD$SEX <- as.factor(COM_COD$SEX)
COM_COD$STONE_TYPE <- as.factor(COM_COD$STONE_TYPE)

# check proportions
prop.table(table(COM_COD$STONE_TYPE))

#COD       COM 
#0.1294821 0.8705179

##### Step 2: Split data training and testing #####

# Split data training and testing
set.seed(2382)
trainIndex <- createDataPartition(COM_COD$STONE_TYPE, p = 0.8, 
                                  list = FALSE, 
                                  # number of partitions
                                  times = 1)

# Make training data
COM_COD_train <- COM_COD[ trainIndex, ]
# Make testing data
COM_COD_test  <- COM_COD[-trainIndex, ]


# Confirm proportions are ok
#prop.table(table(COM_COD$STONE_TYPE))
#prop.table(table(COM_COD_train$STONE_TYPE))
#prop.table(table(COM_COD_test$STONE_TYPE))


##### Step 3: Pre-process training data: impute and change binary to factors #####
# Define pre-processing values
set.seed(2382)
preProcVals <- preProcess(COM_COD_train, method = "bagImpute")

# Apply pre-processing values to training data
COM_COD_train_imputed <- predict(preProcVals, COM_COD_train)

# Round binary variables (CYSTINURIA remove due to no positives)
COM_COD_train_imputed$U_LEUKOCYTES <- round(COM_COD_train_imputed$U_LEUKOCYTES, digits = 0)
COM_COD_train_imputed$U_PROTEIN <- round(COM_COD_train_imputed$U_PROTEIN, digits = 0)
COM_COD_train_imputed$U_GLUCOSE <- round(COM_COD_train_imputed$U_GLUCOSE, digits = 0)
COM_COD_train_imputed$U_KETONES <- round(COM_COD_train_imputed$U_KETONES, digits = 0)
COM_COD_train_imputed$U_BLOOD <- round(COM_COD_train_imputed$U_BLOOD, digits = 0)
COM_COD_train_imputed$U_NITRITE <- round(COM_COD_train_imputed$U_NITRITE, digits = 0)

COM_COD_train_imputed$GOUT <- round(COM_COD_train_imputed$GOUT, digits = 0)
COM_COD_train_imputed$IBD <- round(COM_COD_train_imputed$IBD, digits = 0)
COM_COD_train_imputed$UTI <- round(COM_COD_train_imputed$UTI, digits = 0)
COM_COD_train_imputed$HYPERTENSION <- round(COM_COD_train_imputed$HYPERTENSION, digits = 0)
COM_COD_train_imputed$CARDIAC <- round(COM_COD_train_imputed$CARDIAC, digits = 0)
COM_COD_train_imputed$STROKE <- round(COM_COD_train_imputed$STROKE, digits = 0)
COM_COD_train_imputed$DIABETES <- round(COM_COD_train_imputed$DIABETES, digits = 0)
COM_COD_train_imputed$SARCOIDOSIS <- round(COM_COD_train_imputed$SARCOIDOSIS, digits = 0)
COM_COD_train_imputed$MEDULLARY_SPONGE_KIDNEY <- round(COM_COD_train_imputed$MEDULLARY_SPONGE_KIDNEY, digits = 0)

# Change binary values to factors
COM_COD_train_imputed$U_LEUKOCYTES <- factor(ifelse(test=COM_COD_train_imputed$U_LEUKOCYTES == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$U_PROTEIN <- factor(ifelse(test=COM_COD_train_imputed$U_PROTEIN == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$U_GLUCOSE <- factor(ifelse(test=COM_COD_train_imputed$U_GLUCOSE == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$U_KETONES <- factor(ifelse(test=COM_COD_train_imputed$U_KETONES == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$U_BLOOD <- factor(ifelse(test=COM_COD_train_imputed$U_BLOOD == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$U_NITRITE <- factor(ifelse(test=COM_COD_train_imputed$U_NITRITE == 0, yes="NEG", no="POS"))

COM_COD_train_imputed$GOUT <- factor(ifelse(COM_COD_train_imputed$GOUT == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$IBD <- factor(ifelse(COM_COD_train_imputed$IBD == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$UTI <- factor(ifelse(COM_COD_train_imputed$UTI == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$HYPERTENSION <- factor(ifelse(COM_COD_train_imputed$HYPERTENSION == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$CARDIAC <- factor(ifelse(COM_COD_train_imputed$CARDIAC == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$STROKE <- factor(ifelse(COM_COD_train_imputed$STROKE == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$DIABETES <- factor(ifelse(COM_COD_train_imputed$DIABETES == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$SARCOIDOSIS <- factor(ifelse(COM_COD_train_imputed$SARCOIDOSIS == 0, yes="NEG", no="POS"))
COM_COD_train_imputed$MEDULLARY_SPONGE_KIDNEY <- factor(ifelse(COM_COD_train_imputed$MEDULLARY_SPONGE_KIDNEY == 0, yes="NEG", no="POS"))

# Add in BMI
COM_COD_train_imputed$BMI <- NA 
COM_COD_train_imputed$BMI <- (COM_COD_train_imputed$WEIGHT)/(COM_COD_train_imputed$HEIGHT/100)^2

# Remove HEIGHT and WEIGHT
COM_COD_train_imputed$WEIGHT <- NULL
COM_COD_train_imputed$HEIGHT <- NULL



###### Step 4: Pre-process testing data: impute and change binary to factors ######


# Apply pre-processing values to training data (CYSTINURIA removed due to no positives)
COM_COD_test_imputed <- predict(preProcVals, COM_COD_test)

# Round binary variables
COM_COD_test_imputed$U_LEUKOCYTES <- round(COM_COD_test_imputed$U_LEUKOCYTES, digits = 0)
COM_COD_test_imputed$U_PROTEIN <- round(COM_COD_test_imputed$U_PROTEIN, digits = 0)
COM_COD_test_imputed$U_GLUCOSE <- round(COM_COD_test_imputed$U_GLUCOSE, digits = 0)
COM_COD_test_imputed$U_KETONES <- round(COM_COD_test_imputed$U_KETONES, digits = 0)
COM_COD_test_imputed$U_BLOOD <- round(COM_COD_test_imputed$U_BLOOD, digits = 0)
COM_COD_test_imputed$U_NITRITE <- round(COM_COD_test_imputed$U_NITRITE, digits = 0)

COM_COD_test_imputed$GOUT <- round(COM_COD_test_imputed$GOUT, digits = 0)
COM_COD_test_imputed$IBD <- round(COM_COD_test_imputed$IBD, digits = 0)
COM_COD_test_imputed$UTI <- round(COM_COD_test_imputed$UTI, digits = 0)
COM_COD_test_imputed$HYPERTENSION <- round(COM_COD_test_imputed$HYPERTENSION, digits = 0)
COM_COD_test_imputed$CARDIAC <- round(COM_COD_test_imputed$CARDIAC, digits = 0)
COM_COD_test_imputed$STROKE <- round(COM_COD_test_imputed$STROKE, digits = 0)
COM_COD_test_imputed$DIABETES <- round(COM_COD_test_imputed$DIABETES, digits = 0)
COM_COD_test_imputed$SARCOIDOSIS <- round(COM_COD_test_imputed$SARCOIDOSIS, digits = 0)
COM_COD_test_imputed$MEDULLARY_SPONGE_KIDNEY <- round(COM_COD_test_imputed$MEDULLARY_SPONGE_KIDNEY, digits = 0)

# Change binary values to factors
COM_COD_test_imputed$U_LEUKOCYTES <- factor(ifelse(test=COM_COD_test_imputed$U_LEUKOCYTES == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$U_PROTEIN <- factor(ifelse(test=COM_COD_test_imputed$U_PROTEIN == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$U_GLUCOSE <- factor(ifelse(test=COM_COD_test_imputed$U_GLUCOSE == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$U_KETONES <- factor(ifelse(test=COM_COD_test_imputed$U_KETONES == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$U_BLOOD <- factor(ifelse(test=COM_COD_test_imputed$U_BLOOD == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$U_NITRITE <- factor(ifelse(test=COM_COD_test_imputed$U_NITRITE == 0, yes="NEG", no="POS"))

COM_COD_test_imputed$GOUT <- factor(ifelse(COM_COD_test_imputed$GOUT == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$IBD <- factor(ifelse(COM_COD_test_imputed$IBD == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$UTI <- factor(ifelse(COM_COD_test_imputed$UTI == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$HYPERTENSION <- factor(ifelse(COM_COD_test_imputed$HYPERTENSION == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$CARDIAC <- factor(ifelse(COM_COD_test_imputed$CARDIAC == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$STROKE <- factor(ifelse(COM_COD_test_imputed$STROKE == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$DIABETES <- factor(ifelse(COM_COD_test_imputed$DIABETES == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$SARCOIDOSIS <- factor(ifelse(COM_COD_test_imputed$SARCOIDOSIS == 0, yes="NEG", no="POS"))
COM_COD_test_imputed$MEDULLARY_SPONGE_KIDNEY <- factor(ifelse(COM_COD_test_imputed$MEDULLARY_SPONGE_KIDNEY == 0, yes="NEG", no="POS"))

# Add in BMI
COM_COD_test_imputed$BMI <- NA 
COM_COD_test_imputed$BMI <- (COM_COD_test_imputed$WEIGHT)/(COM_COD_test_imputed$HEIGHT/100)^2

COM_COD_test_imputed$WEIGHT <- NULL
COM_COD_test_imputed$HEIGHT <- NULL


###### Step 5: Bayesian optimization for hyperparameter tuning ######

## Define the resampling method
ctrl_up <- trainControl(method = "repeatedcv", # repeated cross validation
                        repeats = 3, # three repeates
                        number = 10, #ten folds
                        classProbs = TRUE,
                        sampling = "up",
                        allowParallel = TRUE,
                        savePredictions = "final")


## Use this function to optimize the model.
gbm_fit_bayes <- function(n.trees, interaction.depth, shrinkage, n.minobsinnode) {
  txt <- capture.output(
    mod <- train(STONE_TYPE ~ ., data = COM_COD_train_imputed,
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

# Output: 16 minutes
# Round = 30	n.trees = 617.0000	interaction.depth = 2.0000	shrinkage = 0.0392583	n.minobsinnode = 9.0000	Value = 0.2165965


BO_tuning <- data.frame(n.trees = gbm_BO_search$Best_Par["n.trees"],
                        interaction.depth = gbm_BO_search$Best_Par["interaction.depth"],
                        shrinkage = gbm_BO_search$Best_Par["shrinkage"],
                        n.minobsinnode = gbm_BO_search$Best_Par["n.minobsinnode"])

# write BO parameters to table
write.table(BO_tuning, "C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/data/COM_COD_BO_parms.txt")

# Make gbm model with the BO hyperparameters
set.seed(2382)
gbm_model <- train(STONE_TYPE ~ ., data = COM_COD_train_imputed,
                   method = "gbm",
                   tuneGrid = data.frame(n.trees = gbm_BO_search$Best_Par["n.trees"],
                                         interaction.depth = gbm_BO_search$Best_Par["interaction.depth"],
                                         shrinkage = gbm_BO_search$Best_Par["shrinkage"],
                                         n.minobsinnode = gbm_BO_search$Best_Par["n.minobsinnode"]),
                   metric = "Kappa",
                   trControl = ctrl_up,
                   verbose = FALSE)


###### Step 6: Make GLM model ######
set.seed(2382)
glm_model <- train(STONE_TYPE ~ ., data = COM_COD_train_imputed,
                   method = "glm",
                   metric = "Kappa",
                   trControl = ctrl_up,
                   family = binomial(link = "logit")
)


###### Step 7: Compare models and collect model statistics ######

# Save model predictions on testing data
gbm_pred <- postResample(predict(gbm_model, COM_COD_test_imputed), COM_COD_test_imputed$STONE_TYPE)
glm_pred <- postResample(predict(glm_model, COM_COD_test_imputed), COM_COD_test_imputed$STONE_TYPE)

gbm_pred
glm_pred

#Accuracy     Kappa 
#0.820000 0.204244  gbm_pred
#0.740000 0.245502  glm_pred

# Generate confusion matrices and model assessment values
gbm_conf <- confusionMatrix(reference = COM_COD_test_imputed$STONE_TYPE, data = predict(gbm_model, COM_COD_test_imputed), mode = "everything", positive = "COM")
glm_conf <- confusionMatrix(reference = COM_COD_test_imputed$STONE_TYPE, data = predict(glm_model, COM_COD_test_imputed), mode = "everything", positive = "COM")

# save confusion matrix info
cat(capture.output(print(gbm_conf), file="C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/data/COM_COD_conf_mat_gbm.txt"))

cat(capture.output(print(glm_conf), file="C:/Users/john_/Documents/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/data/COM_COD_conf_mat_glm.txt"))


###### Step 8: AUC-ROC values and plot ######

# Calculate AUC-ROC values
COM_COD_gbm_AUCROC <- as.data.frame(evalm(gbm_model)$roc[["data"]])
# AUC-ROC = 0.61
COM_COD_glm_AUCROC <- as.data.frame(evalm(glm_model)$roc[["data"]])
# AUC-ROC = 0.62

# Plot AUC-ROC for GBM and GLM
COM_COD_AUCROC <- ggplot() + 
  geom_line(data = COM_COD_gbm_AUCROC, aes(x = FPR, y = SENS, colour = "GBM (0.61)"),linewidth = 1) +
  scale_colour_manual("AUC-ROC", breaks = c("GBM (0.61)"), values = c("#F8766D")) +
  labs(title = NULL, x = "False Positive Rate", y = "True Positive Rate") +
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
ggsave(COM_COD_AUCROC, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/COM_COD_AUCROC.pdf", width = 42, height = 42, units = "mm", scale = 1.5, useDingbats=FALSE)



###### Step 9: Variable importance and approximate SHAP values ######

# Make function for prediction
pfun <- function(object, newdata) {
  caret::predict.train(object,
                       newdata = newdata,
                       type = "prob")[,2] #Changed to 2 so that COM are positive
  
}

# Generate SHAP values
set.seed(2382)
system.time(
  fastshap <- explain(gbm_model, X = COM_COD_train_imputed[,-3], pred_wrapper = pfun, nsim = 500))

# Output time: ~45 minutes


# Make shapvis object
fastshap_vis <- shapviz(fastshap, X = COM_COD_train_imputed)


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
colnames(fastshap_vis[["X"]]) <- col_fix(fastshap_vis[["X"]])
colnames(fastshap_vis[["S"]]) <- col_fix(fastshap_vis[["S"]])

# View SHAP values
# Global importance
fastshap_imp_all <- sv_importance(fastshap_vis, max_display = 20, fill = "lightblue3") + #dodgerblue3
  scale_y_continuous(expand = c(0,0)) +
  labs(title = "Global Feature Importance") +
  theme_classic() +
  theme(axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9, colour = "black"),
        axis.ticks = element_line(colour = "black"),
        plot.title = element_text(size = 10, colour = "black"),
        plot.background = element_blank())
# save output
ggsave(fastshap_imp_all, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/COM_COD_gbm_global_varimp.pdf", width = 55, height = 65, units = "mm", scale = 1.5, useDingbats=FALSE)


# Local importance (+ = COM and - = COD)
fastshap_imp_bee <- sv_importance(fastshap_vis, kind = "beeswarm", 
                                  max_display = 20,
                                  viridis_args = list(option = "viridis")) +
  guides(color = guide_colorbar(barwidth = 0.5, 
                                barheight = 16,
                                nbin = 300,
                                ticks = FALSE,
                                title.theme = element_text(angle = 90, hjust = 0.5, vjust = 0.5, size = 9),
                                title.position = "left")) +
  labs(title = "Local Feature Importance") +
  geom_hline(yintercept = 0) +
  theme_classic() +
  theme(legend.text = element_text(size = 9),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.line.y = element_blank(),
        axis.ticks.x = element_line(colour = "black"),
        axis.text.x = element_text(size = 9, color = "black"),
        axis.title.x = element_text(size = 9, colour = "black"),
        plot.title = element_text(size = 10, hjust = 0.5),
        plot.background = element_blank())
# save out
ggsave(fastshap_imp_bee, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/COM_COD_gbm_bee_varimp.pdf", width = 55, height = 65, units = "mm", scale = 1.5, useDingbats=FALSE)


## Dependency plots (how a variable interacts with SHAP value)
# 24H Urine Calcium
first_influence <- sv_dependence(fastshap_vis, v = "24H Urine Urea", color = "darkcyan") + 
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_classic() +
  labs(x = "24H Urine Urea (mmol/d)") +
  theme_classic(base_line_size = 0) +
  theme(axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9, colour = "black"),
        axis.ticks = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
        plot.title = element_text(hjust = 0.5),
        plot.background = element_blank())
# save output
ggsave(first_influence, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/first_influence.pdf", width = 33, height = 33, units = "mm", scale = 1.5, useDingbats=FALSE)

# Blood Urate (umol/L)
second_influence <- sv_dependence(fastshap_vis, v = "24H Urine Calcium", color = "darkcyan") + 
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_classic() +
  labs(x = "24H Urine Calcium (mmol/d)") +
  theme_classic(base_line_size = 0) +
  theme(axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9, colour = "black"),
        axis.ticks = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
        plot.title = element_text(hjust = 0.5),
        plot.background = element_blank())
# save output
ggsave(second_influence, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/second_influence.pdf", width = 33, height = 33, units = "mm", scale = 1.5, useDingbats=FALSE)


# Blood Phosphate (mmol/L)
third_influence <- sv_dependence(fastshap_vis, v = "24H Urine Oxalate", color = "darkcyan") + 
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_classic() +
  labs(x = "24H Urine Oxalate (\u03bcmol/L)") +
  theme_classic(base_line_size = 0) +
  theme(axis.text = element_text(size = 9, colour = "black"),
        axis.title = element_text(size = 9, colour = "black"),
        axis.ticks = element_blank(),
        panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
        plot.title = element_text(hjust = 0.5),
        plot.background = element_blank())
# save output
ggsave(third_influence, filename = "~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/figure/third_influence.pdf", width = 33, height = 33, units = "mm", scale = 1.5, useDingbats=FALSE)

###### Step 10: Save R Image ######
save.image(file='~/SCIENCE/Project_ML_urinary_parameters/data/COM-COD/data/COM-COD_data.RData')



