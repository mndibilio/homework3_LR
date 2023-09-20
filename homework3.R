library(tidyverse)
library(DescTools)
library(vcdExtra)
library(car)
library(Hmisc)
library(ROCit)
library(caret)

# Read in binned train data
train_bin <- read.csv('/Users/marissadibilio/Downloads/Homework2_LR/insurance_t_bin.csv')
val <- read.csv('/Users/marissadibilio/Downloads/Homework2_LR/insurance_v_bin.csv')

# Add a missing category value for any missing values for train and val
train_bin <- train_bin %>%
    mutate_at(vars(-INS), ~ replace_na(as.character(.), "MV"))
val <- val %>%
  mutate_at(vars(-INS), ~ replace_na(as.character(.), "MV"))

#As factor for train and val
factor_var <- list()
for (x in colnames(train_bin)) {
  if(grepl("Bin", x)) {
    train_bin[[x]] <- as.factor(train_bin[[x]])
    factor_var <- c(factor_var, x)
  }
}
print(factor_var)

factor_var <- list()
for (x in colnames(val)) {
  if(grepl("Bin", x)) {
    val[[x]] <- as.factor(val[[x]])
    factor_var <- c(factor_var, x)
  }
}
print(factor_var)

as.factor(val$INV)
as.factor(train_bin$INV)

# Helper function to check if any value in the crosstab is 0 with INS
test_convergence <- function(var) {
    table <- table(train_bin$INS, train_bin[[var]])
    if (any(table(train_bin$INS, train_bin[[var]]) == 0)) {
        return(TRUE)
    }
    return(FALSE)
}
convergence_tables <- sapply(colnames(train_bin), test_convergence)
convergence_tables[convergence_tables == TRUE & names(convergence_tables) != "INS"]

# Check which categories are creating convergence problems for CASHBK, MMCRED
table(train_bin$INS, train_bin$CASHBK)
table(train_bin$INS, train_bin$MMCRED)

# Threshold CASHBK and MMCRED categories for train and val
train_bin <- train_bin %>%
    mutate(CASHBK = replace(CASHBK, CASHBK == 2, 1)) %>%
    mutate(CASHBK = replace(CASHBK, CASHBK == 1, "1+"))

train_bin <- train_bin %>%
    mutate(MMCRED = replace(MMCRED, MMCRED == 5, 3)) %>%
    mutate(MMCRED = replace(MMCRED, MMCRED == 3, "3+"))

val <- val %>%
  mutate(CASHBK = replace(CASHBK, CASHBK == 2, 1)) %>%
  mutate(CASHBK = replace(CASHBK, CASHBK == 1, "1+"))

val <- val %>%
  mutate(MMCRED = replace(MMCRED, MMCRED == 5, 3)) %>%
  mutate(MMCRED = replace(MMCRED, MMCRED == 3, "3+"))

alpha <- 0.002
full_model <- glm(INS ~ ., data = train_bin, family = binomial(link = "logit"))
empty_model <- glm(INS ~ 1, data = train_bin, family = binomial(link = "logit"))

# Use forward selection to select a model with interactions based on p-values
if (file.exists("forward_model_int.rds")) {
    forward_model_int <- readRDS("forward_model_int.rds")
} else {
    forward_model_int <- step(
        lower_model_int,
        scope = list(
            lower = lower_model_int,
            upper = full_model_int
        ),
        direction = "forward",
        k = qchisq(alpha, 1, lower.tail = FALSE),
        trace = FALSE
    )
    saveRDS(forward_model_int, "forward_model_int.rds")
}

# Run LRT on interaction model to get p-values
summary(forward_model_int)
forward_int_lrt <- Anova(forward_model_int, test = "LR", type = "III", singular.ok = TRUE)

colnames(forward_int_lrt)[3] <- "pvalue"

# Our only selected interaction DDA:IRA has a pvalue of 2.517e-04
forward_int_lrt %>%
    select(pvalue) %>%
    arrange(pvalue) %>%
    print.data.frame()

#Concordance
train_bin$p_hat <- predict(forward_model_int, type = 'response')
somers2(train_bin$p_hat, train_bin$INS)

#Discrimination slope
p1 <- train_bin$p_hat[train_bin$INS == 1]
p0 <- train_bin$p_hat[train_bin$INS == 0]
coef_discrim <- mean(p1) - mean(p0)

ggplot(train_bin, aes(p_hat, fill = factor(INS))) + 
  geom_density(alpha = 0.7) + scale_fill_grey() + 
  scale_fill_manual(values = c("blue", "red")) +
  labs(x = "Predicted Probability", y = "Density", fill = "Outcome",
       title = paste("Coefficient of Discrimination = ", round(coef_discrim, 3), sep = "")) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust=0.5),
        panel.background = element_rect(fill = "lightgray"))

#ROC Curve
logit_meas <- measureit(train_bin$p_hat, train_bin$INS, measure = c('ACC', 'SENS', 'SPEC'))
logit_roc <- rocit(train_bin$p_hat, train_bin$INS)
opt <- plot(logit_roc)$optimal
optimal_fpr <- opt[2]
optimal_tpr <- opt[3]
  
roc <- data.frame(FPR = logit_roc$FPR,
                  TPR = logit_roc$TPR,
                  Cutoff = logit_roc$Cutoff)

ggplot(roc, aes(x=FPR, y=TPR)) +
  geom_line(color = "deepskyblue2", size = 1) + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "blue") +
  geom_point(x = optimal_fpr, y = optimal_tpr, color = "blue") +
  geom_text(x = optimal_fpr, y = optimal_tpr,
            label = "Optimal (Youden Index) Point",
            hjust = -0.2, vjust = -0.2) +
  labs(x="1 - Specificity (FPR)", y="Sensitivity (TPR)") +
  ggtitle("ROC Curve") + 
  theme_minimal() +
  theme(plot.title = element_text(size=16, hjust=0.5),
        panel.background = element_rect(fill = "lightgray"))

#K-S Statistic 
ksplot(logit_roc)
ksplot(logit_roc)$'KS stat'
ksplot(logit_roc)$'KS Cutoff'

#Validation metrics
val$p_hat <- predict(forward_model_int, val, type = 'response')
logit_roc_val <- rocit(val$p_hat, val$INS)
logit_lift <- gainstable(logit_roc_val)
print(logit_lift)
plot(logit_lift)

lift <- data.frame(
  Population_Depth = logit_lift$Depth,
  Lift = logit_lift$Lift,
  Cum_Lift = logit_lift$CLift
)

ggplot(lift, aes(x=Population_Depth)) + 
  geom_line(aes(y=Lift), color="blue") +
  geom_line(aes(y = Cum_Lift), linetype = "dashed", color="cornflowerblue") +
  labs(
    x="Population Depth",
    y="Lift & Cumulative Lift",
    title = "Lift"
  ) +
  theme_minimal() +
  theme(plot.background = element_rect(fill = "gray94"),
        panel.grid.major = element_line(color="gray"),
        plot.title = element_text(size = 16, hjust=0.5))

#Accuracy on val
logit_val_meas <- measureit(val$p_hat, val$INS, measure = c("ACC"), cutoff = 0.298)
summary(logit_val_meas)

acc_table <- data.frame(Cutoff = logit_val_meas$Cutoff, Acc = logit_val_meas$ACC)
head(arrange(acc_table, desc(Acc)), n=10)

cutoff <- 0.298
acc_table %>%
  filter(Cutoff == 0.2977325)

bin_pred <- ifelse(val$p_hat > cutoff, 1, 0)
actual <- val$INS
n <- length(actual)
TP <- sum(bin_pred == 1 & actual == 1)
TN <- sum(bin_pred == 0 & actual == 0)
accuracy <- (TP + TN) / n
print(accuracy)

#Confusion matrix
val <- val %>%
  mutate(p_hat = ifelse(val$p_hat > 0.5, 1, 0))

table(val$p_hat, val$INS)
