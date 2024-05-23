# Used to double-check hosmer-lemenshow test statistics from 3-2
library(ResourceSelection)
library(tidyverse)

data <- read_csv("./tables/logreg/Model Eval Uncomplicated Colic - Discharge vs Admission Feature Scores.csv")
hoslem.test(data$predicted, data$expected, g=40)
# X-squared = 7023.4, df = 38, p-value < 2.2e-16

data <- read_csv("./tables/logreg/Model Eval Uncomplicated Colic - Given Admission - Immediate Surgery vs Others Feature Scores.csv")
hoslem.test(data$predicted, data$expected, g=40)
# X-squared = 3.7046e-10, df = 38, p-value = 1

data <- read_csv("./tables/logreg/Model Eval Uncomplicated Colic - Given Discharge - No Surgery vs Others Feature Scores.csv")
hoslem.test(data$predicted, data$expected, g=40)
# X-squared = 11.72, df = 38, p-value = 1

data <- read_csv("./tables/logreg/Model Eval Complicated Colic - Discharge vs Admission Feature Scores.csv")
hoslem.test(data$predicted, data$expected, g=40)
# X-squared = 8458, df = 38, p-value < 2.2e-16

data <- read_csv("./tables/logreg/Model Eval Complicated Colic - Given Admission - Immediate Cholecystectomy vs Others Feature Scores.csv")
hoslem.test(data$predicted, data$expected, g=40)
# X-squared = 3584.6, df = 38, p-value < 2.2e-16
