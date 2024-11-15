library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(embed)
library(naivebayes)
library(discrim)
library(keras)

train <- vroom('train.csv')
test <- vroom('test.csv')

my_recipe <- recipe(formula = type ~ ., data = train) |>
  step_mutate(color= factor(color)) |>
  step_dummy(all_nominal_predictors()) |>
  step_range(all_numeric_predictors(), min = 0, max = 1)
  
model <- mlp(hidden_units = tune(),
             epochs = 50,
             activation = 'softmax') |>
  set_engine('keras') |>
  set_mode('classification')

nn_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(model)  

tuneGrid <- grid_regular(hidden_units(range = c(1, 20)),
                         levels=5)

folds <- vfold_cv(train, v = 5, repeats = 1)

tuned <- nn_wf |>
  tune_grid(resamples = folds,
            grid = tuneGrid,
            metrics = metric_set(accuracy, roc_auc))

## Plot Results
tuned |> collect_metrics() |>
  filter(.metric=='accuracy') |>
  ggplot(aes(x=hidden_units, y = mean)) + geom_line()
