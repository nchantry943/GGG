library(tidymodels)
library(vroom)
library(kknn)

train <- vroom('train.csv')
test <- vroom('test.csv')

## Imputation
# trainmv <- vroom('trainWithMissingValues.csv')
# trainmv <- trainmv %>%
#   mutate(across(where(is.character), as.factor))
# 
# my_recipe <- recipe(type ~., data=trainmv) |>
#   step_impute_knn(hair_length, impute_with = imp_vars(id, has_soul, color), neighbors = 5) |>
#   step_impute_knn(rotting_flesh, impute_with = imp_vars(id, has_soul, color, hair_length), neighbors = 5) |>
#   step_impute_knn(bone_length, impute_with = imp_vars(id, has_soul, color, hair_length, rotting_flesh), neighbors = 5)
# 
# prep <- prep(my_recipe)
# baked <- bake(prep, new_data = trainmv)
# 
# rmse_vec(train[is.na(trainmv)], baked[is.na(trainmv)])

## Regression Tree
my_recipe <- recipe(type ~ ., data = train) |>
  step_mutate_at(color, fn = factor)
  
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

## rf Analysis
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 2000) |>
  set_engine('ranger') |>
  set_mode('classification')

rf_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(rf_mod) 

tuning_grid <- grid_regular(mtry(c(1, 6)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(train, v = 10, repeats = 1)

CV_results <- rf_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_rf <- rf_wf |>
  finalize_workflow(best) |>
  fit(data = train)

rf_preds <- predict(final_wf_rf, new_data = test, type = 'class')

preds_rf <- rf_preds |>
  bind_cols(test) |>
  rename(type = .pred_class) |>
  select(id, type)

vroom_write(x=preds_rf, file = "./RFPreds3.csv", delim=",")

