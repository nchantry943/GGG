library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)
library(themis)

train <- vroom('train.csv')
test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = train) |>
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) |>
  step_range(all_numeric_predictors(), min = 0, max = 1) |>
  step_smote(all_outcomes(), neighbors = 6)

nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_mode('classification') |>
  set_engine('naivebayes')

nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10)

folds <- vfold_cv(train, v = 10, repeats = 2)

CV_results <- nb_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

best <- CV_results |>
  select_best()

final_wf_nb <- nb_wf |>
  finalize_workflow(best) |>
  fit(data = train)

nb_preds <- predict(final_wf_nb, new_data = test, type = 'class')

preds_nb <- nb_preds |>
  bind_cols(test) |>
  rename(type = .pred_class) |>
  select(id, type)

vroom_write(x=preds_nb, file = "./NBPreds9.csv", delim=",")
