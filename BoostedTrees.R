library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)

train <- vroom('train.csv')
test <- vroom('test.csv')

my_recipe <- recipe(type ~ ., data = train) |>
  step_mutate_at(color, fn = factor)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

boost <- boost_tree(tree_depth = tune(),
                    trees = tune(),
                    learn_rate = tune()) |>
  set_engine('lightgbm') |>
  set_mode('classification')

# bart <- bart(trees = tune()) |>
#   set_engine('dbarts') |>
#   set_mode('classification')

boost_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(boost) 

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

folds <- vfold_cv(train, v = 10, repeats = 1)

CV_results <- boost_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_boost <- boost_wf |>
  finalize_workflow(best) |>
  fit(data = train)

boost_preds <- predict(final_wf_boost, new_data = test, type = 'class')

preds_boost <- boost_preds |>
  bind_cols(test) |>
  rename(type = .pred_class) |>
  select(id, type)

vroom_write(x=preds_boost, file = "./BoostedPreds.csv", delim=",")
save(file = "./BoostedPreds.csv")
