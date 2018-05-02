library(tidyverse)
library(lubridate)
library(text2vec)
library(tokenizers)
library(Matrix)
library(lightgbm)
library(ggplot2)
options(scipen=999) # turn off scientific notation (for output)

setwd('~/Files/DataScience/HawaiiML/working')

train <- read_csv("../input/hawaiiml-data/train.csv")
test <- read_csv("../input/hawaiiml-data/test.csv")
subm <- read_csv("../input/hawaiiml-data/sample_submission.csv") 

cat("Basic preprocessing & stats...\n")
# remove outliers (https://www.kaggle.com/ahnate/exploration-word-cloud)
train <- train[train$quantity < 60000,]
train <- train[(train$unit_price >= 0) & (train$unit_price < 5000),]
train <-(train[!(train$unit_price > 500 & train$quantity > 50),])

# combine test and train data
full <- bind_rows(train, test)

# log transform quantity and price
full$quantity_log <- log1p(full$quantity)

# format dates
full$date <- mdy(full$date) 
full$dayofweek <- wday(full$date)  # numerical
full$week  <- week(full$date)
full$month <- month(full$date)
full$year  <- year(full$date)
#full$year2011  <- ifelse(year(full$date) == 2011, 1, 0)   # simplify year as binary
full$minuteinday <- as.numeric(full$time)/60   # time in minute, 00:00 as origin

# run 3 ways: 1) just numerical, 2) do binary encoding for invoiceID, stockID,
# customerID 3) keeping as categorical? Given the heavy loading on unit_price
# and customer_id, let's keep these in their original form? Might not be able to
# keep customer_id as categorical, so maybe use boostmtree for that (or xgboost
# with binary encoding?)

# Categorical encoding ---------------------------------------------------------
# https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

# Encoding country as numeric
full$countrynum <- as.numeric(as.factor(full$country))  # numerical encoding since only 38 countries

# create function to encode frequencies for all categorical variables

# FINAL: do binary encoding for customer_id, stock_id, and invoice_id (stock_id maybe
# keep as numeric)


# Create aggregates -------------------------------------------------------
# They're like proxies for customer_ids and invoice_ids

# summarize by customer_id x invoice_id
train_cust_inv <- train %>%
  select(customer_id, invoice_id, unit_price, quantity) %>%
  mutate(tot_price = unit_price * quantity) %>%
  group_by(customer_id, invoice_id) %>%
  summarize(inv_tot_price = sum(tot_price), inv_tot_quant = sum(quantity), inv_n_stockid = n())

# summarize by stock_id
train_stock <- train %>%
  select(stock_id, quantity) %>%
  group_by(stock_id) %>%
  summarize(stock_mean_quant = mean(quantity), stock_median_quant = median(quantity), 
            stock_iqr_quant = IQR(quantity))

# summarize by by customer_id
train_cust <- train %>%
  select(customer_id, quantity) %>%
  group_by(customer_id) %>%
  summarize(cust_mean_quant = mean(quantity), cust_median_quant = median(quantity), 
            cust_iqr_quant = IQR(quantity))
# consider normalizing range (IQR) by median?

# summarize by country
train_country <- train %>%
  select(country, quantity) %>%
  group_by(country) %>%
  summarize(country_mean_quant = mean(quantity), country_median_quant = median(quantity),
            country_iqr_quant = IQR(quantity))

# join summary statistics to train and test data
full <- full %>% 
  left_join(train_cust_inv, by = c("customer_id", "invoice_id")) %>%
  left_join(train_stock, by = c("stock_id")) %>%
  left_join(train_cust, by = c("customer_id")) %>%
  left_join(train_country, by = c("country"))

# Binary encoding  -----------------------------------------

for(v in c("customer_id","invoice_id","stock_id")) {
  bin_feat <- matrix(
    as.integer(intToBits(as.integer(as.factor(full[[v]])))),
    ncol = 32,
    nrow = length(full[[v]]),
    byrow = TRUE
  )[, 1:ceiling(log(length(unique(full[[v]])) + 1)/log(2))]
  bin_feat <- as.data.frame(bin_feat)
  names(bin_feat) <- paste0(v, "_bin_", c(1:ncol(bin_feat)))
  full <- cbind(full, bin_feat)
}

# Process Description column ----------------------------------------------

cat("Parsing description...\n")
it <- full %$%
  str_to_lower(description) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems) # text2vec's iterator
vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%   # counts words, etc.
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5, vocab_term_max = 3884) %>%  # cleaning function. 3884 = unique descriptions
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)  # text2vec's Term Frequency Inverse Document Frequency model
tfidf <- create_dtm(it, vectorizer) %>%  # create vector file (nrow x 3884max)
  fit_transform(m_tfidf)

m_lsa <- LSA$new(n_topics = 25, method = "randomized")  # text2vec's latent semantic analysis, with 25 topics
lsa <- fit_transform(tfidf, m_lsa) # creates a nrow x 25 topics matrix
dimnames(lsa)[[2]] <- paste0("lsa",1:dim(lsa)[2])
full <- full %>% cbind(lsa)  # attach LSA to full

# Explore as classification -----------------------------------------------

length(unique(train$quantity))
table(train$quantity)
quantity_log <- log1p(sample(train$quantity, 10000))
plot(quantity_log)
plot(quantity_log[quantity_log < 6])
plot(quantity_log[quantity_log < 3.5])
table(quantity_log[quantity_log < 3.5])
prop.table(table(quantity_log[quantity_log < 3.5])) * 100
quant_freq_tab <- sort(prop.table(table(train$quantity)) * 100, decreasing = T)
sum(quant_freq_tab[1:20])
# [1] 96.677
# Top 20 quantities account for 97% of the data
top20quantity <- as.numeric(names(quant_freq_tab[1:20]))
sort(top20quantity)

train$quantity_class_21 <- ifelse(train$quantity %in% top20quantity, train$quantity, 0)


#---------------------------
cat("Preparing data for model\n")

# missing data
colSums(is.na(full))

# replace NAs with 0s
for(v in colnames(full)[-2]) {full[[v]][is.na(full[[v]])] <- 0}

full_sparse <- full %>%
  select(-c(id, date, time, description, country)) %>%
  sparse.model.matrix(~ . - 1, .)

rm(tfidf, m_tfidf, it, m_lsa, vectorizer, bin_feat, v, quant_freq_tab, quantity_log); gc()
#rm(tfidf, m_tfidf, it, lsa, m_lsa, vectorizer); gc()


# Regression -----------------------------------------------------------------
cat("Training & predicting...\n")

varnames <- setdiff(colnames(full_sparse), names(full)[grepl("quantity", names(full))])
#varnames <- setdiff(colnames(full_sparse), c("quantity", "quantity_log"))
train_sparse <- full_sparse[full$quantity_log != 0, varnames]
y_train  = full$quantity_log[full$quantity_log != 0]

# y_train_class <- as.numeric(as.factor(train$target_class)) - 1  # xgboost classification label
# # limits training to top 20 quantities
# train_sparse <- train_sparse[train$target_class != 0, ]
# y_train_class <- as.numeric(as.factor(train$target_class)) - 1  # xgboost classification label
# #y_train_class <- full$target_class[full$quantity_log != 0]

lgb.train = lgb.Dataset(data=train_sparse, label=y_train)
categoricals.vec = c("invoice_id","stock_id","customer_id","countrynum")

# Setting up LGBM Parameters
lgb.grid = list(objective = "regression"
                , metric = "rmse"
)

# Cross Validation
lgb.model.cv = lgb.cv(params = lgb.grid
                      , data = lgb.train
                      , nrounds = 300
                      , early_stopping_rounds = 25
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
                      , nfold = 10
                      , stratified = TRUE
                      )

best.iter <- lgb.model.cv$best_iter
best.iter
lgb.model.cv$record_evals$valid$rmse$eval[[best.iter]]

eval <- unlist(lgb.model.cv$record_evals$valid$rmse$eval)
eval_err <- unlist(lgb.model.cv$record_evals$valid$rmse$eval_err)
plot(eval)
plot(eval_err)
min_err <- which.min(eval_err)
eval[min_err]
eval_err[min_err]

# Train final regression model
lgb.model = lgb.train(params = lgb.grid
                      , data = lgb.train
                      , nrounds = best.iter
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
)

tree_imp <- lgb.importance(lgb.model)
lgb.plot.importance(tree_imp, top_n = 25, measure = "Gain")
tree_imp %>% arrange(desc(Gain))

# make predictions
pred <- predict(lgb.model, full_sparse)
#full$quantity_log_pred_reg <- as.vector(pred)
full$quantity_pred_reg <- as.vector(expm1(pred))  # attach in original metric
#dimnames(full_sparse)[[2]][dim(full_sparse)[2]] <- "quantity_pred_reg"  # rename last column

# # recreate full_sparse with new quantity_pred_reg column
# full_sparse <- full %>%
#   select(-c(id, date, time, description, country)) %>%
#   sparse.model.matrix(~ . - 1, .)

# look at stratification of data
train_pred_reg <- full$quantity_pred_reg[1:371893]

par(mfrow=c(1,2))
plot(train$quantity, ylim = range(train$quantity))
plot(train_pred_reg, ylim = range(train$quantity))

plot(train$quantity[train$quantity < 100], ylim = c(0,100))
plot(train_pred_reg[train_pred_reg < 100], ylim = c(0,100))

par(mfrow=c(1,1))


# for classification labels, should we do: 1) top 20 VS everything else = 21
# labels, or 2) all quantities < 100 = 99? Let's try both?
length(unique(train$quantity[train$quantity <= 100]))
# create quantity_class_99 column
train$quantity_class_99 <- ifelse(train$quantity <= 100, train$quantity, NA)

# # compare plots
# plot(train$quantity)
# plot(full$quantity_pred_reg[full$quantity_log != 0])

# setup LGBM
#varnames <- setdiff(colnames(full_sparse), names(full)[grepl("quantity", names(full))])

# update varnames
#varnames <- varnames[-grep("bin|country_", varnames)]

# subset training data for classification (quantity <= 100)
train_sparse <- full_sparse[full$quantity_log != 0 & full$quantity <= 100, varnames]
y_train_21  = train$quantity_class_21[train$quantity <= 100]
y_train_99  = train$quantity_class_99[train$quantity <= 100]

# create lookup tables for classification
y_train_21.lookup <- data.frame(quantity = sort(unique(y_train_21)), label = (1:length(unique(y_train_21))-1))
y_train_99.lookup <- data.frame(quantity = sort(unique(y_train_99)), label = (1:length(unique(y_train_99))-1))

y_train_21  = as.data.frame(y_train_21) %>%
  left_join(y_train_21.lookup, by = c("y_train_21" = "quantity"))
y_train_99  = as.data.frame(y_train_99) %>%
  left_join(y_train_99.lookup, by = c("y_train_99" = "quantity"))

# lightGBM classification -------------------------------------------------

# Cross Validation
lgb.model.cv = lgb.cv(data = lgb.Dataset(data=train_sparse, label=y_train_21$label)
                      , nrounds = 2000
                      , early_stopping_rounds = 25
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
                      , nfold = 5
#                      , stratified = TRUE
                      , params = list(objective = "multiclass"
                                      , num_class = 21
                                      )
                      )

lgb.model.cv.21 <- lgb.model.cv

best.iter <- lgb.model.cv$best_iter
best.iter
lgb.model.cv$record_evals$valid$multi_logloss$eval[[best.iter]]

eval <- unlist(lgb.model.cv$record_evals$valid$multi_logloss$eval)
eval_err <- unlist(lgb.model.cv$record_evals$valid$multi_logloss$eval_err)
plot(eval)
plot(eval_err)

# Train final classification model
lgb.model = lgb.train(params = list(objective = "multiclass"
                                    , num_class = 21
                                    )
                      , data = lgb.Dataset(data=train_sparse, label=y_train_21$label)
                      , nrounds = best.iter
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
)

tree_imp <- lgb.importance(lgb.model)

# make predictions
pred <- predict(lgb.model, full_sparse, reshape = TRUE)
pred <- as.data.frame(pred)
names(pred) <- y_train_21.lookup$quantity
pred_quantity_multiclass <- colnames(pred)[max.col(pred,ties.method="random")]
full$quantity_pred_class21 <- as.numeric(pred_quantity_multiclass)

full$quantity_pred_ens <- full$quantity_pred_reg
full$quantity_pred_ens <- if (full$quantity_pred_class21 %in% y_train_21.lookup$quantity[-1]) {
  
}


full$quantity_pred_ens <- ifelse(full$quantity_pred_class21 %in% y_train_21.lookup$quantity[-1]
                                 , full$quantity_pred_class21
                                 , full$quantity_pred_reg
                                 )

# plot
par(mfrow=c(1,2))
plot(train$quantity[train$quantity < 100])
plot(full$quantity_pred_ens[full$quantity != 0 & full$quantity_pred_ens < 100])
par(mfrow=c(1,1))

par(mfrow=c(1,2))
plot(train$quantity[train$quantity < 100])
plot(full$quantity_pred_reg[train$quantity < 100])
par(mfrow=c(1,1))



# calculate rmsle
library(Metrics)
rmsle(actual, predicted)










# need to create a look up table for 

# setup xgb matrix
xgb_train <- xgb.DMatrix(data=train_sparse, label=y_train_class)
xgb_test  <- xgb.DMatrix(data=test_sparse)

# m_xgb <- xgboost(data = xgb_train
#                  , params = list(objective = "multi:softmax"
#                                  , eval_metric = "mlogloss"
#                                  , num_class = 20
#                                  , eta = 0.2
#                  )
#                  , print_every_n = 100
#                  , nrounds = 500
# )

# cross validation
xgb.model.cv <- xgb.cv(data = xgb_train
                       , params = list(objective = "multi:softmax"
                                       , eval_metric = "mlogloss"
                                       , num_class = 21
                                       , eta = 0.2
                                       )
                       , nrounds = 500
                       , nfold = 10
                       , print_every_n = 20
                       , early_stopping_rounds = 25
                       )







# Light GBM ---------------------------------------------------------------

#maybe no need for unit_price_log, since it's a decision tree and works on splits
#compare model with summarize statistics to binary coding. or just do binary
#coding for customer_id

# Create LGB Dataset



varnames = setdiff(colnames(full_sparse), "quantity_log")
# varnames = setdiff(colnames(full_sparse)
#                    , c("quantity_log"
#                    , "inv_tot_price", "inv_tot_quant", "inv_n_stockid"
#                    , "stock_mean_quant", "stock_median_quant", "stock_iqr_quant"
#                    , "cust_mean_quant", "cust_median_quant", "cust_iqr_quant"
#                    ))

train_sparse = full_sparse[full$quantity_log != 0, varnames]
test_sparse  = full_sparse[full$quantity_log == 0, varnames]

y_train  = full$quantity_log[full$quantity_log != 0]
#test_ids = data[is.na(target) ,id]

lgb.train = lgb.Dataset(data=train_sparse, label=y_train)

#categoricals.vec = colnames(train)[c(grep("cat",colnames(train)))]
categoricals.vec = c("invoice_id","stock_id","customer_id","countrynum")

# Setting up LGBM Parameters
lgb.grid = list(objective = "regression"
                , metric = "rmse"
                # , min_sum_hessian_in_leaf = 1
                # , feature_fraction = 0.7
                # , bagging_fraction = 0.7
                # , bagging_freq = 5
                # , min_data = 100
                # , max_bin = 50
                # , lambda_l1 = 8
                # , lambda_l2 = 1.3
                # , min_data_in_bin = 100
                # , min_gain_to_split = 10
                # , min_data_in_leaf = 30
                # , is_unbalance = TRUE
)

# # Setting up Gini Eval Function
# # Gini for Lgb
# lgb.normalizedgini = function(preds, dtrain){
#   actual = getinfo(dtrain, "label")
#   score  = NormalizedGini(preds,actual)
#   return(list(name = "gini", value = score, higher_better = TRUE))
# }

# Cross Validation
lgb.model.cv = lgb.cv(params = lgb.grid
                      , data = lgb.train
                      , learning_rate = 0.02
                      , num_leaves = 25
                      , nrounds = 500
                      , early_stopping_rounds = 25
                      , eval_freq = 10
                      # , eval = lgb.normalizedgini
                      , categorical_feature = categoricals.vec
                      , nfold = 5
                      , stratified = TRUE
)

best.iter <- lgb.model.cv$best_iter
best.iter
lgb.model.cv$record_evals$valid$rmse$eval[[best.iter]]

# Train final model
lgb.model = lgb.train(params = lgb.grid
                      , data = lgb.train
                      , learning_rate = 0.02
                      , num_leaves = 25
                      , nrounds = best.iter
                      , eval_freq = 20
                      # , eval = lgb.normalizedgini
                      , categorical_feature = categoricals.vec
)

tree_imp <- lgb.importance(lgb.model)
lgb.plot.importance(tree_imp, top_n = 25, measure = "Gain")
tree_imp %>% arrange(desc(Gain))



#tree_imp_ind <- str_replace_all(tree_imp$Feature, "Column_", "")
#tree_imp_ind <- as.numeric(tree_imp_ind)
#dimnames(train_sparse)[[2]][tree_imp_ind]

# make predictions
pred <- predict(lgb.model,test_sparse)
subm$quantity <- as.vector(expm1(pred))
write.csv(subm, "lgb_humanML.csv", row.names=FALSE)