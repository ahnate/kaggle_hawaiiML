
# Load libraries and import data ------------------------------------------

library(tidyverse)
library(lubridate) # dates
library(text2vec)
library(tokenizers)
library(Matrix) # create sparse matrices for models
library(lightgbm)
options(scipen=999) # turn off scientific notation (for output to kaggle)

train <- read_csv("./input/hawaiiml-data/train.csv")
test <- read_csv("./input/hawaiiml-data/test.csv")
subm <- read_csv("./input/hawaiiml-data/sample_submission.csv") 


# Preprocessing -----------------------------------------------------------

# remove outliers (https://www.kaggle.com/ahnate/exploration-word-cloud)
train <- train[train$quantity < 60000,]
train <- train[(train$unit_price >= 0) & (train$unit_price < 5000),]
train <-(train[!(train$unit_price > 500 & train$quantity > 50),])

# combine test and train data
full <- bind_rows(train, test)

# log transform quantity
full$quantity_log <- log1p(full$quantity)

# format dates
full$date <- mdy(full$date) 
full$dayofweek <- wday(full$date)  # numerical
full$week  <- week(full$date)
full$month <- month(full$date)
full$year  <- year(full$date)
full$minuteinday <- as.numeric(full$time)/60   # time in minute, 00:00 as origin


# Feature engineering -----------------------------------------------------

# Summary statistics are used as alternate representations for customer_id,
# invoice_id, and stock_id, providing extra characteristics beyond labels.
# https://stats.stackexchange.com/questions/49243/rs-randomforest-can-not-handle-more-than-32-levels-what-is-workaround
# 1st answer

# Summarize quantity by each customer_id x invoice_id combo
train_cust_inv <- train %>%
  select(customer_id, invoice_id, unit_price, quantity) %>%
  mutate(tot_price = unit_price * quantity) %>%
  group_by(customer_id, invoice_id) %>%
  summarize(inv_tot_price = sum(tot_price)
            , inv_tot_quant = sum(quantity), inv_n_stockid = n())

# Summarize quantity by stock_id
train_stock <- train %>%
  select(stock_id, quantity) %>%
  group_by(stock_id) %>%
  summarize(stock_mean_quant = mean(quantity), stock_median_quant = median(quantity), 
            stock_iqr_quant = IQR(quantity))  # IQR = better dispersion metric for skewed data

# Summarize quantity by by customer_id
train_cust <- train %>%
  select(customer_id, quantity) %>%
  group_by(customer_id) %>%
  summarize(cust_mean_quant = mean(quantity), cust_median_quant = median(quantity), 
            cust_iqr_quant = IQR(quantity))

# Summarize quantity by country
train_country <- train %>%
  select(country, quantity) %>%
  group_by(country) %>%
  summarize(country_mean_quant = mean(quantity), country_median_quant = median(quantity),
            country_iqr_quant = IQR(quantity))

# Join summary statistics to train and test data
full <- full %>% 
  left_join(train_cust_inv, by = c("customer_id", "invoice_id")) %>%
  left_join(train_stock, by = c("stock_id")) %>%
  left_join(train_cust, by = c("customer_id")) %>%
  left_join(train_country, by = c("country"))


# Categorical encoding ----------------------------------------------------

# https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

# Encode country as numerical since there's only 38 countries
full$countrynum <- as.numeric(as.factor(full$country))

# Binary encoding
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

# Creates 1) a Term Frequency Inverse Document Frequency object (counts how
# often words occurs in description), and 2) 25 Latent Semantic Analysis
# columns (likelihood of descriptions matching each semantic topic). Due to the
# large size of the tfidf object (~1770 columns), I only used the LSA, although
# the tfidf does help prediction.

it <- full %$%
  str_to_lower(description) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems) # text2vec's iterator
vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%   # counts words, etc.
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.5) %>%  # cleaning function
  vocab_vectorizer()

m_tfidf <- TfIdf$new(norm = "l2", sublinear_tf = T)  # text2vec's Term Frequency Inverse Document Frequency model
tfidf <- create_dtm(it, vectorizer) %>%  # create vector file (nrow x 3884max)
  fit_transform(m_tfidf)

m_lsa <- LSA$new(n_topics = 25, method = "randomized")  # text2vec's latent semantic analysis, 25 topics
lsa <- fit_transform(tfidf, m_lsa) # creates a nrow x 25 topics matrix
dimnames(lsa)[[2]] <- paste0("lsa",1:dim(lsa)[2])  # LSA column names
full <- full %>% cbind(lsa)  # attach LSA to train and test data


# Prep data for model -----------------------------------------------------

# Many models won't run on missing data, so handling the NAs is required. Many
# also require sparse matrices as inputs (as opposed to regular dataframes).

colSums(is.na(full))  # show missing data
for(v in colnames(full)[-2]) {full[[v]][is.na(full[[v]])] <- 0}  # replace NAs with 0s

# Convert to sparse matrix
full_sparse <- full %>%
  select(-c(id, date, time, description, quantity, country)) %>%
  sparse.model.matrix(~ . - 1, .)

rm(m_tfidf, tfidf, it, m_lsa, vectorizer, bin_feat, v); gc()  # garbage cleanup

varnames = setdiff(colnames(full_sparse), "quantity_log")  # Select predictors
train_sparse = full_sparse[full$quantity_log != 0, varnames]  # Sparse version of train
test_sparse  = full_sparse[full$quantity_log == 0, varnames]  # and test
y_train  = full$quantity_log[full$quantity_log != 0]  # Training labels

# lightGBM prep
lgb.train = lgb.Dataset(data=train_sparse, label=y_train)
categoricals.vec = c("invoice_id","stock_id","customer_id","countrynum")



# Run lightGBM cross validation -------------------------------------------

lgb.model.cv = lgb.cv(data = lgb.train
                      , nrounds = 5000
                      , early_stopping_rounds = 25
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
                      , nfold = 10
                      , stratified = TRUE
                      , params = list(objective = "regression"
                                      , metric = "rmse"
                                      )
                      )

# Best cv iteration and corresponding rmse
best.iter <- lgb.model.cv$best_iter
best.iter
lgb.model.cv$record_evals$valid$rmse$eval[[best.iter]]

# Plot cv rmse and error
eval <- unlist(lgb.model.cv$record_evals$valid$rmse$eval)
eval_err <- unlist(lgb.model.cv$record_evals$valid$rmse$eval_err)
plot(eval)
plot(eval_err)


# Final lightGBM model ----------------------------------------------------

# I used 7000 for personal time constraints. Increasing iterations should yield
# lower RMSEs, although at this point I think it's mostly diminishing returns.
lgb.model = lgb.train(data = lgb.train
                      , nrounds = 7000
                      , eval_freq = 10
                      , categorical_feature = categoricals.vec
                      , params = list(objective = "regression"
                                      , metric = "rmse"
                                      )
                      )

# Feature importance
tree_imp <- lgb.importance(lgb.model)
lgb.plot.importance(tree_imp, top_n = 25, measure = "Gain")
tree_imp %>% arrange(desc(Gain))


# Create predictions ------------------------------------------------------

pred <- predict(lgb.model,test_sparse)
subm$quantity <- as.vector(expm1(pred))  # Convert back to original metric
write.csv(subm, "lgb_7000_87feats_04_30_1339.csv", row.names=FALSE)  # Write to file
