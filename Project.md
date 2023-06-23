Prediction project
================
2023-06-22

## Getting data to R

Downloading csv files from URL links:

``` r
tr <- read.csv(url(
  "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),
  na.strings=c("NA","#DIV/0!",""))

te <- read.csv(url(
  "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),
  na.strings=c("NA","#DIV/0!",""))
```

Cross-table for the dependent variable:

``` r
table(tr$classe)
```

    ## 
    ##    A    B    C    D    E 
    ## 5580 3797 3422 3216 3607

## Data processing

Split the training data randomly into 80% of training and 20% of
validation data:

``` r
set.seed(6) #ensures reproducibility
library(caret)
train <- createDataPartition(tr$classe, p = 0.8, list = FALSE)
training <- tr[train, ]
validation <- tr[-train, ]
```

Removing first seven descriptive columns:

``` r
training <- training[,-c(1:7)]
validation <- validation[,-c(1:7)]
```

Identifying and removing predictors (features) in the training dataset
that have little or no variance to improve modelling:

``` r
nonzero_columns <- nearZeroVar(training)
training <- training[, -nonzero_columns]
```

Removing columns with 35% or more NAs (missing values)

``` r
countlength <- sapply(training, function(x) {
  sum(!(is.na(x) | x == ""))
})

drop_columns <- names(countlength[countlength < 0.75 * length(training$classe)])

training <- training[, !names(training) %in% drop_columns]
```

## Training the model and validation

Applying random forest modelling due to its known good performance in
making predictions (boosting was not used due to its long computational
time):

``` r
library(randomForest)
```

``` r
modelRF <- randomForest(as.factor(classe)~ ., data = training, importance = T, ntrees = 10)
```

Assessing model accuracy and error within the training dataset:

``` r
training_prediction <- predict(modelRF, training)

#make sure that the dependent variable is here as a factor!
#..otherwise it will give an error about levels
confusionMatrix(training_prediction, as.factor(training$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 4464    0    0    0    0
    ##          B    0 3038    0    0    0
    ##          C    0    0 2738    0    0
    ##          D    0    0    0 2573    0
    ##          E    0    0    0    0 2886
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9998, 1)
    ##     No Information Rate : 0.2843     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##                                      
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
    ## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

The model has 99.9% accuracy and thus about 0.01% error.

Out of sample validation:

``` r
validation_prediction  <- predict(modelRF, validation)

#make sure that the dependent variable is here as a factor!
#..otherwise it will give an error about levels
confusionMatrix(validation_prediction, as.factor(validation$classe))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1116    1    0    0    0
    ##          B    0  757    2    0    0
    ##          C    0    1  682    9    0
    ##          D    0    0    0  634    4
    ##          E    0    0    0    0  717
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9957          
    ##                  95% CI : (0.9931, 0.9975)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9945          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9974   0.9971   0.9860   0.9945
    ## Specificity            0.9996   0.9994   0.9969   0.9988   1.0000
    ## Pos Pred Value         0.9991   0.9974   0.9855   0.9937   1.0000
    ## Neg Pred Value         1.0000   0.9994   0.9994   0.9973   0.9988
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2845   0.1930   0.1738   0.1616   0.1828
    ## Detection Prevalence   0.2847   0.1935   0.1764   0.1626   0.1828
    ## Balanced Accuracy      0.9998   0.9984   0.9970   0.9924   0.9972

Validation reveals 99.6% accuracy and thus about 0.04% out of sample
error. The model seems to be rather adequate in predicting the classe
variable.

## Model prediction based on the test data

``` r
test_prediction <- predict(modelRF, te)
test_prediction
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
