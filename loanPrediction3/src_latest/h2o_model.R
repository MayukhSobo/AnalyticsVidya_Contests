# Loading the libraries
suppressMessages(library(h2o))
suppressMessages(library(dplyr))

# Initialize H2o cluster in localhost
localH2o <- h2o.init(nthreads = -1)

# Don't show the progress bars
h2o.no_progress()

# Loading the datasets 
loan_train <- read.csv('train.csv', header = T)
loan_test <- read.csv('test.csv', header = T)

# Combining the test and train datasets
loan_test["Loan_Status"] <- "Y"
loan_data <- rbind(loan_train, loan_test)
rm(loan_test, loan_train)

# Cleaning the dataset and handling the outliers
loan_data$LoanAmount[loan_data$LoanAmount >= 400] <- 400
loan_data$LoanAmount[which(is.na(loan_data$LoanAmount))] <- median(loan_data$LoanAmount, na.rm=T)
loan_data$Loan_Amount_Term[which(is.na(loan_data$Loan_Amount_Term))] <- 360
loan_data$Married[loan_data$Married == ""] <- "Yes"
loan_data$Married<-factor(loan_data$Married)
loan_data$Credit_History <- as.factor(loan_data$Credit_History)

for (i in 1:dim(loan_data)[1]) {
    if (is.na(loan_data$Credit_History[i])) {
        if (loan_data$Loan_Amount_Term[i] >= 360) {
            loan_data$Credit_History[i] <- 1
        }else{
            loan_data$Credit_History[i] <- 0
        }
    }
}

# Feature Engiineering
loan_data$Total_Income <- loan_data$ApplicantIncome + loan_data$CoapplicantIncome
loan_data$Loan_per_month <- loan_data$LoanAmount / loan_data$Loan_Amount_Term


# Preparing the cleaned data
loan_data <- loan_data[, c(1:12, 14, 15, 13)]
loan_train <- loan_data[1:614,]
loan_test <- loan_data[615:981,]
loan_test$Loan_Status <- NULL
train.hex <- as.h2o(loan_train, destination_frame = "train.hex")
test.hex <- as.h2o(loan_test, destination_frame = "test.hex")

# Fitting the GLM model

binomial.fit <- h2o.glm(
    y = "Loan_Status", 
    x = c("Credit_History","Property_Area", "Married", "Total_Income","Loan_per_month"), 
    training_frame = train.hex, 
    family = "binomial", 
    nfolds=4, 
    lambda_search = T, 
    seed = 0xDECAF
    )
print(binomial.fit)

# Prediction and Submission
pred = h2o.predict(binomial.fit, test.hex)
Loan_Status <- as.data.frame(pred)[,1]
submission <- cbind(loan_test, Loan_Status)
submission <- submission %>%
    select(c(Loan_ID, Loan_Status))
rownames(submission) <- 1:nrow(submission)
write.csv(submission,file = "submission6.csv", row.names = FALSE)