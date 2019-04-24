# Dataset

- Trainset - [**here**](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/download/train-file)
- Testset - [**here**](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018-1/download/test-file)

# Objective

The objective is the find from different attributes of an employee, who is the right person for promotion. There is a substantial delay in the process of promoting people because most of it is being wasted in evaluating employees. If using Machine Learnig, we can speed up the process of finding the right employee for promotion then the transition becomes much faster and smoother.



# Procedure

I trained 2 models.

1. LightGBM model finetuned (OneHotEncoded features)
2. XGBoost model finetuned (Ploynomial Features)

# Results

My Best Score - 0.522, rank 60 on leaderboard

Best Score - 0.53+

# Assumptions

Polynomial features are linearly seperable

# Reflections

Need to try out stacking of different ensemle models

