---
title: "Final Group project"
author: "Paola Pavon, Maria Meneses, Yesenia Silva, Hauwa Umaru"
date: "`r Sys.Date()`"
output:
  html_document:
    theme: flatly
    highlight: zenburn
    number_sections: yes
    toc: yes
    toc_float: yes
    code_folding: show
  pdf_document:
    toc: yes
---


```{r}
#| label: load-libraries
#| echo: false # This option disables the printing of code (only output is displayed).
#| message: false
#| warning: false

library(tidyverse)
library(tidymodels)
library(skimr)
library(kknn)
library(here)
library(tictoc)
library(vip)
library(ranger)
library(tidygeocoder)
```

# The problem: predicting credit card fraud

The goal of the project is to predict fraudulent credit card transactions.

We will be using a dataset with credit card transactions containing legitimate and fraud transactions. Fraud is typically well below 1% of all transactions, so a naive model that predicts that all transactions are legitimate and not fraudulent would have an accuracy of well over 99%-- pretty good, no? 

You can read more on credit card fraud on [Credit Card Fraud Detection Using Weighted Support Vector Machine](https://www.scirp.org/journal/paperinformation.aspx?paperid=105944)

The dataset we will use consists of credit card transactions and it includes information about each transaction including customer details, the merchant and category of purchase, and whether or not the transaction was a fraud.

## Obtain the data

The dataset is too large to be hosted on Canvas or Github, so please download it from dropbox https://www.dropbox.com/sh/q1yk8mmnbbrzavl/AAAxzRtIhag9Nc_hODafGV2ka?dl=0 and save it in your `dsb` repo, under the `data` folder.

As we will be building a classifier model using tidymodels, there's two things we need to do:

1. Define the outcome variable `is_fraud` as a factor, or categorical, variable, instead of the numerical 0-1 varaibles.
2. In tidymodels, the first level is the event of interest. If we leave our data as is, `0` is the first level, but we want to find out when we actually did (`1`) have a fraudulent transaction

```{r}
#| echo: false
#| message: false
#| warning: false

card_fraud <- read_csv(here::here("data", "card_fraud.csv")) %>% 

  mutate(
    # in tidymodels, outcome should be a factor  
    is_fraud = factor(is_fraud),
    
    # first level is the event in tidymodels, so we need to reorder
    is_fraud = relevel(is_fraud, ref = "1")
         )

glimpse(card_fraud)
```

The data dictionary is as follows

| column(variable)      | description                                 |
|-----------------------|---------------------------------------------|
| trans_date_trans_time | Transaction DateTime                        |
| trans_year            | Transaction year                            |
| category              | category of merchant                        |
| amt                   | amount of transaction                       |
| city                  | City of card holder                         |
| state                 | State of card holder                        |
| lat                   | Latitude location of purchase               |
| long                  | Longitude location of purchase              |
| city_pop              | card holder's city population               |
| job                   | job of card holder                          |
| dob                   | date of birth of card holder                |
| merch_lat             | Latitude Location of Merchant               |
| merch_long            | Longitude Location of Merchant              |
| is_fraud              | Whether Transaction is Fraud (1) or Not (0) |

We also add some of the variables we considered in our EDA for this dataset during homework 2.

```{r}
card_fraud <- card_fraud %>% 
  mutate( hour = hour(trans_date_trans_time),
          wday = wday(trans_date_trans_time, label = TRUE),
          month_name = month(trans_date_trans_time, label = TRUE),
          age = interval(dob, trans_date_trans_time) / years(1)
) %>% 
  rename(year = trans_year) %>% 
  
  mutate(
    
    # convert latitude/longitude to radians
    lat1_radians = lat / 57.29577951,
    lat2_radians = merch_lat / 57.29577951,
    long1_radians = long / 57.29577951,
    long2_radians = merch_long / 57.29577951,
    
    # calculate distance in miles
    distance_miles = 3963.0 * acos((sin(lat1_radians) * sin(lat2_radians)) + cos(lat1_radians) * cos(lat2_radians) * cos(long2_radians - long1_radians)),

    # calculate distance in km
    distance_km = 6377.830272 * acos((sin(lat1_radians) * sin(lat2_radians)) + cos(lat1_radians) * cos(lat2_radians) * cos(long2_radians - long1_radians))

  )

```

## Exploratory Data Analysis (EDA) 

You have done some EDA and you can pool together your group's expertise in which variables to use as features.
You can reuse your EDA from earlier, but we expect at least a few visualisations and/or tables to explore teh dataset and identify any useful features.


Group all variables by type and examine each variable class by class. The dataset has the following types of variables:

1.  Strings
    category
    city
    state
    job
    
    Factors:
        is_fraud
    
        
    Ordered:
        wday
        month_name


2.  Geospatial Data
    lat
    long
    merch_lat
    merch_long
    
    lat1_radians
    lat2_radians
    long1_radians
    long2_radians

3.  Dates
    dob
    
4.  Date/Times
    trans_date_trans_time

5.  Numerical
    year
    amt
    city_pop
    hour
    age
    distance_miles
    distance_km


Strings are usually not a useful format for classification problems. The strings should be converted to factors, dropped, or otherwise transformed.

***Strings to Factors*** 

-   `category`, Category of Merchant
-   `job`, Job of Credit Card Holder

***Strings to Geospatial Data*** 

We have plenty of geospatial data as lat/long pairs, so I want to convert city/state to lat/long so I can compare to the other geospatial variables. This will also make it easier to compute new variables like the distance the transaction is from the home location. 

-   `city`, City of Credit Card Holder
-   `state`, State of Credit Card Holder

##  Exploring factors: how is the compactness of categories?

-   Do we have excessive number of categories? Do we want to combine some?

```{r}
card_fraud %>% 
  count(category, sort=TRUE)%>% 
  mutate(perc = n/sum(n))

card_fraud %>% 
  count(job, sort=TRUE) %>% 
  mutate(perc = n/sum(n))

```


The predictors `category` and `job` are transformed into factors.

```{r}
#| label: convert-strings-to-factors


card_fraud <- card_fraud %>% 
  mutate(category = factor(category),
         job = factor(job))
```

`category` has 14 unique values, and `job` has 494 unique values. The dataset is quite large, with over 670K records, so these variables don't have an excessive number of levels at first glance. However, it is worth seeing if we can compact the levels to a smaller number.

### Why do we care about the number of categories and whether they are "excessive"?

Consider the extreme case where a dataset had categories that only contained one record each. There is simply insufficient data to make correct predictions using category as a predictor on new data with that category label. Additionally, if your modeling uses dummy variables, having an extremely large number of categories will lead to the production of a huge number of predictors, which can slow down the fitting. This is fine if all the predictors are useful, but if they aren't useful (as in the case of having only one record for a category), trimming them will improve the speed and quality of the data fitting.

If I had subject matter expertise, I could manually combine categories. If you don't have subject matter expertise, or if performing this task would be too labor intensive, then you can use cutoffs based on the amount of data in a category. If the majority of the data exists in only a few categories, then it might be reasonable to keep those categories and lump everything else in an "other" category or perhaps even drop the data points in smaller categories. 


## Do all variables have sensible types?

Consider each variable and decide whether to keep, transform, or drop it. This is a mixture of Exploratory Data Analysis and Feature Engineering, but it's helpful to do some simple feature engineering as you explore the data. In this project, we have all data to begin with, so any transformations will be performed on the entire dataset. Ideally, do the transformations as a `recipe_step()` in the tidymodels framework. Then the transformations would be applied to any data the recipe was used on as part of the modeling workflow. There is less chance of data leakage or missing a step when you perform the feature engineering in the recipe.

## STRINGS

__Exploring category__
```{r}

# Category
card_fraud %>%
  count(category) %>%
  ggplot(aes(x = reorder(category, n), y = n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Count by Category", x = "Category", y = "Count")

# Amt by Category
ggplot(card_fraud, aes(x = category, y = amt)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Amount by Category", x = "Category", y = "Amount") +
  coord_flip()

# Category by is_fraud
card_fraud %>%
  group_by(category, is_fraud) %>%
  summarise(count = n(), .groups = "drop") %>%
  mutate(freq = count / sum(count) * 100) %>%
  filter(is_fraud == 1) %>%
  arrange(desc(freq)) %>%
  ggplot(aes(x = reorder(category, freq), y = freq)) +
  geom_bar(stat = "identity", fill = "#e74c3c", alpha = 0.8) +
  geom_text(aes(label = round(freq, 1)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(
    title = "Fraud Is Concentrated in a Few Spending Categories",
    subtitle = "Relative share of fraudulent transactions by purchase category (2019-2020)",
    x = NULL,
    y = "Percentage (%)"
  ) +
  theme_minimal() +
  theme(
    plot.title     = element_text(size = 14, face = "bold"),
    axis.title.x   = element_text(size = 10),
    axis.title.y   = element_text(size = 10),
    axis.text      = element_text(size = 8),
    panel.grid.minor = element_blank()
  )

```
__CONCLUSION:__ 
The category variable displays a clear imbalance in transaction counts and fraudulent cases across merchant types, with some categories being much more frequent than others. This suggests that merchant category is a crucial feature, as certain types are more likely to be targeted for fraud. 

__Exploring city__
City is a categorical variable with 894 unique values. It is not practical to use city as a predictor in a model, so we will drop it.

```{r}
#---  Fraudulent transactions rate by city   ---#
card_fraud %>%
  group_by(city) %>%
  summarise(
    n_fraud = sum(is_fraud == "1"),
    n_total = n(),
    fraud_rate = n_fraud / n_total
  )  %>%
  arrange(desc(fraud_rate))
```

_CONCLUSION:_ city is NOT an useful variable to predict fraud.

__Exploring state__
State is a categorical variable with 51 unique values. It is not practical to use state as a predictor in a model, so we will drop it.

```{r}
#---  Fraudulent transactions rate by state   ---#
card_fraud %>%
  group_by(state) %>%
  summarise(
    n_fraud = sum(is_fraud == "1"),
    n_total = n(),
    fraud_rate = n_fraud / n_total
  )  %>%
  arrange(desc(fraud_rate))
```

_CONCLUSION:_ state is NOT an useful variable to predict fraud.

__Exploring job__ 
The job variable has 494 unique values, which is a lot for a machine learning model. The most common job appears in only 0.5% of the data, and the least common in just 0.0001%.
To simplify, we used ChatGPT to group jobs into 20 broader categories like "Professional", "Sales", or "Service". However, the histogram shows that most jobs ended up in the "Other" group. Since our goal is to predict fraud transactions, and this variable would offer little interpretability if the "Other" category dominates, it’s better to drop it from the model.

__Conclusion__
We drop the variable.

```{r}
card_fraud %>% 
  count(job, sort = TRUE) %>% 
  mutate(perc = n * 100 / sum(n)) %>% 
  arrange(desc(perc)) 

card_fraud %>% 
  count(job, sort = TRUE) %>% 
  mutate(perc = n * 100 / sum(n)) %>% 
  summarise(
    min = min(perc),
    max = max(perc),
    mean = mean(perc),
    median = median(perc),
    sd = sd(perc),
    q1 = quantile(perc, 0.25),
    q3 = quantile(perc, 0.75)
  )

# Classification done by chat gpt 
unique(card_fraud$job)

classified_jobs <- card_fraud %>%
  mutate(job_category_20 = case_when(
    str_detect(job, regex("engineer|engineering|architect|surveyor|planner|technologist", TRUE)) ~ "Engineering & Architecture",
    str_detect(job, regex("doctor|surgeon|oncologist|physician|pathologist", TRUE)) ~ "Medicine",
    str_detect(job, regex("nurse|midwife|paramedic", TRUE)) ~ "Nursing & Emergency Care",
    str_detect(job, regex("therapist|psychologist|counsellor|psychiatrist|psychotherapist", TRUE)) ~ "Mental Health",
    str_detect(job, regex("teacher|lecturer|professor|education|instructor|trainer", TRUE)) ~ "Education",
    str_detect(job, regex("scientist|researcher|biochemist|chemist|geologist|physicist", TRUE)) ~ "Scientific Research",
    str_detect(job, regex("it|developer|programmer|software|data scientist|web|cyber|applications", TRUE)) ~ "Technology & Data",
    str_detect(job, regex("accountant|finance|auditor|bookkeeper", TRUE)) ~ "Accounting & Finance",
    str_detect(job, regex("banker|investment|trader|analyst", TRUE)) ~ "Banking & Markets",
    str_detect(job, regex("manager|coordinator|administrator|officer|director", TRUE)) ~ "Management & Administration",
    str_detect(job, regex("sales|retail|buyer|promotion|merchandiser", TRUE)) ~ "Sales & Retail",
    str_detect(job, regex("marketing|advertising|public relations|pr", TRUE)) ~ "Marketing & Communication",
    str_detect(job, regex("artist|designer|illustrator|animator|art", TRUE)) ~ "Art & Design",
    str_detect(job, regex("media|journalist|broadcast|camera|film|editor", TRUE)) ~ "Media & Journalism",
    str_detect(job, regex("lawyer|solicitor|legal|barrister|attorney", TRUE)) ~ "Legal",
    str_detect(job, regex("police|firefighter|armed forces|immigration|customs|military", TRUE)) ~ "Law Enforcement & Defense",
    str_detect(job, regex("environmental|ecologist|conservation|geochemist", TRUE)) ~ "Environmental & Earth Sciences",
    str_detect(job, regex("pharmacist|pharmacologist|toxicologist", TRUE)) ~ "Pharmacy & Toxicology",
    str_detect(job, regex("hospitality|tourism|hotel|restaurant|barista", TRUE)) ~ "Hospitality & Tourism",
    str_detect(job, regex("charity|social|aid|community|volunteer", TRUE)) ~ "Social & Community Work",
    TRUE ~ "Other"
  ))

classified_jobs %>%
  count(job_category_20, sort = TRUE) %>%
  ggplot(aes(x = reorder(job_category_20, n), y = n)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = n), hjust = -0.1, size = 2.5) +  # Tamaño reducido
  coord_flip() +
  labs(
    title = "Job Distribution by Category",
    x = "Job Category",
    y = "Count"
  ) +
  theme_minimal() +
  expand_limits(y = max(classified_jobs %>% count(job_category_20) %>% pull(n)) * 1.1)
```

__Exploring wday__
```{r}
card_fraud %>% 
  group_by(wday) %>%
  summarise(
    n_fraud = sum(is_fraud == "1"), # The total # of fraudulent transactions
    total_n = n(), # The total # of f transactions
    percent_fraud = (n_fraud / total_n) * 100 # The percentage of fraudulent transactions
  ) %>% 
  
  # Plotting our bar graph
  ggplot(aes(x =  wday, y = percent_fraud)) + # The days are already in order
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      title = "Sunday and Monday are the days with least credit card frauds",
      x = NULL,
      y = "Fraudulent transactions as percentage of total (%)"
    ) +
  theme(plot.title.position = "plot",
        plot.title = element_text(hjust = 0)) + # To move the title to the left
  theme_minimal()

# Convert to factor only (without order because step_dummy does not work well with ordered factors)
card_fraud$wday <- factor(card_fraud$wday, ordered = FALSE)
```

Just by LOOKING at the graph, we do see a difference in weekdays, specially if the transaction was made in Sunday and Monday. To test this assumption, we use a Chi-squared test. Our null hypotesis is that fraud is independent to weekday.

```{r}
# Chi-squared test for independence
chisq.test(table(card_fraud$wday, card_fraud$is_fraud))
```
The evidence against the null variable (the p-value) is way to small. So we REJECT the null hypotesis (meaning, fraud DOES depend on weekday). 

__CONCLUSION:__ Weekday IS an useful variable to predict fraud. 

__Exploring month_name__
```{r}
card_fraud %>% 
  group_by(month_name) %>%
  summarise(
    n_fraud = sum(is_fraud == "1"), # The total # of fraudulent transactions
    total_n = n(), # The total # of f transactions
    percent_fraud = (n_fraud / total_n) * 100 # The percentage of fraudulent transactions
  ) %>% 
  
  # Plotting our bar graph
  ggplot(aes(x =  month_name, y = percent_fraud)) + # The days are already in order
    geom_bar(stat = "identity", fill = "steelblue") +
    labs(
      title = "January and February are the months with more credit card frauds",
      x = NULL,
      y = "Fraudulent transactions as percentage of total (%)"
    ) +
  theme(plot.title.position = "plot",
        plot.title = element_text(hjust = 0)) + # To move the title to the left
  theme_minimal()
```
Just by LOOKING at the graph, we do see a difference in months, specially if the transaction was made in January and February. To test this assumption, we use a Chi-squared test. Our null hypotesis is that fraud is independent to month.

```{r}
# Chi-squared test for independence
chisq.test(table(card_fraud$month_name, card_fraud$is_fraud))
```

The evidence against the null variable (the p-value) is way to small. So we REJECT the null hypotesis (meaning, fraud DOES depend on month). 

Finally, for it not to be necessary to create 11 dummy variables in the model (one for each month and excluding the base one), I will create a new variable called quarter and I will use this one as a variable in the model. 

```{r}
card_fraud <- card_fraud %>%
  mutate(quarter = quarter(trans_date_trans_time, with_year = FALSE)) %>% 
  mutate(quarter = factor(quarter))
```

__CONCLUSION:__ Month name is an useful variable to predict fraud. However, instead of this one we create a new variable (quarter) in order to have less categories.

## GEOSPACIAL
__CONCLUSION:__ Geospacial variables are redundant to our analysis since we are already creating two new variables for distance, which are both computed based on our geospacial variables.

## DATES

__Exploring dob__
Makes no sense to explore this variable as we are already creating a new variable called age, which is the age of the card holder at the time of the transaction (computed as the difference between both dates - transaction date and dob). 

__CONCLUSION:__ Date of birth (dob) is NOT an useful variable to predict fraud.

# DATE/TIMES:
__Exploring trans_date_trans_time__
__CONCLUSION:__ Makes no sense to explore this variable as we already created new variables (hour, wday, month_name, age) based on this.

# NUMERICAL
__Exploring year__
Since we only have two years of data (2019 and 2020), there's no need to create a graph for this one, by simply looking two fraud percentage we can conclude if the variable is useful or not.
```{r}
card_fraud %>% 
  group_by(year) %>%
  summarise(
    n_fraud = sum(is_fraud == "1"), # The total # of fraudulent transactions
    total_n = n(), # The total # of f transactions
    percent_fraud = (n_fraud / total_n) * 100 # The percentage of fraudulent transactions
  ) 
```
As we can observe, the fraud percentage is very similar in both years, so we can conclude that year is not a useful variable to predict fraud.

__CONCLUSION:__ Year is NOT an useful variable to predict fraud.

__Exploring amt__

```{r}
# Amt
ggplot(card_fraud, aes(x = amt)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Amount", x = "Amount", y = "Frequency")
summary(card_fraud$amt)

ggplot(card_fraud, aes(x = log(amt))) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Amount in Logs", x = "Amount (log)", y = "Frequency")
```
__CONCLUSION:__
The amt variable is highly right-skewed, with most transactions at lower amounts and a few extreme outliers at high values. This indicates that fraudulent activity may be associated with atypical transaction amounts, making this variable important for fraud detection.

__Exploring city population__
The city_pop variable was originally numeric, but its histogram showed a wide distribution with extreme outliers. To address this, we grouped it into five categories:

Very Small: 0–1,000
Small: 1,001–5,000
Medium: 5,001–50,000
Large: 50,001–500,000
Very Large: 500,001 and above

When we compared these categories with the is_fraud variable, we didn’t observe any clear or significant differences in fraud occurrence across groups.However we will include it in the model to check it statistically. 

__Conclusion__ 
We transformed the variable into a categorical one with four classifications.

```{r}
# Initial histogram: There is a high dispersion
ggplot(card_fraud, aes(x = city_pop)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white") +
  labs(
    title = "Histogram of City Population",
    x = "City Population",
    y = "Count"
  ) +
  theme_minimal()

# Transform the variable into categories
card_fraud = card_fraud %>%
  mutate(city_pop_cat = case_when(
    city_pop <= 1000 ~ "Very Small",
    city_pop <= 5000 ~ "Small",
    city_pop <= 50000 ~ "Medium",
    city_pop <= 500000 ~ "Large",
    TRUE ~ "Very Large"
  )) 

# Count and percentage by category
card_fraud %>% 
  count(city_pop_cat, sort = TRUE) %>%
  mutate(perc = round(n * 100 / sum(n), 2)) %>% 
  ggplot(aes(x = reorder(city_pop_cat, -perc), y = perc)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = paste0(perc, "%")), vjust = -0.5, size = 3) +
  labs(
    title = "City Population Categories",
    x = "City Population Category",
    y = "Percentage (%)"
  ) +
  theme_minimal()

# Barplot of is_fraud and city_pop_cat
card_fraud %>% 
  group_by(city_pop_cat, is_fraud) %>%
  summarise(n = n()) %>%
  mutate(perc = round(n * 100 / sum(n), 2)) %>% 
  ggplot(aes(x = city_pop_cat, y = perc, fill = is_fraud)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = paste0(perc, "%")), position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  labs(
    title = "Fraud Distribution by City Population Category",
    x = "City Population Category",
    y = "Percentage (%)",
    fill = "Is Fraud"
  ) +
  theme_minimal() 
  
```
__Exploring Hour__
The percentage of fraudulent transactions varies significantly by hour. Fraud is most likely to occur during the early hours of the day (midnight to 3 a.m.) and again late at night (10 p.m. to midnight), where fraud rates exceed 2.5%. In contrast, fraud is least common during regular business hours, with percentages remaining well below 1%. These patterns suggest that fraudulent activity tends to occur outside of typical working hours, possibly when detection is less likely.

__CONCLUSION__
We kept the variable because it seems visually to be important for the classification. We will test it with the model. 

```{r}
summary(card_fraud$hour)

card_fraud %>%
  group_by(hour, is_fraud) %>%
  count() %>%
  group_by(hour) %>%
  mutate(perc = round(n * 100 / sum(n), 2)) %>%
  filter(is_fraud == 1) %>%
  ggplot(aes(x = hour, y = perc)) +
  geom_col(fill = "blue") +
  labs(
    title = "Percentage of Fraudulent Transactions by Hour",
    x = "Hour of Day",
    y = "Fraud Percentage (%)"
  ) +
  theme_minimal()

```

__Exploring age__
```{r}
# Age
ggplot(card_fraud, aes(x = age)) +
  geom_histogram(bins = 30, fill = "salmon", color = "black") +
  labs(title = "Distribution of Age", x = "Age", y = "Frequency")
summary(card_fraud$age)

# Amt vs Age (Scatter Plot)
ggplot(card_fraud, aes(x = age, y = amt)) +
  geom_point(alpha = 0.5) +
  labs(title = "Amount vs Age", x = "Age", y = "Amount")

# Amt by Age Group  
card_fraud <- card_fraud %>%
  mutate(age_group = cut(age, breaks = c(0, 18, 30, 45, 60, Inf),
                         labels = c("0-18", "19-30", "31-45", "46-60", "61+")))

ggplot(card_fraud, aes(x = age_group, y = amt)) +
  geom_boxplot(fill = "violet") +
  labs(title = "Amount by Age Group", x = "Age Group", y = "Amount")

# Missing Values
colSums(is.na(card_fraud[c("category", "amt", "age")]))

```
__CONCLUSION:__
The age variable covers a wide range of cardholder ages, with clusters in young and middle-aged groups. The scatter and boxplots show that transaction amounts vary across age groups, with some groups exhibiting higher value outliers. This suggests age influences spending behavior and may contribute to identifying fraud risk.

__Exploring distance_miles and distance_km__
The distance_miles and distance_km variables are numerical variables that represent the distance between the card holder's location and the merchant's location. We will keep only one of these variables, as they represent the same information in different units. We will keep distance_km.

```{r}
#---  Fraudulent transactions rate by distance_miles   ---#

# Percentile 99.5
quantile(card_fraud$distance_km, 0.995, na.rm = TRUE)


# Plot distance by is_fraud
card_fraud %>% 
  ggplot(aes(x = is_fraud, y = distance_km, fill = is_fraud)) +
  
  # Plot type
  geom_violin(trim = FALSE, alpha = 0.5, color = NA) +
  
  # Add boxplot in the center
  geom_boxplot(width = 0.1, outlier.size = 0.8, alpha = 0.7) +
  scale_fill_manual(values = c("steelblue", "tomato")) +
  labs(
    x = "Fraud (0 = No, 1 = Yes)",
    y = "Distance (km)",
    title = "Distribution of distance by fraud"
  ) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 13, face = "bold")
  ) +
  NULL


card_fraud <- card_fraud %>%
  mutate(
    distance_bin = cut(
      distance_km,
      breaks = seq(0, max(distance_km, na.rm=TRUE) + 15, by = 15),
      right = FALSE,
      include.lowest = TRUE)
  )

# Fraudulent transaction by distance and state
card_fraud %>%
  group_by(distance_bin) %>%
  summarise(
    total = n(),
    frauds = sum(is_fraud == "1"),
    fraud_rate = frauds / total,
    .groups = "drop"
  ) %>% 

  # Plot
  ggplot(aes(x = distance_bin, y = fraud_rate)) +
  
  # Plot type
  geom_col() +
  
  # Flip x and y axes
  coord_flip() +
  
  # Add facet wrap
  #facet_wrap(~state, ncol = 5) + 
  
  # Add labels
  labs(
    x = "Distance",
    y = "Fraud percentage",
    title = "Fraudulent transactions rate by distance"
  ) +
  
  # Add theme
  theme_bw() + 
  
  # Adjust y label to percentage
  scale_y_continuous(labels = scales::percent) +
  # Adjust x label size
  theme(axis.text.x = element_text(size = 5)) +
  theme(axis.text.y = element_text(size = 5)) +
  NULL

# Chi-squared test for independence
chisq.test(table(card_fraud$distance_bin, card_fraud$is_fraud))

# Convert distance_bin to a boolean factor variable
card_fraud <- card_fraud %>%
  mutate(
    distance_135_km = as.factor(ifelse(distance_km > 135, "1", "0"))
  )

# Chi-squared test for independence
chisq.test(table(card_fraud$distance_135_km, card_fraud$is_fraud))

```
_CONCLUSION:_ distance_miles is NOT an useful variable to predict fraud.

distance_km is NOT an useful variable to predict fraud, the distribution of distance_km appears similar for both fraudulent and non-fraudulent transactions, suggesting that distance alone may not be a strong predictor of fraud. However, we observe that transactions involving distances greater than 135 km —corresponding to the 99.5th percentile— can be flagged with a new binary variable. Notably, the fraud rate increases for these long-distance transactions; for instance, nearly 1% of transactions over 135 km are fraudulent. This pattern could indicate that such transactions carry a higher risk and may be useful for model features or rule-based alerts. Besides, the p-value of the chi-squared test for independence between distance_135_km and is_fraud is less than 0.05, indicating that there is a significant association between these two variables.

distance_135_km is YES an useful variable to predict fraud.


## Which variables to keep in your model?

You have a number of variables and you have to decide which ones to use in your model. For instance, you have the latitude/lognitude of the customer, that of the merchant, the same data in radians, as well as the `distance_km` and `distance_miles`. Do you need them all? 

### STRING:
- category
- wday
- quarter
- city_pop_cat
- distance_135_km

### NUMERICAL:
- amt
- hour
- age

```{r}
# Select the variables to keep in the model
card_fraud <- card_fraud %>% 
  select(is_fraud, category, wday, quarter, city_pop_cat, distance_135_km,
         amt, hour, age)
```


## Fit your workflows in smaller sample

You will be running a series of different models, along the lines of the California housing example we have seen in class. However, this dataset has 670K rows and if you try various models and run cross validation on them, your computer may slow down or crash.

Thus, we will work with a smaller sample of 10% of the values the original dataset to identify the best model, and once we have the best model we can use the full dataset to train- test our best model.


```{r}
# select a smaller subset
my_card_fraud <- card_fraud %>% 
  # select a smaller subset, 10% of the entire dataframe 
  slice_sample(prop = 0.10) 
```


## Split the data in training - testing

```{r}
# **Split the data**

set.seed(123)

data_split <- initial_split(my_card_fraud, # updated data
                           prop = 0.8, 
                           strata = is_fraud)

card_fraud_train <- training(data_split) 
card_fraud_test <- testing(data_split)
```


## Cross Validation

Start with 3 CV folds to quickly get an estimate for the best model and you can increase the number of folds to 5 or 10 later.

```{r}
set.seed(123)
cv_folds <- vfold_cv(data = card_fraud_train, 
                          v = 3, 
                          strata = is_fraud)
cv_folds 
```


## Define a tidymodels `recipe`

What steps are you going to add to your recipe? Do you need to do any log transformations?

```{r, define_recipe}

fraud_rec <- recipe(is_fraud ~ ., data = card_fraud_train) %>%
  step_novel(all_nominal(), -all_outcomes()) %>% # Use *before* `step_dummy()` so new level is dummified
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_numeric(), -all_outcomes())  %>% 
  step_normalize(amt,hour,age) #%>% 
  #step_corr(all_predictors(), threshold = 0.75, method = "spearman") 

```

Once you have your recipe, you can check the pre-processed dataframe 

```{r}
prepped_data <- 
  fraud_rec %>% # use the recipe object
  prep() %>% # perform the recipe on training data
  juice() # extract only the preprocessed dataframe 

glimpse(prepped_data)

```


## Define various models

You should define the following classification models:

1. Logistic regression, using the `glm` engine
2. Decision tree, using the `C5.0` engine
3. Random Forest, using  the `ranger` engine and setting `importance = "impurity"`)  
4. A boosted tree using Extreme Gradient Boosting, and the `xgboost` engine
5. A k-nearest neighbours,  using 4 nearest_neighbors and the `kknn` engine  

```{r, define_models}
## Model Building 

# 1. Pick a `model type`
# 2. set the `engine`
# 3. Set the `mode`:  classification


```

## Bundle recipe and model with `workflows`

```{r, define_workflows}


## Bundle recipe and model with `workflows`


log_wflow <- # new workflow object
 workflow() %>% # use workflow function
 add_recipe(fraud_rec) %>%   # use the new recipe
 add_model(log_spec)   # add your model spec

```


## Fit models

You may want to compare the time it takes to fit each model. `tic()` starts a simple timer and `toc()` stops it

```{r, fit_models}
tic()
log_res <- log_wflow %>% 
  fit_resamples(
    resamples = cv_folds, 
    metrics = metric_set(
      recall, precision, f_meas, accuracy,
      kap, roc_auc, sens, spec),
    control = control_resamples(save_pred = TRUE)) 
time <- toc()
log_time <- time[[4]]


```

## Compare models

```{r, compare_models}
## Model Comparison

log_metrics <- 
  log_res %>% 
  collect_metrics(summarise = TRUE) %>%
  # add the name of the model to every row
  mutate(model = "Logistic Regression",
         time = log_time)

# add mode models here

# create dataframe with all models
model_compare <- bind_rows(log_metrics,
                            tree_metrics,
                            rf_metrics,
                           xgb_metrics,
                           knn_metrics
                      ) %>% 
  # get rid of 'sec elapsed' and turn it into a number
  mutate(time = str_sub(time, end = -13) %>% 
           as.double()
         )


```

## Which metric to use

This is a highly imbalanced data set, as roughly 99.5% of all transactions are ok, and it's only 0.5% of transactions that are fraudulent. A `naive` model, which classifies everything as ok and not-fraud, would have an accuracy of 99.5%, but what about the sensitivity, specificity, the AUC, etc?

## `last_fit()`
```{r}

## `last_fit()` on test set

# - `last_fit()`  fits a model to the whole training data and evaluates it on the test set. 
# - provide the workflow object of the best model as well as the data split object (not the training data). 
 

```



## Get variable importance using `vip` package


```{r}

```

## Plot Final Confusion matrix and ROC curve


```{r}
## Final Confusion Matrix

last_fit_xgb %>%
  collect_predictions() %>% 
  conf_mat(is_fraud, .pred_class) %>% 
  autoplot(type = "heatmap")


## Final ROC curve
last_fit_xgb %>% 
  collect_predictions() %>% 
  roc_curve(is_fraud, .pred_1) %>% 
  autoplot()
```


##  Calculating the cost of fraud to the company


- How much money (in US\$ terms) are fraudulent transactions costing the company? Generate a table that summarizes the total amount of legitimate and fraudulent transactions per year and calculate the % of fraudulent transactions, in US\$ terms. Compare your model vs the naive classification that we do not have any fraudulent transactions. 

```{r}
#| label: savings-for-cc-company

best_model_preds <- 
  best_model_wflow %>% 
  fit(data = card_fraud_train) %>%  
  
  ## Use `augment()` to get predictions for entire data set
  augment(new_data = card_fraud)

best_model_preds %>% 
  conf_mat(truth = is_fraud, estimate = .pred_class)

cost <- best_model_preds %>%
  select(is_fraud, amt, pred = .pred_class) 

cost <- cost %>%
  mutate(
  

  # naive false-- we think every single transaction is ok and not fraud


  # false negatives-- we thought they were not fraud, but they were

  
  
  # false positives-- we thought they were fraud, but they were not

  
    
  # true positives-- we thought they were fraud, and they were 


  
  # true negatives-- we thought they were ok, and they were 
)
  
# Summarising

cost_summary <- cost %>% 
  summarise(across(starts_with(c("false","true", "amt")), 
            ~ sum(.x, na.rm = TRUE)))

cost_summary

```


- If we use a naive classifier thinking that all transactions are legitimate and not fraudulent, the cost to the company is `r scales::dollar(cost_summary$false_naives)`.
- With our best model, the total cost of false negatives, namely transactions our classifier thinks are legitimate but which turned out to be fraud, is `r scales::dollar(cost_summary$false_negatives)`.

- Our classifier also has some false positives, `r scales::dollar(cost_summary$false_positives)`, namely flagging transactions as fraudulent, but which were legitimate. Assuming the card company makes around 2% for each transaction (source: https://startups.co.uk/payment-processing/credit-card-processing-fees/), the amount of money lost due to these false positives is `r scales::dollar(cost_summary$false_positives * 0.02)`

- The \$ improvement over the naive policy is `r scales::dollar(cost_summary$false_naives - cost_summary$false_negatives - cost_summary$false_positives * 0.02)`.
