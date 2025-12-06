# Customer Personality - Streamlit App

## Problem Statement
The company wants to identify customers who are likely to respond positively to marketing campaigns (response column = 1). However, the number of customers who respond is very small compared to those who do not respond, **resulting in class imbalance.**

In a business context, **failing to detect customers who are actually interested (False Negative)** will cause the company to lose conversion opportunities, reduce campaign effectiveness, and waste marketing budget allocation.

Therefore, the main objective of the model is not only to achieve high accuracy, but to **maximize Recall** for the positive class (response = 1) — that is, to ensure that as many genuinely interested customers as possible are detected by the model.


## Data Understanding

| Variable               | Type               | Categories / Levels                 | Description                          |
|------------------------|--------------------|--------------------------------------|--------------------------------------|
| **income**             | Numeric (int/float)| —                                    | Annual customer revenue              |
| **recency**            | Numeric (int)      | —                                    | Days since last purchase             |
| **numwebvisitsmonth**  | Numeric (int)      | —                                    | Website visits per month             |
| **numwebpurchases**    | Numeric (int)      | —                                    | Purchases via website                |
| **numstorepurchases**  | Numeric (int)      | —                                    | Purchases in physical store          |
| **numcatalogpurchases**| Numeric (int)      | —                                    | Purchases via catalog                |
| **numdealspurchases**  | Numeric (int)      | —                                    | Purchases during deals/promotions    |
| **mntwines**           | Numeric (float/int)| —                                    | Expenditure on wine                  |
| **mntfruits**          | Numeric (float/int)| —                                    | Expenditure on fruits                |
| **mntmeatproducts**    | Numeric (float/int)| —                                    | Expenditure on meat                  |
| **mntfishproducts**    | Numeric (float/int)| —                                    | Expenditure on fish                  |
| **mntsweetproducts**   | Numeric (float/int)| —                                    | Expenditure on sweets                |
| **mntgoldprods**       | Numeric (float/int)| —                                    | Expenditure on gold products         |
| **education**          | Categorical        | Basic, Graduation, Master, PhD       | Highest education level              |
| **marital_status**     | Categorical        | Married, Together, Divorced, Single  | Customer marital status              |


