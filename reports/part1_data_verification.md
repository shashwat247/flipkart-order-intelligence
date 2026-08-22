# Part 1 — Task 2: Dataset Verification

Source file: `orders_dataset.csv` (regenerate with `python3 generate_orders.py`)

## Shape and headline rates

| check | value |
|---|---|
| total rows | 6000 |
| total columns | 13 |
| columns | `order_id`, `product_category`, `price_inr`, `discount_pct`, `payment_method`, `customer_tenure_days`, `num_previous_orders`, `num_previous_returns`, `delivery_distance_km`, `delivery_days`, `is_weekend_order`, `rating_given`, `returned` |
| overall return rate | 0.2275 (22.75%) |
| missing `rating_given` | 783 rows (13.05%) |

Expected by the brief: 6,000 rows, 13 columns, return rate between 18% and 27%,
`rating_given` missing on 8%-18% of rows. Both properties come from the fixed
seeded generator — they are not tuned.

## Return rate by `product_category`

| product_category | orders | returns | return rate |
|---|---:|---:|---:|
| Apparel | 1979 | 523 | 0.2643 (26.43%) |
| Footwear | 1071 | 278 | 0.2596 (25.96%) |
| Beauty | 579 | 116 | 0.2003 (20.03%) |
| Home | 1055 | 202 | 0.1915 (19.15%) |
| Electronics | 1316 | 246 | 0.1869 (18.69%) |

## Return rate by `payment_method`

| payment_method | orders | returns | return rate |
|---|---:|---:|---:|
| COD | 2501 | 769 | 0.3075 (30.75%) |
| Wallet | 594 | 106 | 0.1785 (17.85%) |
| Prepaid_UPI | 1448 | 245 | 0.1692 (16.92%) |
| Prepaid_Card | 1457 | 245 | 0.1682 (16.82%) |

## Missingness mechanism: MAR (missing at random)

`rating_given` is missing on **13.05%** of rows overall, but
that rate is not uniform:

| group | rows | missing `rating_given` | missing rate |
|---|---:|---:|---:|
| COD | 2501 | 571 | 22.83% |
| non-COD | 3499 | 212 | 6.06% |
| **measured gap** | | | **16.77 percentage points** |

**Classification: MAR, not MCAR and not MNAR.**

* **Not MCAR.** Under MCAR the missing rate would be statistically
  indistinguishable across groups. It is not: COD orders drop their rating
  3.77x as often as non-COD orders, a measured gap of
  16.77 percentage points. There is a real dependency, so the missingness
  is not completely at random.
* **MAR.** The dependency is entirely on `payment_method`, a column we
  **observe** for every row. Once you condition on `payment_method`, the
  probability of a missing rating carries no further information. That is the
  definition of missing-at-random, and it is what the generator does: it draws
  the missing mask as `P(missing) = 0.22 if payment_method == "COD" else 0.06`.
* **Not MNAR.** The mask is drawn *before* and *independently of* the rating
  value itself — a customer who would have given a 1-star rating is exactly as
  likely to be missing as one who would have given 5 stars, given the same
  payment method. Nothing depends on the unobserved `rating_given` value, so
  the missingness is not missing-not-at-random.

**Consequence for modelling.** Because the mechanism is MAR on an observed
column, median imputation inside a `ColumnTransformer` fitted on the training
split only is a defensible treatment: the one-hot `payment_method` columns
remain in the feature matrix, so the model can still learn the
"rating was missing *and* this was COD" pattern rather than having it silently
washed away by the imputer.
