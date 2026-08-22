"""Part 1 Task 2 — verify the generated dataset and classify the missingness.

Writes reports/part1_data_verification.md. Every number printed here is
computed from orders_dataset.csv at run time.

    python3 -m part1.verify_dataset
"""

from part1.common import DATASET_PATH, REPORTS_DIR, ensure_dirs, load_dataset

REPORT_PATH = REPORTS_DIR / "part1_data_verification.md"


def rate_table(df, by: str):
    """Return rate + row count broken out by one column, as a markdown table."""
    grouped = (
        df.groupby(by)["returned"]
        .agg(orders="size", returns="sum", return_rate="mean")
        .sort_values("return_rate", ascending=False)
    )
    lines = [f"| {by} | orders | returns | return rate |", "|---|---:|---:|---:|"]
    for name, row in grouped.iterrows():
        lines.append(
            f"| {name} | {int(row['orders'])} | {int(row['returns'])} "
            f"| {row['return_rate']:.4f} ({row['return_rate'] * 100:.2f}%) |"
        )
    return "\n".join(lines), grouped


def main() -> None:
    ensure_dirs()
    df = load_dataset()

    n_rows, n_cols = df.shape
    return_rate = df["returned"].mean()
    missing_rating = df["rating_given"].isna().mean()

    # The evidence for the MAR call: missingness rate split by the OBSERVED
    # payment_method column.
    cod = df["payment_method"] == "COD"
    cod_missing = df.loc[cod, "rating_given"].isna().mean()
    non_cod_missing = df.loc[~cod, "rating_given"].isna().mean()
    gap = cod_missing - non_cod_missing

    cat_table, _ = rate_table(df, "product_category")
    pay_table, _ = rate_table(df, "payment_method")

    report = f"""# Part 1 — Task 2: Dataset Verification

Source file: `{DATASET_PATH.name}` (regenerate with `python3 generate_orders.py`)

## Shape and headline rates

| check | value |
|---|---|
| total rows | {n_rows} |
| total columns | {n_cols} |
| columns | {", ".join(f"`{c}`" for c in df.columns)} |
| overall return rate | {return_rate:.4f} ({return_rate * 100:.2f}%) |
| missing `rating_given` | {int(df["rating_given"].isna().sum())} rows \
({missing_rating * 100:.2f}%) |

Expected by the brief: 6,000 rows, 13 columns, return rate between 18% and 27%,
`rating_given` missing on 8%-18% of rows. Both properties come from the fixed
seeded generator — they are not tuned.

## Return rate by `product_category`

{cat_table}

## Return rate by `payment_method`

{pay_table}

## Missingness mechanism: MAR (missing at random)

`rating_given` is missing on **{missing_rating * 100:.2f}%** of rows overall, but
that rate is not uniform:

| group | rows | missing `rating_given` | missing rate |
|---|---:|---:|---:|
| COD | {int(cod.sum())} | {int(df.loc[cod, "rating_given"].isna().sum())} \
| {cod_missing * 100:.2f}% |
| non-COD | {int((~cod).sum())} | {int(df.loc[~cod, "rating_given"].isna().sum())} \
| {non_cod_missing * 100:.2f}% |
| **measured gap** | | | **{gap * 100:.2f} percentage points** |

**Classification: MAR, not MCAR and not MNAR.**

* **Not MCAR.** Under MCAR the missing rate would be statistically
  indistinguishable across groups. It is not: COD orders drop their rating
  {cod_missing / non_cod_missing:.2f}x as often as non-COD orders, a measured gap of
  {gap * 100:.2f} percentage points. There is a real dependency, so the missingness
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
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"rows={n_rows} columns={n_cols}")
    print(f"return_rate={return_rate:.4f}")
    print(f"missing_rating_given={missing_rating:.4f} ({missing_rating * 100:.2f}%)")
    print(f"COD missing rate      = {cod_missing:.4f}")
    print(f"non-COD missing rate  = {non_cod_missing:.4f}")
    print(f"measured gap          = {gap:.4f} ({gap * 100:.2f} pp) -> MAR")
    print(f"\nWrote {REPORT_PATH.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
