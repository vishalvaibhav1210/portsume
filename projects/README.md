# Independent retail data science projects

Five end-to-end projects on the core problems of retail and customer data science: promotions, pricing, customer value, customer understanding and range. They are personal projects built on **synthetic data**, not client or employer work.

All five share one simulated grocery retailer ([`retail_sim/simulate.py`](retail_sim/simulate.py)): 120 products in 24 subcategories over 104 weeks, plus 6,000 loyalty customers and about 600k basket lines. Because the simulator knows the true elasticities, cannibalisation, churn and transfer rates, every method is checked against **ground truth** as well as a holdout period. Each result below can be reproduced by running the script.

| Project | Question | Headline result |
|---|---|---|
| [Promotion uplift](promo-uplift/) | Which promotions actually add volume and margin? | 25% of gross promo lift is cannibalised or pulled forward; 82% of promotions lose margin, and every depth over 10% is margin-negative on average |
| [Price optimisation](price-optimisation/) | What should shelf prices be within guardrails? | Shrinkage cuts elasticity error from RMSE 5.96 to 0.37; the price plan delivers +4.8% margin against true demand (92% of the oracle's +5.3%) |
| [Customer lifetime value](customer-lifetime-value/) | Who will be worth what over the next 6 months? | BG/NBD + Gamma-Gamma: aggregate error 2.2% vs 182% for a run-rate baseline; top decile captures 62% of future spend |
| [Need-state segmentation](need-state-segmentation/) | What missions do customers shop for? | LDA recovers the five need states (89% assignment accuracy) **and** each customer's mission mix (r = 0.90), which K-means cannot |
| [Assortment & substitution](assortment-substitution/) | Which SKUs can be delisted safely? | A transfer-aware range review cuts revenue lost from an 18-SKU delist by 19% compared with cutting the lowest sellers |

## Run it

```bash
cd projects
python3 -m pip install -r requirements.txt
python3 promo-uplift/promo_uplift.py
python3 price-optimisation/price_optimisation.py
python3 customer-lifetime-value/clv.py
python3 need-state-segmentation/segmentation.py
python3 assortment-substitution/assortment.py
```

Each script writes `results/metrics.json`, a chart and a CSV of detailed outputs. The simulation is seeded, so the numbers above reproduce exactly.
