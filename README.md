# SmartDemand Planner

A retail planning prototype that turns uploaded sales data into demand forecasts, stock actions, and purchase guidance. It predicts product demand from historic sales context, product/store details, promotions, calendar effects, weather, price, and inventory, then translates that forecast into a clear planning action.

## Features

- Smart upload flow with retail column mapping and default fallback fields
- Streamlit dashboard with business overview, watchlist, scenario planner, and action center
- Product/store-level Random Forest demand forecast
- Chronological train/test evaluation with MAE, RMSE, R², and MAPE
- Explainable inventory decision engine: safety stock, target stock, order quantity, and reorder/overstock status
- What-if comparison between base and planned scenarios
- Upload-first data flow, so the app accepts the supplied Excel dataset without hard-coded data dependencies

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. In the sidebar, upload `SmartDemandAI_Dataset.xlsx`.

The dashboard also looks for `data/SmartDemandAI_Dataset.xlsx` and, on the original development machine, the supplied Downloads location.

## Optional command-line training

```powershell
python scripts/train_model.py "C:\path\to\SmartDemandAI_Dataset.xlsx"
```

This saves a reusable model and metric report in `artifacts/`. CSV input is also supported for automated pipelines.

## Dataset contract

The best results come from a retail workbook with these fields:

`Date, Product_ID, Product_Name, Category, Store_ID, City, Unit_Price, Promotion, Discount_Percent, Holiday, Weekend, Temperature, Rainfall, Inventory, Reorder_Level, Demand`

The app can auto-map common retail aliases such as `sales`, `stock`, `price`, `sku`, and `store`. Missing non-core retail fields are backfilled with safe defaults for demo use.

## Model design

The app orders records by date and holds out the newest 20% for evaluation. This avoids evaluating the model on data that predates its training data. The Random Forest uses categorical product/store/location details plus operational and calendar features. It does not use the dataset's pre-labelled `Inventory_Recommendation` column to predict demand.

Inventory logic follows:

- Safety stock = max(reorder level, 20% of predicted demand)
- Target stock = predicted demand + safety stock
- Recommended order = max(0, target stock - current inventory)
- Inventory is marked reorder, sufficient, or overstock based on current inventory relative to target stock.

## Demo flow

1. Open **Overview** to show trend, category mix, and the latest stock watchlist.
2. Open **Forecast Studio**, select a product and store, then simulate promotion, discount, inventory, weather, and date conditions.
3. Show the base-vs-planned demand comparison, recommended order, and decision notes.
4. Open **Action Center** to highlight urgent purchase approvals and overstock items.
5. Open **Forecast Quality** to explain out-of-time evaluation and model drivers.

## Scope and next steps

This prototype is decision support, not an autonomous purchasing system. Production deployment should add store-level daily sales feeds, lead time and supplier constraints, backtesting by product, drift monitoring, authentication, and a database/API layer.
