

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  


# change this to wherever your csv file actually is
CSV_FILE = "nepse_data.csv"

# my csv has these columns (adjust the names if yours are different):
#   Date, Index, Abs_Change, Pct_Change


raw = pd.read_csv(CSV_FILE)

# quick peek so i know nothing bad happened
print("first few rows:")
print(raw.head())
print("\ncolumn names:", raw.columns.tolist())



# if YOUR columns have different names, change the left side of this dict
raw.rename(columns={
    raw.columns[0]: "index_val",
    raw.columns[1]: "abs_change",
    raw.columns[2]: "pct_change",
    raw.columns[3]: "date",
}, inplace=True)

raw["date"] = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")

# drop rows where date parsing failed 
raw.dropna(subset=["date"], inplace=True)

# the csv comes newest-first, so flip it so oldest is at the top
# this makes all the math work correctly (returns go forward in time)
df = raw.sort_values("date").reset_index(drop=True)

# make sure index_val is numeric -- sometimes it comes with commas like "2,345.67"
df["index_val"] = (
    df["index_val"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

print(f"\ndata loaded: {len(df)} trading days  |  {df['date'].min().date()} → {df['date'].max().date()}")

# log returns

df["log_return"] = np.log(df["index_val"] / df["index_val"].shift(1))

# drop the very first row - no return on day 1 since there's nothing before it
df.dropna(subset=["log_return"], inplace=True)


# annualized avg return
TRADING_DAYS = 240

daily_avg   = df["log_return"].mean()
annual_avg  = daily_avg * TRADING_DAYS     # scale daily → yearly
annual_pct  = (np.exp(annual_avg) - 1) * 100   # convert log return to % 

print(f"\naverage daily log return : {daily_avg:.6f}")
print(f"annualised return        : {annual_pct:.2f}%")
print("  → this is roughly what someone earned each year ON AVERAGE just by")
print("    being invested in the nepse index (before inflation, before taxes)")


# volatility

daily_vol  = df["log_return"].std()
annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

print(f"\ndaily volatility         : {daily_vol:.6f}")
print(f"annualised volatility    : {annual_vol * 100:.2f}%")
print("  → this means in a typical year, the nepse index could swing")
print(f"    roughly ±{annual_vol * 100:.0f}% around its average (that's one standard deviation)")


# sharpe ratio
RISK_FREE_RATE = 0.06

sharpe = (annual_pct / 100 - RISK_FREE_RATE) / annual_vol

print(f"\nsharpe ratio             : {sharpe:.2f}")
print("  → above 1.0 is generally considered good")
print("    above 2.0 is great, below 0 means the bank was better than the market")


#drawdowns

rolling_max  = df["index_val"].cummax()
drawdown     = (df["index_val"] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

worst_date = df.loc[drawdown.idxmin(), "date"]
print(f"\nmax drawdown             : {max_drawdown * 100:.2f}%")
print(f"  → the worst fall from a peak happened around {worst_date.strftime('%b %Y')}")
print("    this is the scenario where someone bought at the top and held to the bottom")


# rolling volatility
df["rolling_vol_1y"] = (
    df["log_return"]
    .rolling(TRADING_DAYS)
    .std() * np.sqrt(TRADING_DAYS) * 100   # in %
)


# Cumulative Returns
df["cumulative_return"] = (1 + df["log_return"]).cumprod()
total_growth = (df["cumulative_return"].iloc[-1] - 1) * 100

print(f"\ntotal growth (full period): {total_growth:.1f}%")
print(f"  → rs 100 invested in {df['date'].min().year} would be roughly")
print(f"    rs {df['cumulative_return'].iloc[-1] * 100:.0f} today (ignoring dividends and costs)")


# Plots

# -- window 1: index over time --
fig1, ax1 = plt.subplots(figsize=(11, 5))
fig1.canvas.manager.set_window_title("NEPSE – Index Value Over Time")
ax1.plot(df["date"], df["index_val"], color="#1f77b4", linewidth=1.2)
ax1.fill_between(df["date"], df["index_val"], alpha=0.12, color="#1f77b4")
ax1.set_title("NEPSE Index Value Over Time", fontweight="bold")
ax1.set_ylabel("Index Points")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
fig1.tight_layout()
 
# -- window 2: cumulative growth (rs 100 → ?) --
fig2, ax2 = plt.subplots(figsize=(11, 5))
fig2.canvas.manager.set_window_title("NEPSE – Growth of Rs 100")
ax2.plot(df["date"], df["cumulative_return"] * 100, color="#2ca02c", linewidth=1.3)
ax2.axhline(100, color="gray", linewidth=0.8, linestyle="--", label="starting Rs 100")
ax2.set_title("If You Invested Rs 100 at the Start...", fontweight="bold")
ax2.set_ylabel("Value of Rs 100 Investment")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(fontsize=9)
fig2.tight_layout()
 
# -- window 3: daily returns distribution --
fig3, ax3 = plt.subplots(figsize=(9, 5))
fig3.canvas.manager.set_window_title("NEPSE – Daily Returns Distribution")
ax3.hist(df["log_return"] * 100, bins=80, color="#ff7f0e", edgecolor="white", linewidth=0.3)
ax3.axvline(0, color="black", linewidth=1, linestyle="--")
ax3.axvline(daily_avg * 100, color="red", linewidth=1.5, linestyle="-",
            label=f"avg = {daily_avg*100:.3f}%")
ax3.set_title("Distribution of Daily Returns", fontweight="bold")
ax3.set_xlabel("Daily Log Return (%)")
ax3.set_ylabel("Number of Days")
ax3.legend(fontsize=9)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
fig3.tight_layout()
 
# -- window 4: rolling 1-year volatility --
fig4, ax4 = plt.subplots(figsize=(11, 5))
fig4.canvas.manager.set_window_title("NEPSE – Rolling 1-Year Volatility")
ax4.plot(df["date"], df["rolling_vol_1y"], color="#d62728", linewidth=1.2)
ax4.fill_between(df["date"], df["rolling_vol_1y"],
                 where=(df["rolling_vol_1y"] > 30),
                 color="red", alpha=0.2, label=">30% vol (high risk periods)")
ax4.set_title("Rolling 1-Year Volatility", fontweight="bold")
ax4.set_ylabel("Annualised Volatility (%)")
ax4.grid(axis="y", linestyle="--", alpha=0.4)
ax4.legend(fontsize=9)
fig4.tight_layout()
 
# -- window 5: drawdown over time --
fig5, ax5 = plt.subplots(figsize=(11, 5))
fig5.canvas.manager.set_window_title("NEPSE – Drawdown from Peak")
ax5.fill_between(df["date"], drawdown * 100, 0, color="#9467bd", alpha=0.6)
ax5.set_title("Drawdown from Peak  (How Far Did It Fall?)", fontweight="bold")
ax5.set_ylabel("Drawdown (%)")
ax5.grid(axis="y", linestyle="--", alpha=0.4)
ax5.annotate(f"Worst: {max_drawdown*100:.1f}%",
             xy=(worst_date, max_drawdown * 100),
             xytext=(worst_date, max_drawdown * 100 - 8),
             fontsize=9, color="darkred",
             arrowprops=dict(arrowstyle="->", color="darkred"))
fig5.tight_layout()
 
# open all five windows at once and wait until they're all closed
plt.show()


# print

print("\n" + "="*60)
print("\n" + "="*60)

print(f"""
1. THE INDEX:
   The NEPSE index tracks the overall health of the Nepal
   stock market. Think of it like an average price of all
   listed companies combined.

2. AVERAGE RETURN:
   Over the full period, the index grew about {annual_pct:.1f}% per year
   on average. A savings account gives you maybe 6-7%, so
   the stock market {"beat" if annual_pct > 7 else "didn't clearly beat"} the bank BUT with much more risk.

3. VOLATILITY ({annual_vol*100:.1f}% per year):
   This is how wild the ups and downs are. A number this
   high means the market can easily go up or down by
   {annual_vol*100:.0f}% in a single year. Savings accounts have ~0% volatility.
   More volatility = more stress, but also more potential gain.

4. SHARPE RATIO ({sharpe:.2f}):
   This asks: "was the extra risk worth it compared to just
   leaving money in the bank?"
   {"Yes, above 1 means the risk paid off." if sharpe > 1 else "Debatable. below 1 means the bank might have been safer for the stress."}

5. WORST CRASH ({max_drawdown*100:.1f}% drawdown):
   At its worst point (around {worst_date.strftime('%b %Y')}), someone who bought
   at the peak would have seen {abs(max_drawdown)*100:.1f}% of their money
   disappear on paper. It recovered over time, but that's
   the risk of stocks.
""")