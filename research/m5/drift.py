import numpy as np, pandas as pd
d = pd.read_parquet("m1.parquet")
ny = d.time.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
d["nyh"] = ny.dt.hour.values; d["date"] = (ny + pd.Timedelta(hours=6)).dt.date.values   # trading day starts 18:00 NY
d["yr"] = d.time.dt.year
COST = 0.30
g = d.groupby(["date", "nyh"]).agg(o=("open", "first"), c=("close", "last"), yr=("yr", "first")).reset_index()
rows = []
for h in range(24):
    for k in (1, 2, 3, 4):
        # enter at open of NY hour h, exit at close of hour h+k-1, same trading day
        a = g[g.nyh == h][["date", "o", "yr"]]
        last = (h + k - 1) % 24
        b = g[g.nyh == last][["date", "c"]]
        x = a.merge(b, on="date")
        if (h + k - 1) >= 24 or (h < 18 and h + k - 1 >= 18 and h != 17): continue
        x["r"] = x.c - x.o
        IS, OS = x[x.yr <= 2019], x[x.yr >= 2020]
        side = np.sign(IS.r.mean())
        if side == 0: continue
        net_is = side * IS.r - COST; net_os = side * OS.r - COST
        t_is = net_is.mean() / net_is.std() * np.sqrt(len(net_is))
        t_os = net_os.mean() / net_os.std() * np.sqrt(len(net_os))
        yrs = x.assign(n=side * x.r - COST).groupby("yr").n.sum()
        rows.append(dict(ny_hour=h, hold_h=k, side="BUY" if side > 0 else "SELL", is_avg=net_is.mean(), is_t=t_is, os_avg=net_os.mean(), os_t=t_os,
                         os_pf=net_os[net_os > 0].sum() / -net_os[net_os < 0].sum(), yrs_pos=f"{(yrs>0).sum()}/{len(yrs)}", n_os=len(OS)))
R = pd.DataFrame(rows).sort_values("is_t", ascending=False)
pd.set_option("display.width", 200)
print("Top 12 by in-sample t (2013-2019), then how they did 2020-2026 (unseen). $ per trade at 0.01 lot = 1 oz, after 0.30 cost")
print(R.head(12).round(3).to_string(index=False))
print("\nRandom expectation check - share of all hour/hold combos positive OOS:", (R.os_avg > 0).mean().round(2), "of", len(R))
