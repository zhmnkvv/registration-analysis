"""Анализ регистраций из registrations.csv.

Исходный файл только читается. Печатает все показатели для отчёта
и сохраняет графики в папку charts/.

Запуск:  py analysis.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
SRC = ROOT / "registrations.csv"
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------- загрузка
raw_bytes = SRC.read_bytes()
raw = pd.read_csv(SRC, dtype=str, keep_default_na=False)

print("=== КАЧЕСТВО ДАННЫХ ===")
print("BOM в начале файла:", raw_bytes.startswith(b"\xef\xbb\xbf"))
print("Переводы строк CRLF:", b"\r\n" in raw_bytes)
print("Строк данных:", len(raw), "| колонки:", list(raw.columns))
print("Пустые значения по колонкам:\n", (raw.apply(lambda c: c.str.strip() == "")).sum().to_string())
ws = {c: int((raw[c] != raw[c].str.strip()).sum()) for c in raw.columns}
print("Лишние пробелы по краям:", ws)
for c in ["city", "source", "ticket", "paid"]:
    vals = raw[c].str.strip()
    print(f"Уникальные значения {c}: {sorted(vals.unique())}")
    variants = vals.groupby(vals.str.lower()).nunique()
    print(f"  варианты написания одного значения: {variants[variants > 1].to_dict()}")
print("Полностью одинаковых строк:", int(raw.duplicated().sum()))
print("Повторяющихся имён:", raw["name"].str.strip().duplicated(keep=False).sum())
print("Повтор имя+город:", raw[["name", "city"]].duplicated(keep=False).sum())

# ---------------------------------------------------------- нормализация
df = raw.apply(lambda c: c.str.strip())
df["city"] = df["city"].str.title()
df["paid"] = df["paid"].str.lower()
bad_paid = df[~df["paid"].isin(["да", "нет"])]
print("Нестандартные значения paid:", len(bad_paid))
df["registered"] = pd.to_datetime(df["registered"], format="%Y-%m-%d", errors="coerce")
print("Нераспознанные даты:", int(df["registered"].isna().sum()))
print("Диапазон дат:", df["registered"].min().date(), "—", df["registered"].max().date())
df["is_paid"] = df["paid"] == "да"


# ------------------------------------------------------------- показатели
def summary(col: str) -> pd.DataFrame:
    g = df.groupby(col)["is_paid"].agg(всего="size", оплачено="sum")
    g["не_оплачено"] = g["всего"] - g["оплачено"]
    g["доля_рег_%"] = (g["всего"] / len(df) * 100).round(1)
    g["доля_оплат_%"] = (g["оплачено"] / g["всего"] * 100).round(1)
    g["доля_от_всех_оплат_%"] = (g["оплачено"] / df["is_paid"].sum() * 100).round(1)
    return g.sort_values(["всего", "оплачено"], ascending=False)


n, paid = len(df), int(df["is_paid"].sum())
print("\n=== ОБЩЕЕ ===")
print(f"Всего: {n}, оплачено: {paid} ({paid / n:.1%}), не оплачено: {n - paid} ({(n - paid) / n:.1%})")
print("Уникальных городов/источников/билетов:",
      df["city"].nunique(), df["source"].nunique(), df["ticket"].nunique())

tables = {c: summary(c) for c in ["city", "source", "ticket"]}
for c, t in tables.items():
    print(f"\n=== {c.upper()} ===\n{t.to_string()}")

print("\n=== ПЕРЕКРЁСТНЫЕ ТАБЛИЦЫ (всего / оплачено) ===")
for a, b in [("source", "ticket"), ("city", "source"), ("city", "ticket")]:
    print(f"\n{a} x {b}, всего:\n", pd.crosstab(df[a], df[b], margins=True, margins_name="Итого").to_string())
    print(f"{a} x {b}, оплачено:\n", pd.crosstab(df[a], df[b], values=df["is_paid"], aggfunc="sum").fillna(0).astype(int).to_string())

print("\n=== ДИНАМИКА ===")
df["week"] = df["registered"].dt.to_period("W-SUN")
wk = df.groupby("week")["is_paid"].agg(всего="size", оплачено="sum")
wk["доля_оплат_%"] = (wk["оплачено"] / wk["всего"] * 100).round(1)
print(wk.to_string())
daily = df.groupby("registered").size()
print("Дней с регистрациями:", daily.size, "| пиковые дни:\n", daily.sort_values(ascending=False).head(3).to_string())
print("День недели:\n", df["registered"].dt.day_name().value_counts().to_string())

cut = pd.Timestamp("2026-11-01")
for label, part in [("до 1 ноября", df[df["registered"] < cut]), ("с 1 ноября", df[df["registered"] >= cut])]:
    print(f"{label}: {len(part)} рег., оплачено {int(part['is_paid'].sum())} ({part['is_paid'].mean():.1%})")

print("\nНеоплаченные регистрации:\n", df[~df["is_paid"]][["name", "city", "registered", "source", "ticket"]]
      .sort_values("registered").to_string(index=False))

# ------------------------------------------- город × месяц регистрации
MONTHS = {10: "Октябрь", 11: "Ноябрь"}
CITY_GROUPS = {"Астана": "Астана + Алматы", "Алматы": "Астана + Алматы",
               "Караганда": "Караганда + Шымкент", "Шымкент": "Караганда + Шымкент",
               "Актобе": "Актобе"}
GROUP_ORDER = ["Астана + Алматы", "Караганда + Шымкент", "Актобе"]
unknown_month = set(df["registered"].dt.month) - set(MONTHS)
assert not unknown_month, f"В данных есть месяцы вне октября/ноября: {unknown_month}"
assert set(df["city"]) <= set(CITY_GROUPS), "Есть город без группы в CITY_GROUPS"
df["month"] = df["registered"].dt.month.map(MONTHS)
df["city_group"] = df["city"].map(CITY_GROUPS)


def month_table(col: str) -> pd.DataFrame:
    g = df.groupby([col, "month"])["is_paid"].agg(всего="size", оплачено="sum")
    g["доля_оплат_%"] = (g["оплачено"] / g["всего"] * 100).round(1)
    return g


print("\n=== ГОРОД × МЕСЯЦ РЕГИСТРАЦИИ ===")
city_month = month_table("city")
group_month = month_table("city_group")
print(city_month.to_string())
print("\nГруппы городов × месяц:\n", group_month.to_string())
print("\nМесяц в целом:\n", df.groupby("month")["is_paid"].agg(всего="size", оплачено="sum").to_string())

# ---------------------------------------------------------------- графики
PAID_C, UNPAID_C = "#2e7d5b", "#d9a441"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def stacked_bar(col: str, title: str, fname: str) -> None:
    t = tables[col].sort_values("всего")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(t) + 1.6))
    ax.barh(t.index, t["оплачено"], color=PAID_C, label="оплачено")
    ax.barh(t.index, t["не_оплачено"], left=t["оплачено"], color=UNPAID_C, label="не оплачено")
    for i, (tot, pct) in enumerate(zip(t["всего"], t["доля_оплат_%"])):
        ax.text(tot + 0.15, i, f"{tot}  ({pct:.0f}% оплачено)", va="center", fontsize=10)
    ax.set_xlim(0, t["всего"].max() * 1.45)
    ax.set_xlabel("Регистрации")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS / fname, dpi=150)
    plt.close(fig)


stacked_bar("city", "Регистрации по городам", "cities.png")
stacked_bar("source", "Регистрации по источникам", "sources.png")
stacked_bar("ticket", "Регистрации по типам билетов", "tickets.png")

# общий статус оплаты
fig, ax = plt.subplots(figsize=(8, 1.6))
ax.barh([0], [paid], color=PAID_C)
ax.barh([0], [n - paid], left=[paid], color=UNPAID_C)
ax.text(paid / 2, 0, f"оплачено: {paid} ({paid / n:.0%})", ha="center", va="center", color="white", fontweight="bold")
ax.text(paid + (n - paid) / 2, 0, f"не оплачено: {n - paid} ({(n - paid) / n:.0%})", ha="center", va="center", fontweight="bold")
ax.set_xlim(0, n)
ax.axis("off")
ax.set_title(f"Статус оплаты, всего {n} регистраций", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(CHARTS / "paid_status.png", dpi=150)
plt.close(fig)

# динамика по дням (накопительно)
rng = pd.date_range(df["registered"].min(), df["registered"].max())
cum_all = df.groupby("registered").size().reindex(rng, fill_value=0).cumsum()
cum_paid = df[df["is_paid"]].groupby("registered").size().reindex(rng, fill_value=0).cumsum()
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.step(rng, cum_all, where="post", color="#3b5b92", lw=2, label="все регистрации")
ax.step(rng, cum_paid, where="post", color=PAID_C, lw=2, label="оплаченные")
ax.set_title("Накопленное число регистраций по дням", loc="left", fontweight="bold")
ax.set_ylabel("Регистрации (накопительно)")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(CHARTS / "timeline.png", dpi=150)
plt.close(fig)

from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Patch, Rectangle

# оплата по группам городов: октябрь и ноябрь
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True, sharex=True)
xmax = group_month["всего"].max()
for ax, month in zip(axes, MONTHS.values()):
    for i, grp in enumerate(GROUP_ORDER[::-1]):
        if (grp, month) in group_month.index:
            r = group_month.loc[(grp, month)]
            tot, pd_ = int(r["всего"]), int(r["оплачено"])
            ax.barh(i, pd_, color=PAID_C)
            ax.barh(i, tot - pd_, left=pd_, color=UNPAID_C)
            ax.text(tot + 0.2, i, f"{pd_} из {tot} ({pd_ / tot:.0%})", va="center", fontsize=10)
        else:
            ax.text(0.2, i, "нет регистраций", va="center", fontsize=10, color="gray")
    ax.set_yticks(range(len(GROUP_ORDER)), GROUP_ORDER[::-1])
    ax.set_xlim(0, xmax * 1.5)
    ax.set_title({"Октябрь": "Зарегистрировались в октябре",
                  "Ноябрь": "Зарегистрировались в ноябре"}[month], loc="left", fontsize=11)
    ax.set_xlabel("Регистрации")
fig.legend(handles=[Patch(color=PAID_C, label="оплачено"), Patch(color=UNPAID_C, label="не оплачено")],
           loc="lower right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 0.0))
fig.suptitle("Доля оплат по группам городов в зависимости от месяца регистрации",
             x=0.01, ha="left", fontweight="bold")
fig.text(0.01, 0.01, "Подпись: оплачено из всего (доля). Группы малы, проценты неустойчивы.",
         fontsize=9, color="gray")
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(CHARTS / "city_month.png", dpi=150)
plt.close(fig)

# тепловая карта источник x билет: цвет = доля оплаченных
MIN_N = 5  # при меньшем числе регистраций доля считается ненадёжной
ct = pd.crosstab(df["source"], df["ticket"])
cp = pd.crosstab(df["source"], df["ticket"], values=df["is_paid"], aggfunc="sum").fillna(0).astype(int)
pay_cmap = LinearSegmentedColormap.from_list("pay", [UNPAID_C, "#f4f1ea", PAID_C])
norm = Normalize(0, 1)
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v, p = int(ct.values[i, j]), int(cp.values[i, j])
        if v == 0:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#eeeeee", edgecolor="white", lw=2))
            ax.text(j, i, "—", ha="center", va="center", color="gray")
            continue
        rgb = to_rgb(pay_cmap(norm(p / v)))
        small = v < MIN_N
        if small:  # бледнее и со штриховкой: доля по 1–4 регистрациям ненадёжна
            rgb = tuple(c * 0.45 + 0.55 for c in rgb)
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=rgb, edgecolor="none"))
        if small:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="#b5b5b5", lw=0, hatch="///"))
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="white", lw=2))
        dark = not small and (p / v > 0.8 or p / v < 0.15)
        ax.text(j, i, f"{p} / {v}\n{p / v:.0%}", ha="center", va="center",
                color="white" if dark else ("#555555" if small else "black"),
                fontweight="normal" if small else "bold",
                bbox=dict(facecolor=rgb, edgecolor="none", pad=1) if small else None)
ax.set_xlim(-.5, ct.shape[1] - .5)
ax.set_ylim(ct.shape[0] - .5, -.5)
ax.set_xticks(range(ct.shape[1]), ct.columns)
ax.set_yticks(range(ct.shape[0]), ct.index)
ax.set_xlabel("Тип билета")
ax.set_ylabel("Источник регистрации")
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
cb = fig.colorbar(ScalarMappable(norm, pay_cmap), ax=ax, fraction=0.05, pad=0.03)
cb.set_label("Цвет: доля оплаченных регистраций")
cb.set_ticks([0, .25, .5, .75, 1], labels=["0%", "25%", "50%", "75%", "100%"])
cb.outline.set_visible(False)
ax.legend(handles=[Patch(facecolor="white", edgecolor="#999999", hatch="///",
                         label=f"меньше {MIN_N} регистраций — доля ненадёжна"),
                   Patch(facecolor="#eeeeee", label="— регистраций нет")],
          loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
ax.set_title("Доля оплаченных: источник × тип билета\nв ячейке: оплачено / всего",
             loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(CHARTS / "source_ticket.png", dpi=150)
plt.close(fig)

print("\nГрафики сохранены в", CHARTS)
