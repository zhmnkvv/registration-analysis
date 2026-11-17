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

# тепловая карта источник x билет
ct = pd.crosstab(df["source"], df["ticket"])
cp = pd.crosstab(df["source"], df["ticket"], values=df["is_paid"], aggfunc="sum").fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=ct.values.max() * 1.15)
ax.set_xticks(range(ct.shape[1]), ct.columns)
ax.set_yticks(range(ct.shape[0]), ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        v = ct.values[i, j]
        txt = "—" if v == 0 else f"{cp.values[i, j]} из {v}"
        ax.text(j, i, txt, ha="center", va="center", color="white" if v > ct.values.max() / 2 else "black")
ax.set_title("Источник × тип билета: оплачено из всего", loc="left", fontweight="bold")
ax.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(CHARTS / "source_ticket.png", dpi=150)
plt.close(fig)

print("\nГрафики сохранены в", CHARTS)
