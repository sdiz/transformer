### Transformer-basiertes Value-at-Risk (VaR) mit Makrodaten

Dieses Projekt schätzt einen einseitigen VaR auf Tages-/Wochen-/Monatsbasis mit einem Transformer-Modell und Makro-/Marktdaten (FRED, S&P 500). Es enthält Skripte zum Ausführen von Baseline- und Big-Varianten, zur Aggregation der Ergebnisse und zur Erstellung einer Auswertung.

### Voraussetzungen

- **Python**: 3.10–3.11 empfohlen (virtuelle Umgebung empfohlen)
- **OS**: macOS/Linux/Windows. Für Apple Silicon kann optional GPU-Beschleunigung mit `tensorflow-metal` genutzt werden
- **FRED API Key**: erforderlich als Umgebungsvariable `FRED_API_KEY` (kostenlos unter [FRED API Key beantragen](https://fred.stlouisfed.org/docs/api/api_key.html))

### Installation

```bash
cd /Users/sdiz/Desktop/BA/Projekt
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# FRED API Key setzen (Beispiel macOS/Linux)
export FRED_API_KEY="<DEIN_KEY>"
```

Hinweise:
- Beim ersten Lauf lädt `transformer.py` Markt- und Makrodaten und cached sie unter `data/combined_*.csv`. Mit `--refresh-cache` wird der Cache neu aufgebaut.
- Internetzugang ist erforderlich (FRED, Yahoo Finance).

### Schnellstart: Ein einzelner Run

Die Hauptlogik liegt in `transformer.py`. Beispielausführung (täglich, CLS-Head, mit Plots):

```bash
python transformer.py \
  --seq-length 60 --z-window 126 \
  --cls-head --use-returns \
  --refit Y --calibration none \
  --oos-years 1987 1998 2001 2008 2011 2015 2020 2022 \
  --plots \
  --outdir outputs/base
```

Wichtige Flags (Auszug, vollständige Hilfe mit `-h`):
- **Fenster/Preprocessing**: `--seq-length`, `--z-window`, `--zscore/--no-zscore`, `--use-returns`, `--use-spx-close`
- **Architektur**: `--layers`, `--head-size`, `--num-heads`, `--ff-dim`, `--dropout`, `--attn-dropout`, `--cls-head/--last-token`, `--causal`
- **Walk-Forward**: `--refit {Y,Q,M}`, `--oos-years`, `--calibration {none,scale,conformal}`, `--calib-window`, `--timeframe {daily,weekly,monthly}`
- **I/O**: `--outdir`, `--plots`, `--end-date`, `--data-start-year`, `--refresh-cache`

### Base vs. Big Modelle ausführen

Es gibt zwei bequeme Runner-Skripte. Für klare Trennung der Ergebnisse wird empfohlen, das `--outdir`-Flag zu setzen, sodass Runs unter `outputs/base` bzw. `outputs/big` landen (notwendig für die Aggregation).

- Base (kompaktere Architektur; Defaults aus `transformer.py`):
  ```bash
  MAX_JOBS=2 bash run_experiments_base.sh \
    && echo "Tipp: In der Datei run_experiments_base.sh das Python-Kommando um --outdir outputs/base ergänzen."
  ```
  Entspricht sinngemäß Runs mit typischen Defaults (z. B. `layers=4`, `head-size=16`, `num-heads=4`, `ff-dim=128`).

- Big (größeres Modell; ressourcenintensiv):
  ```bash
  MAX_JOBS=1 bash run_experiments_big.sh \
    && echo "Tipp: In der Datei run_experiments_big.sh das Python-Kommando um --outdir outputs/big ergänzen."
  ```
  Entspricht z. B. `--layers 6 --head-size 32 --num-heads 8 --ff-dim 1024`. Reduziere ggf. `--batch-size` bei Speichermangel.

Direkt per CLI ohne Skripte:
- Base-Beispiel:
  ```bash
  python transformer.py --seq-length 60 --z-window 126 \
    --layers 4 --head-size 16 --num-heads 4 --ff-dim 128 \
    --cls-head --use-returns --refit Y --calibration none \
    --oos-years 1987 1998 2001 2008 2011 2015 2020 2022 \
    --plots --outdir outputs/base
  ```
- Big-Beispiel:
  ```bash
  python transformer.py --seq-length 60 --z-window 126 \
    --layers 6 --head-size 32 --num-heads 8 --ff-dim 1024 \
    --cls-head --use-returns --refit Y --calibration none \
    --oos-years 1987 1998 2001 2008 2011 2015 2020 2022 \
    --plots --outdir outputs/big
  ```

Parallelisierung: Die Runner-Skripte respektieren `MAX_JOBS` (z. B. `MAX_JOBS=2`). Auf Laptops die Big-Variante mit `MAX_JOBS=1` laufen lassen.

### Ausgabe-Struktur

Jeder Run schreibt nach `outputs/<gruppe>/run_YYYYmmdd_HHMMSS_<hash>/`:
- `config.json`: alle Hyperparameter
- `wf_metrics.csv`: Kennzahlen pro WF-Periode (Breach Rate, RMSE, Pinball, Backtests, ...)
- `plots/`: Visualisierungen (Serien, Heatmaps, Rolling-Coverage, ... falls `--plots`)
- pro Jahr/Periode: `wf_<period>_series.csv`, `wf_<period>_metrics_compare.csv`, `wf_<period>_permutation_importance.csv`, `wf_<period>_zscore_*`

### Aggregation der Runs (kompakte CSVs)

Aggregiert Serien/Metriken aller Runs in `outputs/aggregated/<gruppe>/*.csv` sowie (optional) Master-CSV über Gruppen.

```bash
python aggregate_runs.py \
  --root outputs \
  --out outputs/aggregated \
  --groups base,big \
  --make-master
```

Wichtige Optionen: `--no-series`, `--no-metrics`, `--no-compare`, `--no-permutation`, `--zscore {none,summary}`.

### Dashboard-Auswertung der Runs

Erzeugt tabellarische Zusammenfassungen und Plots für einen Wurzelordner mit `run_*`-Unterordnern. Empfohlen: getrennt für Base/Big laufen lassen und danach optional zusammenziehen.

```bash
# Base auswerten
python run_analysis.py --root outputs/base --out outputs/runs_dashboard/base

# Big auswerten
python run_analysis.py --root outputs/big  --out outputs/runs_dashboard/big

# Master-Dashboards (CSV) über alle Gruppen zusammenführen
python aggregate_runs_dashboard.py --root outputs/runs_dashboard --out outputs/runs_dashboard
```

Wichtige Artefakte unter `outputs/runs_dashboard/`:
- `runs_summary.csv`: Kennzahlen je Run (Mittel/Std über Perioden)
- `compare_summary.csv`: Transformer vs. Baselines (GARCH/LSTM)
- `correlations_spearman.csv`: Korrelationen Kennzahl × Hyperparameter
- Hierarchie: `<gruppe>/<model_size>/<head>/` mit den jeweiligen Berichten/Plots

### Häufige Probleme

- Fehler "FRED_API_KEY nicht gesetzt": `export FRED_API_KEY=...` setzen und Terminal neu starten
- Download-/TLS-Fehler: Netzwerk prüfen; Firewall/Proxy kann FRED/Yahoo blockieren
- Speicherfehler (Out of Memory): `--batch-size` reduzieren, Modell verkleinern (`--layers`, `--ff-dim`, `--head-size`), `MAX_JOBS=1`

### Lizenz / Zitation

Interne Forschungsnutzung.


