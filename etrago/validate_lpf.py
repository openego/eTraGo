"""
Validierung der run_power_flow-Pipeline (eTraGo, Zooming-Branch).

Geprüft wird je Stufe
  * Komponentenzahlen je Scope (Gesamt / AC / AC-DE / AC-Ausland / Nicht-AC)
  * Konsistenz der Scopes (AC-DE + AC-Ausland = AC, AC + Nicht-AC = Gesamt)
  * Verweise auf nicht existierende Buses
  * p_set-Vollständigkeit (Zeitreihe vorhanden, keine NaN)        -> Stufen 1-4
  * p == p_set (Dispatch-Check, Slack-Generatoren ausgenommen)    -> Stufen 5-7
  * erwartete leere Scopes (z.B. Ausland nach Entfernen)

Zusätzlich je Übergang zwischen Stufen
  * Komponenten gehen nicht verloren
  * effektives p_set der erhaltenen Assets bleibt unverändert
  * Summe des effektiven p_set je Komponentenart und Carrier bleibt erhalten
    (Clustering 1 -> 2, DC-Import 2 -> 3)
"""

import contextlib
import io
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

RES = 150
BASE_PATH = f"Zooming-Tests/lokale_tests/AC-{RES}/"
OUTPUT_DIR = BASE_PATH  # Der PDF-Report wird hier gespeichert
ATOL = 1e-3  # MW

# Komponente -> Attributname am Network
COMPONENTS = {
    "Bus": "buses",
    "Line": "lines",
    "Transformer": "transformers",
    "Link": "links",
    "Generator": "generators",
    "Load": "loads",
    "StorageUnit": "storage_units",
    "Store": "stores",
}

# Bus-Spalten je Komponente (zweipolig: beide Enden werden berücksichtigt)
BUS_COLS = {
    "Generator": ("bus",),
    "Load": ("bus",),
    "StorageUnit": ("bus",),
    "Store": ("bus",),
    "Line": ("bus0", "bus1"),
    "Transformer": ("bus0", "bus1"),
    "Link": ("bus0", "bus1"),
}

# Komponente -> (Attributname, Ergebnisgröße, die p_set entsprechen soll)
DISPATCH = {
    "Generator": ("generators", "p"),
    "Link": ("links", "p0"),
    "StorageUnit": ("storage_units", "p"),
    "Store": ("stores", "p"),
}

SCOPES = {
    "all": "Gesamt",
    "ac": "Nur AC",
    "ac_de": "AC in DE",
    "ac_foreign": "AC Ausland",
    "other": "Nicht-AC",
    "other_de": "Nicht-AC in DE",
    "other_foreign": "Nicht-AC Ausland",
}
# Disjunkte Scopes, die zusammen alles abdecken
PARTITION = ("ac_de", "ac_foreign", "other_de", "other_foreign")

MISSING_BUS = "__MISSING_BUS__"


# --------------------------------------------------------------------------- #
# Ausgabe
# --------------------------------------------------------------------------- #
class Report:
    """Gibt Zeilen aus und zählt Fehlschläge."""

    def __init__(self):
        self.n_fail = 0

    def emit(self, label, ok, msg="", indent=6):
        if ok is False:
            self.n_fail += 1
        mark = {True: "✓", False: "✗", None: "–"}[ok]
        print(f"{' ' * indent}[{label:<11}] {mark} {msg}".rstrip())


def ids_str(index, n=3):
    return str(list(index[:n])) + ("..." if len(index) > n else "")


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Scope-Klassifikation
# --------------------------------------------------------------------------- #
def classify_component(net, comp, df):
    """Liefert ({scope: Index}, Index der Assets mit fehlendem Bus).

    Zweipolige Komponenten: beide Enden zählen.
      ac_de:      beide Enden AC und in DE
      ac_foreign: AC und mindestens ein Ende nicht in DE
      other_de / other_foreign: wie oben, aber für Nicht-AC (z.B. Gas, H2, Wärme)
    Links: carrier == "DC" zählt zusätzlich immer als AC-/Strom-Scope.
    Fehlende Buses (z.B. entfernte Auslands-Buses) werden als AC + Ausland
    gewertet und zusätzlich separat als "dangling" gemeldet.
    """
    empty = pd.Index([])
    if df is None or df.empty:
        return {k: empty for k in SCOPES}, empty

    dangling = pd.Series(False, index=df.index)
    carriers, countries = [], []

    if comp == "Bus":
        carriers.append(df["carrier"])
        countries.append(
            df["country"].fillna("") if "country" in df else pd.Series("", index=df.index)
        )
    else:
        cols = BUS_COLS[comp]
        if not all(c in df.columns for c in cols):
            return {k: empty for k in SCOPES}, empty
        for col in cols:
            b = df[col]
            missing = ~b.isin(net.buses.index)
            dangling |= missing
            carriers.append(b.map(net.buses["carrier"]).mask(missing, "AC"))
            if "country" in net.buses:
                country = b.map(net.buses["country"]).fillna("")
            else:
                country = pd.Series("", index=df.index)
            countries.append(country.mask(missing, MISSING_BUS))

    both_ac = pd.Series(True, index=df.index)
    in_de = pd.Series(True, index=df.index)
    for c in carriers:
        both_ac &= c == "AC"
    for c in countries:
        in_de &= c == "DE"

    if comp == "Link" and "carrier" in df.columns:
        # Bewusst "=" statt "|": Nur DC-Links gelten als AC-Scope.
        both_ac = df["carrier"] == "DC"

    masks = {
        "all": pd.Series(True, index=df.index),
        "ac": both_ac,
        "ac_de": both_ac & in_de,
        "ac_foreign": both_ac & ~in_de,
        "other": ~both_ac,
        # Nicht-AC nach Land der Bus-Enden (mind. ein Ende im Ausland -> Ausland)
        "other_de": ~both_ac & in_de,
        "other_foreign": ~both_ac & ~in_de,
    }
    return {k: df.index[m.to_numpy()] for k, m in masks.items()}, df.index[dangling.to_numpy()]


def classify_net(net):
    scopes, dangling = {}, {}
    for comp, attr in COMPONENTS.items():
        scopes[comp], dangling[comp] = classify_component(net, comp, getattr(net, attr, None))
    return scopes, dangling


# --------------------------------------------------------------------------- #
# p_set / p
# --------------------------------------------------------------------------- #
def p_set_ts(net, comp):
    attr = DISPATCH[comp][0]
    return getattr(net, attr + "_t").get("p_set", pd.DataFrame())


def effective_p_set(net, comp, ids):
    """Statischer p_set als Default, überschrieben durch vorhandene Zeitreihe.
    Nur für die übergebenen Assets (spart Speicher)."""
    attr = DISPATCH[comp][0]
    df = getattr(net, attr)
    ids = pd.Index(ids)

    if "p_set" in df.columns:
        static = df["p_set"].reindex(ids).to_numpy(dtype=float)
    else:
        static = np.zeros(len(ids))
    eff = pd.DataFrame(
        np.tile(static, (len(net.snapshots), 1)), index=net.snapshots, columns=ids
    )

    ts = p_set_ts(net, comp)
    common = ts.columns.intersection(ids)
    if len(common):
        eff[common] = ts[common]
    return eff


# --------------------------------------------------------------------------- #
# Prüfungen je Stufe
# --------------------------------------------------------------------------- #
def print_overview(rep, net, scopes, dangling):
    counts = pd.DataFrame(
        {SCOPES[s]: {comp: len(scopes[comp][s]) for comp in COMPONENTS} for s in SCOPES}
    )
    print("\n  Komponentenübersicht:")
    print("\n".join("    " + line for line in counts.to_string().splitlines()))
    print()

    for comp in COMPONENTS:
        s = scopes[comp]
        if len(s["ac_de"]) + len(s["ac_foreign"]) != len(s["ac"]) or \
                len(s["ac"]) + len(s["other"]) != len(s["all"]) or \
                len(s["other_de"]) + len(s["other_foreign"]) != len(s["other"]):
            rep.emit(comp, False, "Scopes nicht konsistent (AC-DE + AC-Ausland != AC, AC + Nicht-AC != Gesamt bzw. Nicht-AC DE + Ausland != Nicht-AC)", 4)

    n_dangling = {c: len(d) for c, d in dangling.items() if len(d)}
    if n_dangling:
        for comp, n in n_dangling.items():
            rep.emit(comp, False, f"{n} Assets verweisen auf nicht existierende Buses, z.B. {ids_str(dangling[comp])}", 4)
    else:
        print("    ✓ keine Verweise auf fehlende Buses")

    if "country" in net.buses:
        buses = net.buses
        no_country = buses.index[(buses["country"].isna() | (buses["country"] == "")).to_numpy()]
        if len(no_country):
            carriers = sorted(buses.loc[no_country, "carrier"].astype(str).unique())
            rep.emit("Bus", False, f"{len(no_country)} Buses ohne country (zählen als Ausland; carrier: {carriers}), z.B. {ids_str(no_country)}", 4)


def check_link_assumption(rep, net):
    """Annahme: Zwischen zwei AC-Buses gibt es ausschließlich DC-Links."""
    links = getattr(net, "links", None)
    if links is None or links.empty or "carrier" not in links.columns:
        return
    carrier0 = links["bus0"].map(net.buses["carrier"])
    carrier1 = links["bus1"].map(net.buses["carrier"])  # fehlende Buses -> NaN
    violating = links.index[((carrier0 == "AC") & (carrier1 == "AC") & (links["carrier"] != "DC")).to_numpy()]
    if len(violating):
        carriers = sorted(links.loc[violating, "carrier"].astype(str).unique())
        rep.emit("Link", False, f"{len(violating)} Nicht-DC-Links zwischen AC-Buses (carrier: {carriers}), z.B. {ids_str(violating)}", 4)
    else:
        print("    ✓ zwischen AC-Buses nur DC-Links")


def check_p_set(rep, net, scopes):
    """Prüft, dass jedes dispatch-relevante Asset einen definierten p_set hat.

    Asset mit Zeitreihe            -> muss NaN-frei sein (sonst Lücke)
    Asset ohne Zeitreihe           -> PyPSA nutzt den statischen p_set
                                      (Default 0). Das ist gültig (z.B. Komponente
                                      in der Woche ungenutzt, Nullreihen werden oft
                                      weggelassen) und wird nur informativ gezählt.
                                      Lücke nur, wenn auch der statische Wert NaN ist.
    """
    for scope in PARTITION:
        print(f"\n  p_set-Vollständigkeit ({SCOPES[scope]}):")
        for comp, (attr, _) in DISPATCH.items():
            ids = scopes[comp][scope]
            if len(ids) == 0:
                rep.emit(comp, None, "keine Assets in diesem Scope")
                continue

            ts = p_set_ts(net, comp)
            no_ts = ids.difference(ts.columns)
            has_ts = ids.intersection(ts.columns)
            vals = ts[has_ts].reindex(net.snapshots)
            nan_assets = has_ts[vals.isna().any().to_numpy()]

            # Assets ohne Zeitreihe: statischer p_set als Fallback
            df = getattr(net, attr)
            if "p_set" in df.columns:
                static = df["p_set"].reindex(no_ts)
            else:
                static = pd.Series(0.0, index=no_ts)
            static_nan = no_ts[static.isna().to_numpy()]
            n_zero = int((static == 0).sum())
            n_nonzero = len(no_ts) - n_zero - len(static_nan)

            n_gaps = len(nan_assets) + len(static_nan)
            msg = f"{len(ids) - n_gaps}/{len(ids)} vollständig"
            if len(no_ts):
                msg += f" | ohne Zeitreihe (statisch): {len(no_ts)} (davon 0: {n_zero}"
                if n_nonzero:
                    nonzero_ids = no_ts[(static.fillna(0) != 0).to_numpy()]
                    msg += f", ≠0: {n_nonzero}, z.B. {ids_str(nonzero_ids)}"
                msg += ")"
            if len(nan_assets):
                msg += f" | NaN in Zeitreihe: {len(nan_assets)}, z.B. {ids_str(nan_assets)}"
            if len(static_nan):
                msg += f" | weder Zeitreihe noch statischer Wert: {len(static_nan)}, z.B. {ids_str(static_nan)}"
            rep.emit(comp, n_gaps == 0, msg)


def check_dispatch(rep, net, scopes, scope_keys=("ac_de", "ac_foreign")):
    """p (bzw. p0) == effektives p_set. Slack-Generatoren sind ausgenommen,
    da sie in der LPF die Bilanz ausgleichen."""
    for scope in scope_keys:
        print(f"\n  p == p_set ({SCOPES[scope]}):")
        for comp, (attr, p_attr) in DISPATCH.items():
            ids = scopes[comp][scope]
            note = ""
            if comp == "Generator" and len(ids) and "control" in net.generators:
                is_slack = (net.generators.loc[ids, "control"] == "Slack").to_numpy()
                if is_slack.any():
                    note = f" ({is_slack.sum()} Slack ausgenommen)"
                    ids = ids[~is_slack]
            if len(ids) == 0:
                rep.emit(comp, None, "keine Assets in diesem Scope")
                continue

            p = getattr(net, attr + "_t").get(p_attr, pd.DataFrame()).reindex(net.snapshots)
            missing_p = ids.difference(p.columns)
            cols = ids.intersection(p.columns)

            problems = []
            if len(missing_p):
                problems.append(f"{p_attr} fehlt: {len(missing_p)}, z.B. {ids_str(missing_p)}")
            if len(cols):
                diff = (effective_p_set(net, comp, cols) - p[cols]).abs()
                nan_assets = cols[diff.isna().any().to_numpy()]
                max_diff = diff.max()  # NaN werden übersprungen -> separat oben
                mism = max_diff.index[(max_diff > ATOL).to_numpy()]
                if len(nan_assets):
                    problems.append(f"NaN in p/p_set: {len(nan_assets)}, z.B. {ids_str(nan_assets)}")
                if len(mism):
                    problems.append(f"p ≠ p_set: {len(mism)}, max Δ={max_diff.max():.4f}, z.B. {ids_str(mism)}")

            if problems:
                rep.emit(comp, False, f"{len(cols)}/{len(ids)} geprüft{note} | " + " | ".join(problems))
            else:
                rep.emit(comp, True, f"p == p_set ({len(cols)}/{len(ids)} Assets geprüft{note})")


def check_expected_empty(rep, scopes, scope_keys):
    for scope in scope_keys:
        n = sum(len(scopes[comp][scope]) for comp in COMPONENTS)
        rep.emit("Erwartung", n == 0, f"Scope '{SCOPES[scope]}' leer (ist: {n})", 2)


# --------------------------------------------------------------------------- #
# Übergänge zwischen Stufen
# --------------------------------------------------------------------------- #
def check_transition(rep, ref, new, ref_scopes, scope_keys, label):
    """Assets der Referenzstufe (in den Scopes) müssen in der neuen Stufe noch
    existieren, und ihr effektives p_set muss unverändert sein."""
    print(f"\n  {label}")
    for comp, attr in COMPONENTS.items():
        ids = reduce(lambda a, b: a.union(b), (ref_scopes[comp][s] for s in scope_keys))
        if len(ids) == 0:
            continue
        lost = ids.difference(getattr(new, attr).index)
        msg = f"{len(ids) - len(lost)}/{len(ids)} erhalten"
        if len(lost):
            msg += f" | verloren, z.B. {ids_str(lost)}"
        rep.emit(comp, len(lost) == 0, msg)

        if comp in DISPATCH:
            common = ids.intersection(getattr(new, attr).index)
            a = effective_p_set(ref, comp, common)
            b = effective_p_set(new, comp, common)
            changed = ((a - b).abs() > ATOL) | (a.isna() != b.isna())
            changed_assets = common[changed.any().to_numpy()] if len(common) else common
            if len(changed_assets):
                rep.emit(comp, False, f"p_set verändert bei {len(changed_assets)}/{len(common)}, z.B. {ids_str(changed_assets)}")


def p_set_totals(net):
    """{(Komponente, Carrier): Summe des effektiven p_set über alle Assets je Snapshot}"""
    out = {}
    for comp, (attr, _) in DISPATCH.items():
        df = getattr(net, attr)
        if df is None or df.empty:
            continue
        if "carrier" in df.columns:
            carriers = df["carrier"].fillna("").astype(str)
        else:
            carriers = pd.Series("", index=df.index)
        for carrier, ids in carriers.groupby(carriers).groups.items():
            out[(comp, carrier)] = effective_p_set(net, comp, ids).sum(axis=1)
    return out


def check_p_set_totals(rep, ref, new, label, allow_changes=()):
    """Vergleicht die Summe des effektiven p_set je (Komponente, Carrier) pro Snapshot.
    Ein Carrier, der in einer Stufe fehlt, zählt als Summe 0 (harmlos, wenn alles 0 war).
    allow_changes: (Komponente, Carrier) die sich erwartungsgemäß ändern -> nur Info.
    Achtung bei Links: p_set bezieht sich auf p0; wenn das Clustering bus0/bus1
    vertauscht, kann sich das Vorzeichen der Summe ändern."""
    print(f"\n  {label}")
    a, b = p_set_totals(ref), p_set_totals(new)
    for comp in DISPATCH:
        keys = sorted(k for k in a.keys() | b.keys() if k[0] == comp)
        if not keys:
            continue
        bad, expected = [], []
        for key in keys:
            x = a.get(key)
            y = b.get(key)
            note = ""
            if x is None:
                x, note = pd.Series(0.0, index=y.index), " (Carrier nur in neuer Stufe)"
            if y is None:
                y, note = pd.Series(0.0, index=x.index), " (Carrier fehlt in neuer Stufe)"
            d = (x - y.reindex(x.index)).abs()
            tol = ATOL + 1e-6 * max(x.abs().max(), y.abs().max(), 0.0)
            if d.isna().any() or d.max() > tol:
                text = f"{key[1]!r}: Σ {x.sum():.1f} -> {y.sum():.1f}, max Δ pro Snapshot {d.max():.3f}{note}"
                (expected if key in allow_changes else bad).append(text)
        n_ok = len(keys) - len(bad) - len(expected)
        rep.emit(comp, not bad, f"{n_ok}/{len(keys)} Carrier mit erhaltener Summe" + (f", {len(bad)} abweichend:" if bad else ""))
        for t in bad:
            print(f"          ✗ {t}")
        for t in expected:
            print(f"          – erwartet verändert: {t}")


# --------------------------------------------------------------------------- #
# Ablauf
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    key: str
    folder: str
    title: str
    mode: str                 # "pset" oder "dispatch"
    expect_empty: tuple = ()  # Scopes, die leer sein sollen


STAGES = [
    Stage("1", "01-vor-cluster", "vor Clustering", "pset"),
    Stage("2", "02-nach-cluster", "nach Clustering", "pset"),
    Stage("3", "03-mit-dc-results", "nach Clustering mit DC-Results", "pset"),
    Stage("4", "04-vor-lpf", "vor LPF (Ausland entfernt)", "pset", ("ac_foreign", "other_foreign")),
    Stage("5", "05-nach-lpf", "nach LPF (Ausland noch entfernt)", "dispatch", ("ac_foreign", "other_foreign")),
    Stage("6", "06-final-results", "final (Ausland wieder hinzugefügt)", "dispatch"),
    Stage("7", "07-dropped-sectors", "nur AC (andere Sektoren gedroppt)", "dispatch", ("other",)),
]

# (Referenz, neu, Scopes der Referenz, die erhalten bleiben müssen, Label)
TRANSITIONS = [
    ("3", "4", ("ac_de", "other_de"), "3 -> 4: Inland (AC + Nicht-AC) muss erhalten bleiben, Ausland darf entfallen"),
    ("4", "5", ("all",), "4 -> 5: LPF darf nichts entfernen"),
    ("3", "6", ("all",), "3 -> 6: nach Wiederhinzufügen muss alles wieder da sein"),
    ("6", "7", ("ac_de", "ac_foreign"), "6 -> 7: AC muss beim Droppen erhalten bleiben"),
]

# (Referenz, neu, Label, erwartete Änderungen)
TOTAL_TRANSITIONS = [
    ("1", "2", "1 -> 2: Summe p_set je Carrier muss durch Clustering erhalten bleiben", ()),
    ("2", "3", "2 -> 3: nur DC-Links dürfen sich durch den DC-Import ändern", (("Link", "DC"),)),
]


def run_validation(nets):
    rep = Report()
    classified = {}

    ref_snapshots = nets[STAGES[0].key].snapshots
    for stage in STAGES:
        net = nets[stage.key]
        banner(f"Stufe {stage.key}: {stage.title}")
        same = net.snapshots.equals(ref_snapshots)
        rep.emit("Snapshots", same, f"{len(net.snapshots)}" + ("" if same else f" (Stufe 1: {len(ref_snapshots)})"), 2)

        scopes, dangling = classify_net(net)
        classified[stage.key] = scopes
        print_overview(rep, net, scopes, dangling)
        check_link_assumption(rep, net)

        if stage.mode == "pset":
            check_p_set(rep, net, scopes)
        else:
            check_dispatch(rep, net, scopes)
        if stage.expect_empty:
            print()
            check_expected_empty(rep, scopes, stage.expect_empty)

    banner("Übergänge")
    for ref_key, new_key, scope_keys, label in TRANSITIONS:
        check_transition(rep, nets[ref_key], nets[new_key], classified[ref_key], scope_keys, label)

    banner("Summen des effektiven p_set je Komponentenart und Carrier")
    for ref_key, new_key, label, allow in TOTAL_TRANSITIONS:
        check_p_set_totals(rep, nets[ref_key], nets[new_key], label, allow)

    banner(f"Validierung abgeschlossen: {rep.n_fail} Fehlschläge")
    return rep.n_fail


class Tee:
    """Schreibt gleichzeitig auf die Konsole und in einen Puffer."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def save_text_as_pdf(text, path, title="", font_size=7, wrap=175, lines_per_page=58):
    """Schreibt Text in einer Monospace-Schrift (DejaVu Sans Mono, in matplotlib
    enthalten, kennt ✓ ✗ – Δ ≠) als mehrseitiges A4-Querformat-PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    lines = []
    for raw in text.splitlines():
        indent = " " * (len(raw) - len(raw.lstrip()))
        wrapped = textwrap.wrap(
            raw, width=wrap, subsequent_indent=indent + "    ",
            replace_whitespace=False, drop_whitespace=False,
        )
        lines.extend(wrapped or [""])

    pages = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)] or [[]]
    width, height = 11.69, 8.27  # A4 quer
    line_h = font_size * 1.3 / 72 / height

    with PdfPages(path) as pdf:
        for n, page in enumerate(pages, 1):
            fig = Figure(figsize=(width, height))
            y = 1 - 0.5 / height
            for line in page:
                color = "#b00020" if "✗" in line else "black"
                fig.text(0.04, y, line, family="DejaVu Sans Mono", fontsize=font_size,
                         color=color, va="top", ha="left")
                y -= line_h
            fig.text(0.96, 0.02, f"{title}  Seite {n}/{len(pages)}", fontsize=6,
                     family="DejaVu Sans Mono", color="gray", ha="right")
            pdf.savefig(fig)


def run_and_save(nets, output_dir=OUTPUT_DIR):
    """Führt die Validierung aus (Ausgabe auch auf der Konsole) und speichert
    die Ausgabe als PDF im Ordner output_dir. Gibt (n_fail, pdf_path) zurück."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buf)):
        n_fail = run_validation(nets)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_dir / f"validation_report_{stamp}.pdf"
    save_text_as_pdf(buf.getvalue(), pdf_path, title=f"Validierung AC-{RES} ({stamp})")
    print(f"\nReport gespeichert: {pdf_path}")
    return n_fail, pdf_path


def load_networks(base_path=BASE_PATH):
    """Lädt alle Stufen. Rückgabe: {"1": Network, ..., "7": Network}
    (1 = vor Clustering, 2 = nach Clustering, 3 = mit DC-Results, 4 = vor LPF,
    5 = nach LPF, 6 = final, 7 = dropped sectors)."""
    from etrago import Etrago

    return {
        s.key: Etrago(results_folder_name=base_path + s.folder + "/").network
        for s in STAGES
    }


def main():
    nets = load_networks()
    run_and_save(nets, OUTPUT_DIR)
    return nets  # in einer interaktiven Sitzung: nets = main()


if __name__ == "__main__":
    main()
    
'''
import validate_lpf as v

nets = v.main()            # lädt alles, validiert und gibt die Netzwerke zurück
# oder nur laden:
# nets = v.load_networks()

n_vor_cluster  = nets["1"]
n_nach_cluster = nets["2"]
n_dc_results   = nets["3"]
n_vor_lpf      = nets["4"]
n_nach_lpf     = nets["5"]
n_final        = nets["6"]
n_dropped      = nets["7"]
'''
