# ═══════════════════════════════════════════════════════════════
#  KALKULATOR GEODEZYJNY
#  Politechnika Morska w Szczecinie | Geoinformatyka | PiG
#  Autorzy: [I.I. 1], [J.D. 2], [S.K. 3]
# ═══════════════════════════════════════════════════════════════
#URUCHOMIENIE apki i lokalnego hosta venv
#Lokalnie:
#1. Zainstaluj biblioteki: `pip install -r requirements.txt`
#2. Uruchom: `streamlit run app.py`
#3.\venv\Scripts\Activate.ps1
#4. Otwórz przeglądarkę: `localhost:8501`
#Na telefonie (ta sama sieć Wi-Fi):**
#Wpisz adres IP komputera zamiast `localhost`.

import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OPCJE = [
    "Odległość między punktami",
    "Azymut kierunku",
    "Pole powierzchni wieloboku",
    "Transformacja biegunowe ↔ prostokątne",
    "Wcięcie liniowe",
    "Wcięcie kątowe w przód"
]

# ── Konfiguracja strony ──────────────────────────────────────
st.set_page_config(
    page_title="Kalkulator Geodezyjny",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Style CSS ────────────────────────────────────────────────

st.markdown("""
<style>

/* ───────── OGÓLNE ───────── */
.main { padding-top: 0.5rem; }

h1 {
    font-size: 1.6rem !important;
    margin-bottom: 0 !important;
}

h2 {
    font-size: 1.2rem !important;
    color: #1e40af;
}

h3 {
    font-size: 1.1rem !important;
    margin: 0.4rem 0 0.2rem 0 !important;
}

/* ───────── PRZYCISKI ───────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
}

/* ───────── ALERT – BAZA (tylko wygląd kontenera!) ───────── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    padding: 0.4rem 0.6rem !important;
}

/* ───────── SUCCESS (wyniki) ───────── */
div[data-testid="stAlert"][kind="success"] p:first-child {
    font-size: 0.95rem !important;
    color: #94a3b8 !important;
    text-align: left !important;
    margin-bottom: 0.2rem !important;
}

div[data-testid="stAlert"][kind="success"] p:last-child {
    font-size: 1.7rem !important;
    font-weight: 600 !important;
    text-align: center !important;
    margin: 0 !important;
}

/* fallback gdy użyjesz ### */
div[data-testid="stAlert"][kind="success"] h3 {
    font-size: 1.7rem !important;
    text-align: center !important;
    margin: 0 !important;
}

/* ───────── INFO (hinty / pomoc) ───────── */
div[data-testid="stAlert"][kind="info"] p {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    text-align: center !important;
}

/* ───────── ERROR ───────── */
div[data-testid="stAlert"][kind="error"] p {
    font-size: 1rem !important;
    text-align: center !important;
    font-weight: 500;
}

/* ───────── CAPTION ───────── */
.stCaption {
    font-size: 0.85rem !important;
    line-height: 1.2 !important;
    margin-bottom: 0 !important;
}

/* ───────── WYKRESY ───────── */
.stPyplot {
    display: flex;
    justify-content: center;
}

/* ───────── LISTY ───────── */
ul { margin-top: 0 !important; }
li { margin-bottom: 0.05rem !important; }

/* ───────── KATEX ───────── */
.katex-display {
    margin: 0.4em 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# FUNKCJE
# ═══════════════════════════════════════════════════════════════

def odleglosc(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def azymut(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        raise ValueError("Punkty są identyczne – brak kierunku.")
    
    # math.atan2 zwraca radiany.
    az_grad = math.atan2(dy, dx) * (200.0 / math.pi)
    
    # Normalizacja do zakresu 0-400 gon
    return az_grad % 400.0

def pole_gaussa(punkty):
    n = len(punkty)
    if n < 3:
        raise ValueError("Wielobok musi mieć co najmniej 3 wierzchołki.")
    return abs(sum(
        punkty[i][0] * punkty[(i + 1) % n][1] -
        punkty[(i + 1) % n][0] * punkty[i][1]
        for i in range(n)
    )) / 2

def transformacja_geodezyjna(d=None, az_g=None, dX=None, dY=None, x0=None, y0=None):
    wynik = {}

    if d is not None and az_g is not None:

        a = az_g * math.pi / 200.0
        dX = d * math.sin(a)
        dY = d * math.cos(a)

        wynik["dX"] = dX
        wynik["dY"] = dY

        if x0 is not None and y0 is not None:
            wynik["X2"] = x0 + dX
            wynik["Y2"] = y0 + dY

        return wynik

    elif dX is not None and dY is not None:
        d = math.sqrt(dX**2 + dY**2)
        if d == 0:
            raise ValueError("Oba przyrosty są zerowe.")

        az = math.atan2(dX, dY) * (200.0 / math.pi)

        wynik["d"] = d
        wynik["az_g"] = az % 400.0

        if x0 is not None and y0 is not None:
            wynik["X2"] = x0 + dX
            wynik["Y2"] = y0 + dY

        return wynik

    else:
        raise ValueError("Podaj (d i az_g) albo (dX i dY).")

def wciecie_liniowe(xA, yA, xB, yB, dA, dB):
    dAB = math.sqrt((xB - xA)**2 + (yB - yA)**2)
    
    if dAB == 0:
        raise ValueError("Punkty A i B są identyczne.")
    if dA + dB < dAB or abs(dA - dB) > dAB:
        raise ValueError("Odległości nie tworzą trójkąta - brak przecięcia okręgów.")

    # 1. Obliczamy kąt alfa przy punkcie A z twierdzenia cosinusów
    cos_alfa = (dA**2 + dAB**2 - dB**2) / (2 * dA * dAB)
    cos_alfa = max(-1.0, min(1.0, cos_alfa)) # Zabezpieczenie przed błędami float
    alfa = math.acos(cos_alfa)

    # 2. Obliczamy azymut bazy AB
    az_AB = math.atan2(yB - yA, xB - xA) # Zwraca kąt w radianach

    rozw = []
    # 3. Wyznaczamy dwa rozwiązania: lewe i prawe względem bazy
    # Dodajemy i odejmujemy kąt alfa od azymutu bazy
    for znak in [1, -1]:
        kat_P = az_AB + (znak * alfa)
        xP = xA + dA * math.cos(kat_P)
        yP = yA + dA * math.sin(kat_P)
        rozw.append((xP, yP))
        
    return rozw

def dms(deg):
    d = int(deg)
    m = int((deg - d) * 60)
    s = ((deg - d) * 60 - m) * 60
    return f"{d}° {m}' {s:.2f}\""

def wciecie_katowe_wprzod(xA, yA, xB, yB, alfa_deg, beta_deg):
    # zamiana na radiany
    alfa = math.radians(alfa_deg)
    beta = math.radians(beta_deg)

    # zabezpieczenia
    if abs(math.tan(alfa)) < 1e-12 or abs(math.tan(beta)) < 1e-12:
        raise ValueError("Kąt bliski 0° – niepoprawna geometria.")

    cot_alfa = 1 / math.tan(alfa)
    cot_beta = 1 / math.tan(beta)

    denom = cot_alfa + cot_beta
    if abs(denom) < 1e-12:
        raise ValueError("Suma kątów daje układ nieoznaczony.")

    Xp = (xA * cot_beta + yA + xB * cot_alfa - yB) / denom
    Yp = (-xA + yA * cot_beta + xB + yB * cot_alfa) / denom

    return Xp, Yp

# ═══════════════════════════════════════════════════════════════
# FUNKCJE DO INSTRUKCJI
# ═══════════════════════════════════════════════════════════════
def instrukcja_odleglosc():
    st.markdown("""
Wyznaczenie odległości między dwoma punktami w układzie prostokątnym.

**Wzór:**
$$ d = \\sqrt{(X_2-X_1)^2 + (Y_2-Y_1)^2}$$

**Dane wejściowe:**
- współrzędne punktu P1 (X₁, Y₁)
- współrzędne punktu P2 (X₂, Y₂)

**Wynik:** odległość d [m], przyrosty ΔX, ΔY
""")
    
def instrukcja_azymut():
    st.markdown("""
Wyznaczenie azymutu kierunku z punktu **P1 → P2**.

**Wzór:**

$$ A = \\arctan2(\\Delta X, \\Delta Y) $$

**Dane wejściowe:**
- współrzędne punktu początkowego P1
- współrzędne punktu końcowego P2

**Wynik:** azymut w gradach i stopniach, długość odcinka

**Uwagi:**
- kierunek liczony od osi północy
- zakres: 0–400ᵍ
- dla punktów identycznych azymut jest nieokreślony
""")
    
def instrukcja_pole():
    st.markdown("""
Obliczenie pola powierzchni wieloboku na podstawie współrzędnych punktów.

**Wzór Gaussa (sznurowy):**

$$ P = \\frac{1}{2} \\left| \\sum (x_i y_{i+1} - x_{i+1} y_i) \\right| $$

**Dane wejściowe:**
- współrzędne punktów w kolejności obiegu

**Warunki:**
- minimum 3 punkty
- wielobok nie może się przecinać

**Wynik:** pole w metrach kwadratowych [m²], w hektarach [ha], liczba punktów
""")
    
def instrukcja_transformacja():
    st.markdown("""
Transformacja między współrzędnymi biegunowymi i prostokątnymi.

**Biegunowe → Prostokątne**

$$ \\Delta X = d \\cdot \\sin(A) $$

$$ \\Delta Y = d \\cdot \\cos(A) $$

**Prostokątne → Biegunowe**

$$ d = \\sqrt{\\Delta X^2 + \\Delta Y^2} $$

$$ A = \\arctan2(\\Delta X, \\Delta Y) $$

**Dane wejściowe:**
- odległość i azymut lub przyrosty ΔX, ΔY
- opcjonalnie punkt początkowy (X₀, Y₀)

**Wynik:** przyrosty lub odległość i azymut, opcjonalnie współrzędne punktu końcowego

**Uwagi:**
- azymut w gradach
- kierunek liczony od północy
- ΔX = ΔY = 0 → brak kierunku
""")
    
def instrukcja_wciecie_liniowe():
    st.markdown("""
Wyznaczenie punktu **P** na podstawie dwóch punktów **A**, **B** oraz odległości **dA** i **dB**.

**Wzory:**

$$ d_{AB} = \\sqrt{(X_B - X_A)^2 + (Y_B - Y_A)^2} $$

$$ \\cos \\alpha = \\frac{d_A^2 + d_{AB}^2 - d_B^2}{2 d_A d_{AB}} $$

$$ \\alpha = \\arccos(\\cos \\alpha) $$

**Dane wejściowe:**
- współrzędne punktów A i B
- odległości dA, dB

**Wynik:** współrzędne dwóch punktów P₁ i P₂

**Uwagi:**
- istnieją **2 rozwiązania**
- brak rozwiązania gdy: dA + dB < AB lub |dA − dB| > AB
- jedno rozwiązanie gdy okręgi są styczne
""")
    
def instrukcja_wciecie_katowe():
    st.markdown("""
Wyznaczenie punktu **P** na podstawie punktów **A**, **B** oraz kątów **α** i **β**.

**Schemat:**
- α – kąt przy punkcie A
- β – kąt przy punkcie B

**Wzory:**

$$ X_P = \\frac{X_A \\cdot \\cot\\beta + Y_A + X_B \\cdot \\cot\\alpha - Y_B}{\\cot\\alpha + \\cot\\beta} $$

$$ Y_P = \\frac{-X_A + Y_A \\cdot \\cot\\beta + X_B + Y_B \\cdot \\cot\\alpha}{\\cot\\alpha + \\cot\\beta} $$

**Dane wejściowe:**
- współrzędne punktów A i B
- kąty α i β [°]

**Wynik:** współrzędne punktu P (X, Y)

**Uwagi:**
- warunek: **α + β < 180°**
- wrażliwe na błędy pomiarowe
""")
    
# ═══════════════════════════════════════════════════════════════
# WYKRESY
# ═══════════════════════════════════════════════════════════════

def rysuj_azymut(x1, y1, x2, y2, az_deg):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor("#ffffff")
    
    # 1. Obliczanie zakresu
    dx, dy = y2 - y1, x2 - x1
    dlugosc = math.sqrt(dx**2 + dy**2)
    mid_y, mid_x = (y1 + y2) / 2, (x1 + x2) / 2
    
    # Stabilny margines - nie pozwalamy mu urosnąć za bardzo
    view_margin = max(min(dlugosc * 0.4, 30.0), 10.0) 
    
    ax.set_xlim(min(y1, y2) - view_margin, max(y1, y2) + view_margin)
    ax.set_ylim(min(x1, x2) - view_margin, max(x1, x2) + view_margin)
    ax.set_aspect('equal', adjustable='box')

    # 2. Linia kierunku - stały rozmiar kropki
    ax.plot([y1, y2], [x1, x2], color="#1d4ed8", marker="o", lw=1.5, ms=5, zorder=10)

    # 3. Strzałka Północy - stała długość (np. 15 metrów lub mniej jeśli odcinek krótki)
    n_len = min(view_margin * 0.7, 15.0)
    ax.annotate("", xy=(y1, x1 + n_len), xytext=(y1, x1),
                arrowprops=dict(arrowstyle="-|>, head_length=0.5, head_width=0.25", 
                                color="red", lw=1.2),
                zorder=11)
    ax.text(y1, x1 + n_len + (view_margin * 0.05), "N", 
            color="red", fontsize=9, ha="center", fontweight="bold")

    # 4. Łuk azymutu - STAŁY PROMIEŃ (niezależny od długości P1-P2)
    # Ustawiamy promień na sztywno (np. 8-10 metrów), chyba że odcinek jest bardzo krótki
    r_arc = min(dlugosc * 0.4, 10.0)
    
    arc = mpatches.Arc((y1, x1), 2*r_arc, 2*r_arc, angle=0,
                    theta1=90 - az_deg, 
                    theta2=90, 
                    color="#16a34a", lw=1.8, zorder=12)
    ax.add_patch(arc)

    # 5. Wartość kąta - blisko łuku
    angle_rad = math.radians(90 - az_deg / 2)
    # Stałe przesunięcie tekstu od środka łuku
    tx = y1 + (r_arc + 3.0) * math.cos(angle_rad)
    ty = x1 + (r_arc + 3.0) * math.sin(angle_rad)
    az_g_label = az_deg / 0.9 
    ax.text(tx, ty, f"{az_g_label:.2f}ᵍ", color="#16a34a", fontsize=9, fontweight="bold", ha="center", va="center")

    # 6. Etykiety punktów
    ax.text(y1, x1 - (view_margin * 0.1), "P1", fontsize=10, fontweight="bold", ha="center", va="top")
    ax.text(y2, x2 + (view_margin * 0.1), "P2", fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.grid(True, alpha=0.1, ls='--')
    fig.tight_layout()
    
    return fig

def rysuj_wciecie(xA, yA, xB, yB, dA, dB, rozwiazania):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f8fafc")
    ax.set_aspect('equal', adjustable='datalim')

    ax.plot([yA, yB], [xA, xB], "k-o", lw=1.5, ms=5, label="Baza A-B", zorder=5)
    ax.text(yA, xA, " A", va='bottom', ha='right', fontweight='bold', fontsize=10)
    ax.text(yB, xB, " B", va='bottom', ha='left', fontweight='bold', fontsize=10)
    
    ax.add_patch(plt.Circle((yA, xA), dA, fill=False, color='blue', ls='--', alpha=0.3))
    ax.add_patch(plt.Circle((yB, xB), dB, fill=False, color='purple', ls='--', alpha=0.3))

    kolory = ["#16a34a", "#dc2626"]
    for i, (p, kol) in enumerate(zip(rozwiazania, kolory)):
        px, py = p
        ax.plot([yA, py], [xA, px], color=kol, ls="--", lw=1, alpha=0.6)
        ax.plot([yB, py], [xB, px], color=kol, ls=":", lw=1, alpha=0.6)
        ax.plot(py, px, "o", color=kol, ms=5, zorder=10)
        ax.text(py, px, f" P{i+1}\n X:{px:.2f}\n Y:{py:.2f}", 
                color=kol, fontsize=9, fontweight='bold', va='bottom')

    ax.relim()
    ax.autoscale_view()
    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.grid(True, alpha=0.2)
    
    return fig

def rysuj_wielobok(punkty):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor("#f8fafc")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)

    ys = [p[1] for p in punkty]
    xs = [p[0] for p in punkty]
    margin = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.10

    # zamknięty wielobok
    xs_closed = xs + [xs[0]]
    ys_closed = ys + [ys[0]]
    ax.fill(ys_closed, xs_closed, alpha=0.25, color="#719ce2")
    ax.plot(ys_closed, xs_closed, "b-o", lw=2, ms=5, zorder=4)
    
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    for i, (x, y) in enumerate(zip(xs, ys)):
        dx = y - cy
        dy = x - cx
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            length = 1
        dx /= length
        dy /= length
        offset = margin * 0.25

        ax.text(y + dx * offset, x + dy * offset, f"P{i+1}", fontsize=9, fontweight="bold",
                color="#1e3a5f", zorder=10, ha="center", va="center")
    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def rysuj_transformacje(x0, y0, dx, dy, az_g=None):
    if x0 is None or y0 is None:
        return None  # brak wykresu
    
    fig, ax = plt.subplots(figsize=(5, 5))

    x1 = x0 + dx
    y1 = y0 + dy

    # ── 1. WYZNACZENIE ZAKRESU ──
    xmin = min(x0, x1)
    xmax = max(x0, x1)
    ymin = min(y0, y1)
    ymax = max(y0, y1)

    width = ymax - ymin
    height = xmax - xmin

    margin = 0.15
    size = max(width, height, 6) * (1 + margin)

    cx = (ymin + ymax) / 2
    cy = (xmin + xmax) / 2
    half = size / 2

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect('equal')

    # ── 2. STYL ───────────────────────────────────────────
    ax.set_xlabel("Y [m]", fontsize=9)
    ax.set_ylabel("X [m]", fontsize=9)
    ax.tick_params(labelsize=8)

    # punkty
    ax.plot(y0, x0, 'ko', ms=4)
    ax.text(y0, x0, "P0", fontsize=8, ha="right", va="top")

    ax.plot(y1, x1, 'ro', ms=4)
    ax.text(y1, x1, "P2", fontsize=8, ha="left", va="bottom")

    # ── 3. WEKTOR (lepszy niż arrow → annotate) ───────────
    ax.annotate(
        "",
        xy=(y1, x1),
        xytext=(y0, x0),
        arrowprops=dict(
            arrowstyle="-|>",
            lw=1.5,
            color="blue",
            mutation_scale=12 
        )
    )

    # linie pomocnicze
    ax.axhline(x0, ls="--", alpha=0.2, lw=0.8)
    ax.axvline(y0, ls="--", alpha=0.2, lw=0.8)

    # łuk (opcjonalnie)
    if az_g is not None:
        r = size * 0.15
        arc = mpatches.Arc(
            (y0, x0),
            2*r, 2*r,
            theta1=0,
            theta2=az_g * 0.9,
            color="green",
            lw=1
        )
        ax.add_patch(arc)

    ax.grid(True, alpha=0.2)

    return fig

def rysuj_wciecie_katowe_wprzod(xA, yA, xB, yB, alfa_deg, beta_deg, Xp, Yp):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f8fafc")
    ax.set_aspect('equal', adjustable='datalim')

    # ── styl etykiet ─────────────────────────
    bbox_style = dict(
        boxstyle="round,pad=0.25",
        fc="white",
        ec="none",
        alpha=0.9
    )

    # ── baza A-B ─────────────────────────────
    ax.plot([yA, yB], [xA, xB], "k-o", lw=1.5, ms=5, zorder=5)

    # ── punkt P ──────────────────────────────
    ax.plot(Yp, Xp, "o", color="#16a34a", ms=6, zorder=10)

    # ── linie celowe ─────────────────────────
    ax.plot([yA, Yp], [xA, Xp], color="#1d4ed8", lw=1.6)
    ax.plot([yB, Yp], [xB, Xp], color="#9333ea", lw=1.6)

    # ── etykiety punktów ─────────────────────
    # ── środek układu (do odsuwania etykiet) ──
    cx = (xA + xB + Xp) / 3
    cy = (yA + yB + Yp) / 3

    def offset_point(x, y, scale=1.2):
        dx = y - cy
        dy = x - cx
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return y, x
        dx /= length
        dy /= length
        return y + dx * scale, x + dy * scale


    # ── etykiety ─────────────────────
    yA_off, xA_off = offset_point(xA, yA)
    ax.text(yA_off, xA_off, "A", fontsize=10, fontweight="bold", ha="center", va="center", bbox=bbox_style)

    yB_off, xB_off = offset_point(xB, yB)
    ax.text(yB_off, xB_off, "B", fontsize=10, fontweight="bold", ha="center", va="center", bbox=bbox_style)

    yP_off, xP_off = offset_point(Xp, Yp)
    ax.text(yP_off, xP_off, f"P\nX:{Xp:.2f}\nY:{Yp:.2f}", fontsize=9, fontweight="bold", color="#16a34a",
                ha="center", va="center", bbox=bbox_style)


    # ── etykiety kątów (odsunięte dodatkowo) ──
    yA_ang, xA_ang = offset_point(xA, yA, scale=2.0)
    ax.text(yA_ang, xA_ang, f"α={alfa_deg:.1f}°", color="#1d4ed8", fontsize=9, ha="center", bbox=bbox_style)

    yB_ang, xB_ang = offset_point(xB, yB, scale=2.0)
    ax.text(yB_ang, xB_ang, f"β={beta_deg:.1f}°", color="#9333ea", fontsize=9, ha="center", bbox=bbox_style)

    # ── autoskalowanie ───────────────────────
    ax.relim()
    ax.autoscale_view()

    # ── osie i siatka ────────────────────────
    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════
# NAGŁÓWEK APLIKACJI
# ═══════════════════════════════════════════════════════════════

st.title("Kalkulator Geodezyjny")
st.caption("Politechnika Morska w Szczecinie | Geoinformatyka | PiG | 2026")
st.caption("Autorzy: [I.I. 1], [J.D. 2], [S.K. 3]")

st.divider()

# ─── Wybór funkcji ────────────────────────────────────────────
if "funkcja" not in st.session_state:
    st.session_state.funkcja = OPCJE[0]

st.markdown("## Funkcje kalkulatora:")

cols = st.columns(2)

for i, opcja in enumerate(OPCJE):
    if cols[i % 2].button(opcja, use_container_width=True):
        st.session_state.funkcja = opcja

funkcja = st.session_state.funkcja

# ═══════════════════════════════════════════════════════════════
# FUNKCJA 1 – ODLEGŁOŚĆ
# ═══════════════════════════════════════════════════════════════
if funkcja.startswith("Odległość między punktami"):
    st.subheader("📏 Odległość między punktami")

    with st.expander("Instrukcja"):
        instrukcja_odleglosc()

    c1, c2 = st.columns(2)
    x1 = c1.number_input("X₁ [m]", value=629663.67 , format="%.3f",
                          help="Współrzędna X pierwszego punktu")
    y1 = c2.number_input("Y₁ [m]", value=200851.91, format="%.3f",
                          help="Współrzędna Y pierwszego punktu")
    x2 = c1.number_input("X₂ [m]", value=629701.07, format="%.3f",
                          help="Współrzędna X drugiego punktu")
    y2 = c2.number_input("Y₂ [m]", value=200887.13, format="%.3f",
                          help="Współrzędna Y drugiego punktu")

    if st.button("Oblicz", type="primary",
                 use_container_width=True):
        d = odleglosc(x1, y1, x2, y2)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        st.success(f"Odległość = {d:.3f} m")
        c1, c2 = st.columns(2)
        c1.success(f"Przyrost ΔX:\n{dx:.3f} m")
        c2.success(f"Przyrost ΔY:\n{dy:.3f} m")

# ═══════════════════════════════════════════════════════════════
# FUNKCJA 2 – AZYMUT
# ═══════════════════════════════════════════════════════════════
elif funkcja.startswith("Azymut kierunku"):
    st.subheader("🧭 Azymut kierunku")
    with st.expander("Instrukcja"):
        instrukcja_azymut()

    c1, c2 = st.columns(2)
    x1 = c1.number_input("X₁ [m]", value=10.0, format="%.4f")
    y1 = c2.number_input("Y₁ [m]", value=10.0, format="%.4f")
    x2 = c1.number_input("X₂ [m]", value=30.0, format="%.4f")
    y2 = c2.number_input("Y₂ [m]", value=50.0, format="%.4f")

    if st.button("Oblicz azymut", type="primary", use_container_width=True):
        try:
            az_g = azymut(x1, y1, x2, y2)
            d = odleglosc(x1, y1, x2, y2)
            
            # 2. Przeliczenie na STOPNIE do rysowania
            az_deg = az_g * 0.9
            
            c1, c2, c3 = st.columns(3)
            c1.success(f"Grady:\n{az_g:.4f}")
            c2.success(f"Stopnie:\n{az_deg:.2f}")
            c3.success(f"Odległość:\n{d:.3f} m")
            
            # 4. Wykres
            with st.expander("Prezentacja graficzna", expanded=False):
                fig = rysuj_azymut(x1, y1, x2, y2, az_deg)
                st.pyplot(fig, use_container_width=False)
                
        except ValueError as e:
            st.error(str(e))

# ═══════════════════════════════════════════════════════════════
# FUNKCJA 3 – POLE WIELOBOKU
# ═══════════════════════════════════════════════════════════════

elif funkcja.startswith("Pole powierzchni wieloboku"):
    st.subheader("📐 Pole powierzchni wieloboku")
    with st.expander("Instrukcja"):
        instrukcja_pole()

    st.caption("Wprowadź współrzędne w tabeli.")

    if "poly_df" not in st.session_state:
        st.session_state.poly_df = pd.DataFrame({
            "X": [629663.67, 629652.71, 629690.14, 629701.07],
            "Y": [200851.91, 200863.73, 200898.51, 200887.13]
        })

    df = st.session_state.poly_df.copy()
    df.insert(0, "P", [f"P{i+1}" for i in range(len(df))])

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "P": st.column_config.TextColumn("Punkt", disabled=True),
            "X": st.column_config.NumberColumn("X [m]", format="%.3f"),
            "Y": st.column_config.NumberColumn("Y [m]", format="%.3f"),
        },
        key="poly_editor"
    )

    st.session_state.poly_df = edited_df[["X", "Y"]]

    c1, c2 = st.columns(2)

    if c1.button("Wczytaj przykład", use_container_width=True):
        st.session_state.poly_df = pd.DataFrame({
            "X": [629663.67, 629652.71, 629690.14, 629701.07],
            "Y": [200851.91, 200863.73, 200898.51, 200887.13]
        })
        st.rerun()

    if c2.button("Wyczyść", use_container_width=True):
        st.session_state.poly_df = pd.DataFrame({"X": [], "Y": []})
        st.rerun()

    if st.button("Oblicz pole", type="primary", use_container_width=True):
        try:
            df = st.session_state.poly_df.dropna()

            if len(df) < 3:
                st.error("Podaj co najmniej 3 punkty.")
            else:
                pts = df[["X", "Y"]].values.tolist()
                p = pole_gaussa(pts)

                c1, c2, c3 = st.columns(3)
                c1.success(f"Pole:\n {p:.2f} m²")
                c2.success(f"Pole:\n {p/10000:.4f} ha")
                c3.success(f"Liczba pkt.:\n {len(pts)}")

                with st.expander("Prezentacja graficzna"):
                    st.pyplot(rysuj_wielobok(pts))

        except Exception:
            st.error("Błąd danych. Sprawdź współrzędne.")

# ═══════════════════════════════════════════════════════════════
# FUNKCJA 4 – TRANSFORMACJA BIEGUNOWE ↔ PROSTOKĄTNE
# ═══════════════════════════════════════════════════════════════
elif funkcja.startswith("Transformacja biegunowe"):
    st.subheader("🔁Transformacja biegunowe ↔ prostokątne")
    with st.expander("Instrukcja"):
        instrukcja_transformacja()

    tryb = st.radio(
        "Tryb:",
        ["Biegunowe → Prostokątne", "Prostokątne → Biegunowe"]
    )

    use_start = st.checkbox("Uwzględnij punkt początkowy (X₀, Y₀)")

    if use_start:
        c1, c2 = st.columns(2)
        x0 = c1.number_input("X₀ [m]", value=10.0)
        y0 = c2.number_input("Y₀ [m]", value=10.0)
    else:
        x0 = y0 = None

    st.divider()

    # ───── PRZYPADEK 1 ─────
    if tryb.startswith("Biegunowe → Prostokątne"):
        c1, c2 = st.columns(2)
        d = c1.number_input("Odległość [m]", value=12.0)
        az_g = c2.number_input("Azymut [g]", value=67.0)

        if st.button("Oblicz", type="primary", use_container_width=True):
            try:
                res = transformacja_geodezyjna(
                    d=d, az_g=az_g,
                    x0=x0, y0=y0
                )

                c1, c2 = st.columns(2)
                c1.success(f"Przyrost ΔX:\n{res['dX']:.3f} m")
                c2.success(f"Przyrost ΔY:\n{res['dY']:.3f} m")

                if "X2" in res:
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.success(f"Wspołrzędna X₂:\n{res['X2']:.3f} m")
                    c2.success(f"Współrzędna Y₂:\n{res['Y2']:.3f} m")

                if x0 is not None and y0 is not None:
                    with st.expander("Prezentacja graficzna"):
                        fig = rysuj_transformacje(x0, y0, res["dX"], res["dY"])
                        st.pyplot(fig)
                else:
                    st.info("Zaznacz opcję punktu początkowego (X₀, Y₀), aby włączyć wykres.")
            except ValueError as e:
                st.error(str(e))

    # ───── PRZYPADEK 2 ─────
    else:
        c1, c2 = st.columns(2)
        dX = c1.number_input("ΔX [m]", value=15.0)
        dY = c2.number_input("ΔY [m]", value=22.0)

        if st.button("Oblicz", type="primary", use_container_width=True):
            try:
                res = transformacja_geodezyjna(
                    dX=dX, dY=dY,
                    x0=x0, y0=y0
                )

                c1, c2 = st.columns(2)
                c1.success(f"Odległość d:\n{res['d']:.3f} m")
                c2.success(f"Azymut [g]:\n{res['az_g']:.3f}")

                if "X2" in res:
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.success(f"Współrzędna X₂:\n{res['X2']:.3f} m")
                    c2.success(f"Współrzędna Y₂:\n{res['Y2']:.3f} m")

                if x0 is not None and y0 is not None:
                    with st.expander("Prezentacja graficzna"):
                        fig = rysuj_transformacje(x0, y0, dX, dY)
                        st.pyplot(fig)
                else:
                    st.info("Zaznacz opcję punktu początkowego (X₀, Y₀), aby włączyć wykres.")

            except ValueError as e:
                st.error(str(e))


# ═══════════════════════════════════════════════════════════════
# FUNKCJA 5 – WCIĘCIE KĄTOWE W PRZÓD
# ═══════════════════════════════════════════════════════════════
elif funkcja.startswith("Wcięcie kątowe w przód"):
    st.subheader("📐 Wcięcie kątowe w przód")
    with st.expander("Instrukcja"):
        instrukcja_wciecie_katowe()

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Punkt A**")
        xA = st.number_input("XA [m]", value=10.0, key="xa")
        yA = st.number_input("YA [m]", value=10.0, key="ya")
        alfa = st.number_input("α [°]", value=37.0, min_value=0.0, max_value=180.0, key="alfa")

    with colB:
        st.markdown("**Punkt B**")
        xB = st.number_input("XB [m]", value=25.0, key="xb")
        yB = st.number_input("YB [m]", value=27.0, key="yb")
        beta = st.number_input("β [°]", value=69.0, min_value=0.0, max_value=180.0, key="beta")

    if st.button("Oblicz", type="primary", use_container_width=True):
        try:
            gamma = 180 - (alfa + beta)

            if gamma <= 0:
                st.error("Suma kątów ≥ 180° – brak rozwiązania.")
            else:
                Xp, Yp = wciecie_katowe_wprzod(xA, yA, xB, yB, alfa, beta)

                st.success(f"P: X = {Xp:.3f} m, Y = {Yp:.3f} m")

                with st.expander("Prezentacja graficzna"):
                    fig = rysuj_wciecie_katowe_wprzod(
                        xA, yA, xB, yB, alfa, beta, Xp, Yp
                    )
                    st.pyplot(fig)

        except ValueError as e:
            st.error(str(e))

# ═══════════════════════════════════════════════════════════════
# FUNKCJA 6 – WCIĘCIE LINIOWE
# ═══════════════════════════════════════════════════════════════
elif funkcja.startswith("Wcięcie liniowe"):
    st.subheader("📍 Wcięcie liniowe")

    with st.expander("Instrukcja"):
        instrukcja_wciecie_liniowe()

    c1, c2 = st.columns(2)
    xA = c1.number_input("XA [m]", value=0.0,   format="%.4f",
                          help="Współrzędna X punktu osnowy A")
    yA = c2.number_input("YA [m]", value=0.0,   format="%.4f",
                          help="Współrzędna Y punktu osnowy A")
    xB = c1.number_input("XB [m]", value=40.0, format="%.4f",
                          help="Współrzędna X punktu osnowy B")
    yB = c2.number_input("YB [m]", value=40.0,   format="%.4f",
                          help="Współrzędna Y punktu osnowy B")
    dA = c1.number_input("dA – odległość A→P [m]", value=35.0, format="%.4f",
                          help="Odległość zmierzona od punktu A do punktu P")
    dB = c2.number_input("dB – odległość B→P [m]", value=45.0, format="%.4f",
                          help="Odległość zmierzona od punktu B do punktu P")

    if st.button("Oblicz", type="primary",
                 use_container_width=True):
        try:
            rozw = wciecie_liniowe(xA, yA, xB, yB, dA, dB)
            st.success(f"Rozwiązanie 1:  X = {rozw[0][0]:.4f} m,  Y = {rozw[0][1]:.4f} m")
            st.success(f"Rozwiązanie 2:  X = {rozw[1][0]:.4f} m,  Y = {rozw[1][1]:.4f} m")
            st.info("Wybierz rozwiązanie zgodne z lokalizacją punktu w terenie.")
            with st.expander("Prezentacja graficzna"):
                st.pyplot(rysuj_wciecie(xA, yA, xB, yB, dA, dB, rozw))
        except ValueError as e:
            st.error(str(e))
