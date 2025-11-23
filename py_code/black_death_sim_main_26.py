"""
The Black Death occurred in Europe from 1346 to 1353.
"""

import networkx as nx # Tīklu (grafu) bibliotēka: grafa virsotnes - pilsētas, grafa šķautnes - pilsētu savienojumi (tirdzniecības ceļi)
import random
import numpy as np


class City_SIRD:
    """
    S (susceptible) — cilvēki, kas var saslimst
    I (infectious) — inficētie (saslimušie) (tagad slimo)
    R (recovered) — izveseļojušies (nevar nekad vairs saslimst - rezistenti)
    D (dead) — nomirušie
    N (alive) – S+I+R (cilvēki, kas nav nomiruši)

    β (beta) – inficēšanās varbūtība par vienu dienu (lipīgums) (koeficients)
    γ (gamma) – izveseļošanas varbūtība par vienu dienu (koeficients)
    μ (mu) – nomiršanas varbūtība par vienu dienu [tikai tiem, kuri jau ir I (inficētie]
    """

    def __init__(self, population, initial_infected=0, beta=0.099, gamma=0.0225, mu=0.009, name="City", cap_frac=0.08, overload_mult=1.0):
        self.name = name  # pilsētas nosaukums

        # SIRD sākotnējie stāvokļi
        self.S = population - initial_infected
        self.I = initial_infected
        self.R = 0                # katra pilsēta no sākuma nav izveseļoto
        self.D = 0                # no sākuma nav neviena nomiruša

        self.beta = beta          # inficēšanās varbūtība par vienu dienu (lipīgums) (koeficients)
        self.gamma = gamma * random.uniform(0.90, 1.10) #izveseļošanas varbūtība par vienu dienu. random, lai dažādas pilsētas būtu nejauša izveseļošanas varbūtība (gamma) (lai tas atšķīras)
        self.mu = mu * random.uniform(0.85, 1.15) # nomiršanas varbūtība par vienu dienu. random, lai dažādas pilsētas būtu nejauša nomiršanas varbūtība (mu) (lai tas atšķīras)

        # pilsētas "kapacitāte".
        # Tas ir nepieciešams lai realizētu šo ideju: Ja ir pārāk daudz inficēto relatīvi pret dzīviem (I/N), tad palielinas mirstība (μ).
        self.cap_frac = cap_frac

        # cik daudz reižu var palielināties mirstība pie cap_frac pārpildīšanas.
        self.overload_mult = overload_mult

    def state(self):
        # Atgriež pašreizējo stāvokli (noapaļotu uz leju)
        return {"S": int(self.S), "I": int(self.I), "R": int(self.R), "D": int(self.D)}

    def step(self, external_infection_force=0):

        # N - Dzīvie cilvēki (veselie + inficētie + recovered) (mirušie D netiek skaitīti)
        N = self.S + self.I + self.R
        if N <= 0:
            return None # pilsēta izmirusī, nav ko rēķināt - izejam no funkcijas

        # Iekšējais infekcijas spēks: I / N
        internal_force = self.I / N

        # Pievienojam ārējo infekcijas ienesumu
        raw_force = internal_force + external_infection_force

        # Ierobežojam robežās [0, 1]
        if raw_force < 0:
            force = 0
        elif raw_force > 1:
            force = 1
        else:
            force = raw_force

        # Inficēšanās un izveseļošanās koeficienti
        p_inf = self.beta * force
        p_rec = self.gamma

        # Slodze uz pilsētu: cik liela daļa ir inficēti
        load = self.I / N


        # Ja slodze pārspēj kapacitāti, aprēķina pārslodzi. Ja ir pārāk daudz inficēto relatīvi pret dzīviem (I/N), tad palielinas mirstība (μ).
        if load <= self.cap_frac:
            overload = 0.0
        else:
            overload = (load - self.cap_frac) / self.cap_frac

        mu_eff = self.mu * (1 + self.overload_mult * overload)

        if mu_eff > 1:
            p_die = 1
        else:
            p_die = mu_eff

        if p_inf < 0:
            p_inf = 0
        elif p_inf > 1:
            p_inf = 1

        if p_rec < 0:
            p_rec = 0
        elif p_rec > 1:
            p_rec = 1

        if p_die < 0:
            p_die = 0
        elif p_die > 1:
            p_die = 1

        # === JAUNIE INFICĒTIE ===
        S_int = int(self.S)
        new_infected = int(np.random.binomial(S_int, p_inf))

        # === APRĒĶINI I GRUPAI ===
        I_int = int(self.I)

        # p_rec + p_die nedrīkst pārsniegt 1
        s = p_rec + p_die
        if s > 1:
            p_rec = p_rec / s
            p_die = p_die / s

        # p_stay — tie, kas paliek slimi
        p_stay = 1 - p_rec - p_die
        if p_stay < 0:
            p_stay = 0


        # Cik daudz recovered (izveseļojušies)
        rec = np.random.binomial(I_int, p_rec)

        # Palikušie infected
        remaining_after_rec = I_int - rec
        if remaining_after_rec < 0:
            remaining_after_rec = 0

        # Cik daudz nomira no palikušiem infected
        died = np.random.binomial(remaining_after_rec, p_die / (p_die + p_stay))

        # Cik infected palika
        stay = remaining_after_rec - died
        if stay < 0:
            stay = 0


        # === ATJAUNINĀM STĀVOKLI ===
        self.S -= new_infected
        self.I += new_infected - rec - died
        self.R += rec
        self.D += died


        # Noapaļo un novērš negatīvas vērtības (ja mazāks par 0, tad iestatīt uz 0)
        self.S = int(self.S)
        if self.S < 0:
            self.S = 0

        self.I = int(self.I)
        if self.I < 0:
            self.I = 0

        self.R = int(self.R)
        if self.R < 0:
            self.R = 0

        self.D = int(self.D)
        if self.D < 0:
            self.D = 0


def compute_commute_forces(G: nx.Graph, commute_rate=0.001) -> dict[City_SIRD, float]:
    """
    Aprēķina ārējo infekcijas ienesumu katrai pilsētai, balstoties uz tirdzniecību starp pilsētām.
    Populācijas S/I/R netiek mainītas, tikai atgriež ārējo spiedienu (external_infection_force koeficientu).

    IDEJA:
    Katra pilsēta u saņem "ārējo infekcijas spiedienu" no visām kaimiņu pilsētām v. ("var iedomāties kā infekciju staru")
    Pieņēmums: jo spēcīgāka ir saikne v→u (tirdzniecības/intensitātes svars), un jo lielāka inficēto daļa v (I_v/N_v), jo lielāks ir infekcijas ienesums uz u.
    """

    # === REZULTĀTA SĀKOTNĒJĀ TABULA (0 katrai pilsētai) ===
    # Izveidojam tukšu vārdnīcu ārējās infekcijas spiedienam
    external_infection_force = {}

    # Katrai pilsētai piešķiram sākotnējo vērtību 0.0 (external_infection_force)
    for city in G.nodes:
        external_infection_force[city] = 0

    # === 1) APREĶINĀM SUM_W(v) KATRAI PILSĒTAI ===
    # sum_w(v) = visu izejošo šķautņu svaru summa
    total_edge_weight_from_city = {}

    for city in G.nodes: # ejam cauri katrai city in G.nodes
        neighbors = list(G[city].items())  # visi kaimiņi

        if len(neighbors) == 0:
            # Ja nav kaimiņu, nav ārēja spiediena
            total_edge_weight_from_city[city] = 0
        else:
            total = 0.0
            for _, edge_data in neighbors:
                weight = edge_data.get("weight", 1)
                weight = max(weight, 0)  # negatīvs svars nav atļauts
                total += weight

            total_edge_weight_from_city[city] = total


    # === 2) APRĒĶINĀM IENĀKOŠO INFEKCIJAS SPIEDIENI KATRAI PILSĒTAI ===
    for target_city in G.nodes:                        # pilsēta, kas SAŅEM infekciju
        for source_city, edge_data in G[target_city].items():  # Pilsēta, kas NODOD infekciju
            weight_uv = edge_data.get("weight", 1)
            weight_uv = max(weight_uv, 0)

            # Summa visiem source_city savienojumiem
            total_w = total_edge_weight_from_city[source_city]

            # Ja avota pilsētai nav maršrutu (nav transporta) – izlaižam
            if total_w <= 0.0:
                continue

            # Dzīvā populācija avota pilsētā
            N_v = source_city.S + source_city.I + source_city.R
            if N_v <= 0:
                continue  # pilsēta izmirusi

            infected_ratio = source_city.I / N_v  # I_v / N_v

            # Pieskaitām infekcijas ienesumu
            external_infection_force[target_city] += commute_rate * (weight_uv / total_w) * infected_ratio


    # === ATGRIEŽAM APRĒĶINĀTOS ĀRĒJOS SPIEDIENA KOEFICIENTUS (kā dict) ===
    return external_infection_force


def super_commute_spikes(
    G,
    day,
    super_period = 14,       # ik pa N dienām pārbaudām
    super_prob = 0.50,    # varbūtība, ka šajā dienā notikums notiek
    events = 2,             # cik 'lēciens' (v->u) tiks izspēlēti
    k_min = 100,            # kontaktu 'paciņas' minimālais lielums
    k_max = 300,            # kontaktu 'paciņas' maksimālais lielums
    spike_rate = 0.0008,    # mērogs (tās pašas vienības, kas commute_rate)
    rate_mult = 80.0,     # cik reizes stiprāks par parasto ienesumu
    rng = None
) -> dict[City_SIRD, float]:
    """
    Atgriež: {pilsēta: papildus ārējais infekcijas spiediens} konkrētajā dienā.
    S/I/R/D NETIEK mainīti.

    IDEJA:
    - Reti (reizi super_period dienās) un ar varbūtību super_prob notiek 'liels notikums' (jeb šoks) (piem., kuģa ierašanās).
    """

    # Ja nav padots rng, izmantojam standarta random moduli
    if rng is None:
        rng = random

    # Sākumā visām pilsētām papildus spiediens ir 0
    external_spike = {}
    for city in G.nodes:
        external_spike[city] = 0.0


    # === KALENDĀRA UN VARBŪTĪBAS TRIGGERS ===
    # Ja super_period ir 0 vai mazāks, tad "lielie notikumi" nekad nenotiek
    if super_period <= 0:
        return external_spike

    # Ja šī diena nav dalāma ar super_period, tad šodien "lielais notikums" nav paredzēts
    if (day % super_period) != 0:
        return external_spike

    # Pat ja diena der, notikums notiek tikai ar varbūtību super_prob
    if rng.random() >= super_prob:
        return external_spike


    # === KANDIDĀTU APKOPE: v->u ar svariem ~ w * (I_v/N_v) ====
    candidates = []   # saraksts ar (v, u, w)
    weights = []      # atbilstošie svari izvēlei

    for source_city in G.nodes:
        N_v = source_city.S + source_city.I + source_city.R
        if N_v <= 0:
            continue
        if source_city.I <= 0:
            continue

        infected_ratio = source_city.I / N_v  # I_v / N_v

        for target_city, edge_data in G[source_city].items():
            w = edge_data.get("weight", 1)
            w = max(0.0, w)
            if w <= 0.0:
                continue

            candidates.append((source_city, target_city, w))
            weights.append(w * infected_ratio)

    # Ja nav kandidātu
    if len(candidates) < 1:
        return external_spike

    # Saskaitām svarus pavisam vienkārši
    total_weights = 0
    for w in weights:
        total_weights = total_weights + w

    # Ja svars = 0, tad nav iespējams izvēlēties nevienu kandidātu
    if total_weights == 0:
        return external_spike


    jumps_to_draw = int(events)
    if jumps_to_draw < 1:
        jumps_to_draw = 1


    for _ in range(jumps_to_draw):
        # Kopējais svaru lielums
        total_weight = 0.0
        for w in weights:
            total_weight += w

        # Izvēlamies nejaušu skaitli intervālā [0; total_weight)
        r = rng.random() * total_weight

        # Vienkārša loterija: ejam cauri svariem, līdz sasniedzam r
        cum = 0.0
        index = 0
        while index < len(weights):
            cum += weights[index]
            if r <= cum:
                break
            index += 1

        # Paņemam attiecīgo kandidātu
        source_city, target_city, w = candidates[index]


        N_u = target_city.S + target_city.I + target_city.R
        if N_u <= 0.0:
            continue  # mērķis 'izmiris' — šoks nepiemērojas

        contacts_k = rng.randint(k_min, k_max)

        external_spike[target_city] += rate_mult * spike_rate * (contacts_k / N_u)



    # === REZULTĀTS ===
    return external_spike


# ==== IZVEIDOJAM TĪKLU ====

# Vidusjūra
feodosia        = City_SIRD(40000,   initial_infected=100, name="Feodosia")
constantinople  = City_SIRD(150000,  initial_infected=0, name="Constantinople")
thessaloniki    = City_SIRD(120000,  initial_infected=0, name="Thessaloniki")
ragusa          = City_SIRD(30000,   initial_infected=0, name="Ragusa")
venice          = City_SIRD(150000,  initial_infected=0, name="Venice")
genoa           = City_SIRD(90000,   initial_infected=0, name="Genoa")
marseille       = City_SIRD(31000,   initial_infected=0, name="Marseille")
barcelona       = City_SIRD(48000,   initial_infected=0, name="Barcelona")
palermo         = City_SIRD(51000,   initial_infected=0, name="Palermo")
naples          = City_SIRD(55000,   initial_infected=0, name="Naples")
tunis           = City_SIRD(40000,   initial_infected=0, name="Tunis")
cairo           = City_SIRD(400000,  initial_infected=0, name="Cairo")
athens          = City_SIRD(25000,   initial_infected=0, name="Athens")
florence        = City_SIRD(110000,  initial_infected=0, name="Florence")

sarai           = City_SIRD(100000,  initial_infected=0, name="Sarai")

burgos      = City_SIRD(25000,  initial_infected=0, name="Burgos")
bordeaux    = City_SIRD(35000,  initial_infected=0, name="Bordeaux")
montpellier = City_SIRD(35000,  initial_infected=0, name="Montpellier")
toulouse    = City_SIRD(30000,  initial_infected=0, name="Toulouse")


# Baltijas un Ziemeļu jūra jeb sviesta eiropa

riga            = City_SIRD(6500,   initial_infected=0, name="Riga")
vilnius         = City_SIRD(20000,  initial_infected=0, name="Vilnius")
tallinn         = City_SIRD(4000,   initial_infected=0, name="Tallinn")
berlin          = City_SIRD(5500,   initial_infected=0, name="Berlin")
brussels        = City_SIRD(30000,  initial_infected=0, name="Brussels")
danzig          = City_SIRD(20000,  initial_infected=0, name="Danzig")
ghent           = City_SIRD(55000,  initial_infected=0, name="Ghent")
turku           = City_SIRD(5000,   initial_infected=0, name="Turku")
dublin          = City_SIRD(20000,  initial_infected=0, name="Dublin")
oslo            = City_SIRD(5000,   initial_infected=0, name="Oslo")
paris           = City_SIRD(200000, initial_infected=0, name="Paris")
stockholm       = City_SIRD(8000,   initial_infected=0, name="Stockholm")
york            = City_SIRD(23000,  initial_infected=0, name="York")
vnovgorod       = City_SIRD(50000,  initial_infected=0, name="Novgorod")
nurmberg        = City_SIRD(20000,  initial_infected=0, name="Nurmberg")
prague          = City_SIRD(40000,  initial_infected=0, name="Prague")
lubeck          = City_SIRD(24000,  initial_infected=0, name="Lubeck")
london          = City_SIRD(80000,  initial_infected=0, name="London")
krakow          = City_SIRD(10000,  initial_infected=0, name="Krakow")
bern            = City_SIRD(5000,   initial_infected=0, name="Bern")
erfurt          = City_SIRD(32000,  initial_infected=0, name="Erfurt")
bruges          = City_SIRD(50000,  initial_infected=0, name="Bruges")
bergen          = City_SIRD(7000,   initial_infected=0, name="Bergen")
deventer        = City_SIRD(13000,  initial_infected=0, name="Deventer")
copenhagen      = City_SIRD(3000,   initial_infected=0, name="Copenhagen")
polotsk         = City_SIRD(5000,   initial_infected=0, name="Polotsk")
wroclaw         = City_SIRD(12000,  initial_infected=0, name="Wroclaw")

# ==== ADD NODES TO GRAPH ====

G = nx.Graph()

cities = [
    feodosia, constantinople, thessaloniki, ragusa, venice, genoa,
    marseille, barcelona, palermo, naples, tunis, cairo, athens, florence,
    sarai, burgos, bordeaux, montpellier, toulouse, riga, vilnius, tallinn, berlin, brussels, danzig, ghent, turku, dublin, oslo,
    paris, stockholm, york, vnovgorod, nurmberg, prague, lubeck, london, krakow,
    bern, erfurt, bruges, bergen, deventer, copenhagen, polotsk, wroclaw
]

for c in cities:
    G.add_node(c)

# === EDGES ====
# Vidusjūras pilsētas sakari
G.add_edge(feodosia, constantinople, weight=3.0)
G.add_edge(feodosia, genoa,          weight=2.0)
G.add_edge(feodosia, sarai,          weight=0.4)

G.add_edge(constantinople, thessaloniki, weight=2.0)
G.add_edge(constantinople, ragusa,       weight=1.0)
G.add_edge(constantinople, venice,       weight=1.2)
G.add_edge(constantinople, athens,       weight=1.2)
G.add_edge(constantinople, genoa,        weight=1.3)
G.add_edge(constantinople, cairo,        weight=0.5)
G.add_edge(constantinople, palermo,      weight=1.2)

G.add_edge(cairo, venice,      weight=0.8)
G.add_edge(cairo, genoa,       weight=0.7)

G.add_edge(thessaloniki, athens, weight=1.2)
G.add_edge(thessaloniki, venice, weight=0.8)

G.add_edge(ragusa, venice, weight=2.0)
G.add_edge(ragusa, genoa,  weight=1.0)

G.add_edge(venice, genoa,      weight=2.2)
G.add_edge(venice, marseille,  weight=1.2)
G.add_edge(venice, athens,     weight=0.8)
G.add_edge(venice, florence,   weight=0.6)
G.add_edge(venice, naples,     weight=0.8)

G.add_edge(genoa, marseille,   weight=1.8)
G.add_edge(genoa, barcelona,   weight=1.3)

G.add_edge(marseille, barcelona, weight=2.0)
G.add_edge(marseille, tunis,     weight=1.0)

G.add_edge(barcelona, tunis,     weight=0.8)

G.add_edge(tunis, palermo, weight=1.3)
G.add_edge(palermo, naples, weight=1.0)
G.add_edge(naples, genoa,   weight=0.9)

G.add_edge(tunis, cairo, weight=0.5)

G.add_edge(barcelona, montpellier,  weight=1.3)
G.add_edge(montpellier, marseille,  weight=1.0)
G.add_edge(montpellier, toulouse,   weight=0.8)
G.add_edge(toulouse, bordeaux,      weight=0.9)

G.add_edge(barcelona, burgos,       weight=0.8)
G.add_edge(burgos, bordeaux,        weight=0.7)

G.add_edge(bordeaux, paris,   weight=1.2)
G.add_edge(bordeaux, london,  weight=1.0)

# Baltic-North Sea pilsētas

G.add_edge(london, bruges,  weight=1.2)
G.add_edge(london, paris,   weight=1.0)
G.add_edge(london, bergen,  weight=0.3)

G.add_edge(bruges, ghent,    weight=1.4)
G.add_edge(bruges, deventer, weight=0.7)
G.add_edge(bruges, lubeck,   weight=0.5)

G.add_edge(paris, brussels, weight=1.2)
G.add_edge(paris, bern,     weight=0.5)

G.add_edge(riga, tallinn,    weight=0.8)
G.add_edge(riga, vilnius,    weight=0.6)
G.add_edge(riga, danzig,     weight=1.0)
G.add_edge(riga, stockholm,  weight=0.4)

G.add_edge(danzig, wroclaw,    weight=0.6)
G.add_edge(danzig, lubeck,     weight=1.2)
G.add_edge(danzig, copenhagen, weight=1.0)

G.add_edge(lubeck, copenhagen, weight=1.2)
G.add_edge(lubeck, berlin,     weight=0.7)

G.add_edge(berlin, nurmberg, weight=0.6)
G.add_edge(nurmberg, prague, weight=0.8)
G.add_edge(prague, krakow,   weight=0.7)

G.add_edge(stockholm, bergen,     weight=0.2)
G.add_edge(stockholm, turku,      weight=0.6)
G.add_edge(stockholm, copenhagen, weight=0.8)
G.add_edge(oslo,      bergen,     weight=0.7)

G.add_edge(vnovgorod, polotsk, weight=0.7)
G.add_edge(vnovgorod, riga,    weight=0.7)

G.add_edge(polotsk, vilnius, weight=0.6)
G.add_edge(tallinn, stockholm, weight=0.8)

G.add_edge(erfurt, berlin,   weight=0.4)
G.add_edge(erfurt, nurmberg, weight=0.4)
G.add_edge(erfurt, prague,   weight=0.3)
G.add_edge(erfurt, bruges,   weight=0.3)

G.add_edge(brussels, ghent,    weight=0.7)
G.add_edge(brussels, deventer, weight=0.6)

G.add_edge(york,   london, weight=0.8)
G.add_edge(york,   dublin, weight=0.5)

G.add_edge(dublin, london, weight=0.7)

G.add_edge(wroclaw, prague, weight=0.6)
G.add_edge(wroclaw, krakow, weight=0.6)



initial_pop = {}
for c in G.nodes:
    initial_pop[c] = c.S + c.I + c.R + c.D



rng = random.Random(123)

for c in G.nodes:
    c.beta *= rng.uniform(0.9, 1.1)
    c.mu   *= rng.uniform(0.9, 1.1)
    c.cap_frac *= rng.uniform(0.8, 1.2)


# === STORAGE FOR TIME SERIES === NODZEST LATER!!!
history = {
    "Feodosia": [],
    "Constantinople": [],
    "Genoa": [],
    "Venice": [],
    "Marseille": [],
    "Palermo": [],
    "Paris": [],
    "London": [],
    "Florence": [],
    "Danzig": [],
    "Riga": [],
    "Novgorod": [],
}



# ==== SIMULĀCIJA  ====
rng = random.Random(42)
for day in range(1, 2555):
    base_ext = compute_commute_forces(G, commute_rate=0.00012)

    spike_ext = super_commute_spikes(
        G, day,
        super_period=14,
        super_prob=0.50,
        events=2,
        k_min=80, k_max=300,
        spike_rate=0.0008,
        rate_mult=80.0,
        rng=rng
    )

    ext = {}
    for u in G.nodes:
        ext[u] = base_ext[u] + spike_ext.get(u, 0.0)

    for city in G.nodes:
        city.step(external_infection_force=ext[city])

    print(f"\nDay {day}")
    for city in G.nodes:
        s = city.state()

        if city.name in history: # NODZEST!!!
            history[city.name].append(city.I) # NODZEST!!!

        N = city.S + city.I + city.R
        print(f"{city.name:9s} |  N={N:.0f}, S={s['S']:.0f}, I={s['I']:.0f}, R={s['R']:.0f}, D={s['D']:.0f}")


print("\nResults:")
for c in G.nodes:
    N0 = initial_pop[c]

    if N0 > 0:
        IFR = c.D / N0
    else:
        IFR = float('nan')

    print(f"{c.name:12s} D={int(c.D):7d}  N0={int(N0):7d}  IFR={IFR:.3f}  ({IFR * 100:.1f}%)")



# DELETE LATER!!!
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(history["Feodosia"], label="Feodosia")
plt.plot(history["Constantinople"], label="Constantinople")
plt.plot(history["Marseille"], label="Marseille")
plt.plot(history["Venice"], label="Venice")
plt.plot(history["Genoa"], label="Genoa")
plt.plot(history["Palermo"], label="Palermo")
plt.plot(history["Paris"], label="Paris")
plt.plot(history["Riga"], label="Riga")
plt.plot(history["Novgorod"], label="Novgorod")
plt.plot(history["Florence"], label="Florence")
plt.plot(history["London"], label="London")
plt.plot(history["Danzig"], label="Danzig")

plt.xlabel("Day")
plt.ylabel("Infected (I)")
plt.title("Dynamics of Infection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
