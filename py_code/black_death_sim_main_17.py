"""
The Black Death occurred in Europe from 1346 to 1353.
"""

import networkx as nx # Tīklu (grafu) bibliotēka: grafa virsotnes - pilsētas, grafa šķautnes - pilsētu savienojumi (tirdzniecības ceļi)
import random


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

    def __init__(self, population, initial_infected=1, beta=0.32, gamma=0.06, mu=0.09, name="City", cap_frac=0.02, overload_mult=3.0):
        # SIRD sākotnējie stāvokļi
        self.S = population - initial_infected
        self.I = initial_infected
        self.R = 0                # katra pilsēta no sākuma nav izveseļoto
        self.D = 0                # no sākuma nav neviena nomiruša
        self.beta = beta          # inficēšanās varbūtība par vienu dienu (lipīgums) (koeficients)
        self.gamma = gamma * random.uniform(0.90, 1.10) # random, lai dažādas pilsētas būtu nejauša izveseļošanas varbūtība (gamma) (lai tas atšķīras)
        self.mu = mu * random.uniform(0.85, 1.15) # random, lai dažādas pilsētas būtu nejauša nomiršanas varbūtība (mu) (lai tas atšķīras)
        self.name = name # pilsētas nosaukums

        # pilsētas "kapacitāte". Tas ir nepieciešams lai realizētu šo ideju: Ja ir pārāk daudz inficēto relatīvi pret dzīviem (I/N), tad palielinas mirstība (μ).
        self.cap_frac = cap_frac

        # cik daudz reižu var palielināties mirstība pie cap_frac pārpildīšanas.
        self.overload_mult = overload_mult

    def state(self):
        # Atgriež pašreizējo stāvokli (noapaļotu uz leju)
        return {"S": int(self.S), "I": int(self.I), "R": int(self.R), "D": int(self.D)}

    def step(self, external_infection_force=0.0):
        """
        Viena dienas solis.  external_infection_force = ārējās infekcijas ienesums (no mobilitātes u. c.).
        """

        # === POPULĀCIJAS STĀVOKLIS ===
        N = self.S + self.I + self.R # dzīvo iedzīvotāju skaits (S+I+R) (nav tikai D)
        if N <= 0:
            return None # visi nomira, nav ko darīt ar step.


        # === INFEKCIJAS SPĒKS (force) = vietējā infekcija + ārējais ienesums ===
        # Aprēķinām iekšējo infekcijas īpatsvaru (jo lielāks ir I relatīvi pret dzivajiem (N), jo lielāka varbūtība, ka inficēsies)
        internal_force = self.I / N

        # Saskaitām iekšējo un ārējo ienesumu
        raw_force = internal_force + external_infection_force # external_infection_force ieguvam no arienes

        # Ierobežojam force koeficientu no 0 līdz 1 (jo mēs saskaitām raw_force = internal_force + external_infection_force, un var gadīties, ka force ir lielāks par 1, kas nedrīkst būt.
        force = max(0.0, min(1.0, raw_force))


        # === DIENAS VARBŪTĪBAS: INFEKCIJA UN IZVESEĻOŠANĀS ===
        probability_to_infect = self.beta * force # inficēšanās varbūtība no melnas nāves * force koeficients (probability_infection for one person)
        probability_to_recover = self.gamma # izveseļošanas varbūtība par vienu dienu (koeficients (no pašas melnas nāves)


        # == SLODZE (pilsētas "kapacitāte", ja I/N ir parmerīgs (parāk daudz inficēto), tad palielinās mirstība) ==
        # Aprēķinām infekcijas slodzi: cik liela daļa iedzīvotāju ir inficēti
        load = self.I / N

        # Nosakām, cik slodze pārsniedz veselības sistēmas kapacitāti
        # Ja load <= cap_frac, tad overload = 0 (nav pārslogota)
        if load <= self.cap_frac:
            overload = 0.0
        else:
            overload = (load - self.cap_frac) / self.cap_frac

        # Palielinām mirstību atbilstoši pārsloges līmenim
        mu_eff = self.mu * (1 + self.overload_mult * overload)

        # Ierobežojam varbūtību [0..1]
        probability_to_die = min(1.0, mu_eff)


        # === INFICĒŠANA (S -> I) ===
        # Aprēķinām, cik uzņēmīgie cilvēki (S) inficējas šajā dienā
        new_infected = 0

        # Aprēķinām, cik S var inficēties (S -> I)
        for _ in range(int(self.S)):  # ejam cauri visiem tiem, kas var inficēties šajā pilsētā (S - susceptible)
            r = random.random()  # nejaušs skaitlis [0..1)
            if r < probability_to_infect:  # ja "monēta" krīt infekcijas labā
                new_infected += 1  # šis indivīds kļūst par inficētu




        # === IZVESEĻOŠANĀS (I -> R) ===
        # Aprēķinām, cik inficētie izveseļojas (I -> R)
        recovered = 0
        for _ in range(int(self.I)): # ejam cauri visiem inficējiem šajā pilsētā
            r = random.random()  # skaitlis [0..1)
            if r < probability_to_recover:
                recovered += 1  # šis cilvēks izveseļojas


        # === NĀVE (I -> D) ===
        # Aprēķinām, cik inficētie nomirst  (I -> D)
        died = 0
        for _ in range(int(self.I)):  # ejam cauri visiem inficējiem šajā pilsētā
            r = random.random()  # skaitlis [0..1)
            if r < probability_to_die:
                died += 1  # šis cilvēks nomirst


        # === STĀVOKĻA ATJAUNINĀŠANA ===
        self.S -= new_infected
        self.I += new_infected - recovered - died
        self.R += recovered
        self.D += died


        # === NEGATĪVU VĒRTĪBU NOVĒRŠANA ===
        self.S = max(0, self.S)
        self.I = max(0, self.I)
        self.R = max(0, self.R)


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
        external_infection_force[city] = 0.0

    # === 1) APREĶINĀM SUM_W(v) KATRAI PILSĒTAI ===
    # sum_w(v) = visu izejošo šķautņu svaru summa
    total_edge_weight_from_city = {}

    for city in G.nodes: # ejam cauri katrai city in G.nodes
        neighbors = list(G[city].items())  # visi kaimiņi

        if len(neighbors) == 0:
            # Ja nav kaimiņu, nav ārēja spiediena
            total_edge_weight_from_city[city] = 0.0
        else:
            total = 0.0
            for _, edge_data in neighbors:
                weight = edge_data.get("weight", 1.0)
                weight = max(weight, 0.0)  # negatīvs svars nav atļauts
                total += weight

            total_edge_weight_from_city[city] = total


    # === 2) APRĒĶINĀM IENĀKOŠO INFEKCIJAS SPIEDIENI KATRAI PILSĒTAI ===
    for target_city in G.nodes:                        # pilsēta, kas SAŅEM infekciju
        for source_city, edge_data in G[target_city].items():  # Pilsēta, kas NODOD infekciju
            weight_uv = edge_data.get("weight", 1.0)
            weight_uv = max(weight_uv, 0.0)

            # Summa visiem source_city savienojumiem
            total_w = total_edge_weight_from_city.get(source_city, 0.0)

            # Ja avota pilsētai nav maršrutu (nav transporta) – izlaižam
            if total_w <= 0.0:
                continue

            # Dzīvā populācija avota pilsētā
            N_v = source_city.S + source_city.I + source_city.R
            if N_v <= 0.0:
                continue  # pilsēta izmirusi

            infected_ratio = source_city.I / N_v  # I_v / N_v

            # Pieskaitām infekcijas ienesumu
            external_infection_force[target_city] += commute_rate * (weight_uv / total_w) * infected_ratio


    # === ATGRIEŽAM APRĒĶINĀTOS ĀRĒJOS SPIEDIENA KOEFICIENTUS (kā dict) ===
    return external_infection_force


def super_commute_spikes(
    G: nx.Graph,
    day: int,
    super_period: int = 7,       # ik pa N dienām pārbaudām
    super_prob: float = 0.35,    # varbūtība, ka šajā dienā notikums notiek
    events: int = 3,             # cik 'lēciens' (v->u) tiks izspēlēti
    k_min: int = 100,            # kontaktu 'paciņas' minimālais lielums
    k_max: int = 600,            # kontaktu 'paciņas' maksimālais lielums
    spike_rate: float = 0.02,    # mērogs (tās pašas vienības, kas commute_rate)
    rate_mult: float = 35.0,     # cik reizes stiprāks par parasto ienesumu
    rng: random.Random | None = None
) -> dict[City_SIRD, float]:
    """
    Atgriež: {pilsēta: papildus ārējais infekcijas spiediens} konkrētajā dienā.
    S/I/R/D NETIEK mainīti.

    IDEJA:
    - Reti (reizi super_period dienās) un ar varbūtību super_prob notiek 'liels notikums' (jeb šoks) (piem., kuģa ierašanās).
    """

    if rng is None:
        rng = random

    # === SĀKOTNĒJI: NAV PAPILDU SPIEDIENA NEVIENAI PILSĒTAI ===
    external_spike = {}
    for city in G.nodes:
        external_spike[city] = 0.0

    # === KALENDĀRA UN VARBŪTĪBAS TRIGGERS ===
    # Ja nav īstā diena vai notikums neizkrīt pēc varbūtības - atgriežam nulles
    if super_period <= 0:
        return external_spike
    if (day % super_period) != 0:
        return external_spike
    if rng.random() >= super_prob:
        return external_spike

    # === KANDIDĀTU APKOPE: v->u ar svariem ~ w * (I_v/N_v) ====
    candidates = []   # saraksts ar (v, u, w)
    weights = []      # atbilstošie svari izvēlei

    for source_city in G.nodes:
        N_v = source_city.S + source_city.I + source_city.R
        if N_v <= 0.0:
            continue
        if source_city.I <= 0.0:
            continue

        infected_ratio = source_city.I / N_v  # I_v / N_v

        for target_city, edge_data in G[source_city].items():
            w = edge_data.get("weight", 1.0)
            w = max(0.0, w)
            if w <= 0.0:
                continue

            candidates.append((source_city, target_city, w))
            weights.append(w * infected_ratio)

    # Ja nav ko izvēlēties — atgriežam nulles
    if not candidates:
        return external_spike
    if sum(weights) <= 0.0:
        return external_spike

    jumps_to_draw = max(1, int(events))

    for _ in range(jumps_to_draw):
        source_city, target_city, w = rng.choices(candidates, weights=weights, k=1)[0]

        N_u = target_city.S + target_city.I + target_city.R
        if N_u <= 0.0:
            continue  # mērķis 'izmiris' — šoks nepiemērojas

        contacts_k = rng.randint(k_min, k_max)

        external_spike[target_city] += rate_mult * spike_rate * (contacts_k / N_u)

        # (pēc izvēles) izmantošanai debug:
        # print(f"[day {day}] SUPER SPIKE: {source_city.name} -> {target_city.name}, k={contacts_k}, w={w:.2f}")

    # === REZULTĀTS ===
    return external_spike


# ==== IZVEIDOJAM TĪKLU ====

# Dienvidjūra
feodosia        = City_SIRD(40000,   initial_infected=100, name="Feodosia",      beta=0.28, gamma=0.08, mu=0.06)
constantinople  = City_SIRD(150000,  initial_infected=0, name="Constantinople",  beta=0.28, gamma=0.08, mu=0.06)
thessaloniki    = City_SIRD(120000,  initial_infected=0, name="Thessaloniki",    beta=0.28, gamma=0.08, mu=0.06)
ragusa          = City_SIRD(30000,   initial_infected=0, name="Ragusa",          beta=0.28, gamma=0.08, mu=0.06)
venice          = City_SIRD(150000,  initial_infected=0, name="Venice",          beta=0.28, gamma=0.08, mu=0.06)
genoa           = City_SIRD(90000,   initial_infected=0, name="Genoa",           beta=0.28, gamma=0.08, mu=0.06)
marseille       = City_SIRD(31000,   initial_infected=0, name="Marseille",       beta=0.28, gamma=0.08, mu=0.06)
barcelona       = City_SIRD(48000,   initial_infected=0, name="Barcelona",       beta=0.28, gamma=0.08, mu=0.06)
palermo         = City_SIRD(51000,   initial_infected=0, name="Palermo",         beta=0.28, gamma=0.08, mu=0.06)
naples          = City_SIRD(55000,   initial_infected=0, name="Naples",          beta=0.28, gamma=0.08, mu=0.06)
tunis           = City_SIRD(40000,   initial_infected=0, name="Tunis",           beta=0.28, gamma=0.08, mu=0.06)
cairo           = City_SIRD(400000,  initial_infected=0, name="Cairo",           beta=0.28, gamma=0.08, mu=0.06)
athens          = City_SIRD(25000,  initial_infected=0, name="Athens",           beta=0.28, gamma=0.08, mu=0.06)
florence        = City_SIRD(110000,  initial_infected=0, name="Florence",        beta=0.28, gamma=0.08, mu=0.06)

sarai           = City_SIRD(100000,  initial_infected=0, name="Sarai",           beta=0.28, gamma=0.08, mu=0.06)



# ==== ADD NODES TO GRAPH ====
G = nx.Graph()
for c in [feodosia, constantinople, thessaloniki, ragusa, venice, genoa,
          marseille, barcelona, palermo, naples, tunis, cairo, athens, florence, sarai]:
    G.add_node(c)

# === EDGES ====
G.add_edge(feodosia, constantinople, weight=3.0)
G.add_edge(feodosia, genoa,          weight=3.0)
G.add_edge(feodosia, sarai,          weight=1.0)


G.add_edge(constantinople, thessaloniki, weight=2.0)
G.add_edge(constantinople, ragusa,       weight=1.0)
G.add_edge(constantinople, venice,       weight=1.0)
G.add_edge(constantinople, athens,       weight=1.6)
G.add_edge(constantinople, genoa,        weight=1.4)
G.add_edge(constantinople, cairo,       weight=1.0)


G.add_edge(cairo, venice,      weight=1.2)
G.add_edge(cairo, genoa,       weight=1.0)


G.add_edge(thessaloniki, athens, weight=1.4)
G.add_edge(thessaloniki, venice, weight=0.9)


G.add_edge(ragusa, venice, weight=2.0)
G.add_edge(ragusa, genoa,  weight=1.0)


G.add_edge(venice, genoa,      weight=2.0)
G.add_edge(venice, marseille,  weight=1.0)
G.add_edge(venice, athens,  weight=1.1)
G.add_edge(venice, florence,  weight=0.7)
G.add_edge(venice, naples,  weight=0.9)

G.add_edge(genoa, marseille,   weight=1.5)
G.add_edge(genoa, barcelona,   weight=1.1)

G.add_edge(marseille, barcelona, weight=1.3)
G.add_edge(marseille, tunis,     weight=1.0)

G.add_edge(barcelona, tunis,     weight=0.9)



G.add_edge(tunis, palermo, weight=1.2)
G.add_edge(palermo, naples, weight=1.2)
G.add_edge(naples, genoa,   weight=1.0)


G.add_edge(tunis, cairo, weight=0.6)

initial_pop = {c: (c.S + c.I + c.R + c.D) for c in G.nodes}



rng = random.Random(123)

for c in G.nodes:
    c.beta *= rng.uniform(0.9, 1.1)
    c.mu   *= rng.uniform(0.9, 1.1)
    c.cap_frac *= rng.uniform(0.8, 1.2)



# ==== SIMULĀCIJA  ====
rng = random.Random(42)
for day in range(1, 366):
    base_ext = compute_commute_forces(G, commute_rate=0.001)

    spike_ext = super_commute_spikes(
        G, day,
        super_period=7,
        super_prob=0.35,
        events=1,
        k_min=10, k_max=60,
        spike_rate=0.002,
        rate_mult=35.0,
        rng=rng
    )


    ext = {u: base_ext[u] + spike_ext.get(u, 0.0) for u in G.nodes}

    for city in G.nodes:
        city.step(external_infection_force=ext[city])

    print(f"\nDay {day}")
    for city in G.nodes:
        s = city.state()
        N = city.S + city.I + city.R
        print(f"{city.name:9s} |  N={N:.0f}, S={s['S']:.0f}, I={s['I']:.0f}, R={s['R']:.0f}, D={s['D']:.0f}")


print("\n Results:")
for c in G.nodes:
    N0 = initial_pop[c]
    IFR = c.D / N0 if N0 > 0 else float('nan')
    print(f"{c.name:12s} D={int(c.D):7d}  N0={int(N0):7d}  IFR={IFR:.3f}  ({IFR*100:.1f}%)")