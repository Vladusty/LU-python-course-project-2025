import networkx as nx # tīklu (grafu) bibliotēka: grafa virsotnes - pilsētas, grafa šķautnes - pilsētu savienojumi (tirdzniecības ceļi)
import random
from numpy.random import binomial
import tkinter as tk
from math import sqrt # kvadrātsakne, nepieciešama apļu rādiusa aprēķināšanai vizualizācijā


"""
Melnās nāves izplatības simulācija (1346–1353).
Programma modelē mēra izplatību pilsētu tīklā, izmantojot lokālu SIRD epidemioloģisko modeli un starppilsētu kontaktu grafu.

1) City_SIRD klase
   Katra pilsēta tiek modelēta ar lokālu SIRD modeli (S - uzņēmīgie, I – inficētie, R – izveseļojušies, D – mirušie).
   Viens simulācijas solis atbilst vienai dienai, kuras laikā tiek aprēķināts:
      - jauno saslimušo skaits
      - izveseļojušo skaits
      - mirušo skaits
   Ja inficēto īpatsvars pārsniedz noteikto kapacitātes slieksni, tad mirstības varbūtība pilsētā pieaug, modelējot veselības sistēmas pārslodzi.

2) networkx.Graph (pilsētu tīkls):
   Grafa virsotnes ir City_SIRD objekti (pilsētas).
   Grafa šķautnes ir tirdzniecības ceļi ar noteikto svaru (weight), kas atspoguļo starppilsētu kontaktu intensitāti.

3) Starppilsētu infekcijas pārnese:
   compute_commute_forces() aprēķina ikdienas ārējo infekcijas spiedienu, kas rodas no kaimiņu pilsētām 
   un ir atkarīgs no to inficēto īpatsvara un attiecīgas šķautnes svara.
   super_commute_spikes() ģenerē retus notikumus (piemēram, kuģa ierašanās), kas rada pēkšņus slimības uzliesmojumus.

4) Tkinter GUI:
   1. logs: globālo parametru ievade.
   2. logs: sākotnējo saslimušo I iestatīšana pa pilsētām.
   3. logs: pilnekrāna Eiropas karte + animācija pa dienām.

Cilvēki neceļo starp pilsētām (S/I/R netiek pārnesti starp pilsētām).
Vienas pilsētas slimības pārnese tiek realizēta kā ārējais infekcijas spiediens (external_infection_force), kas palielina saslimšanas varbūtību mērķa pilsētā.
"""

class City_SIRD:
    """
    Klase, kas apraksta vienu pilsētu un tās iedzīvotāju stāvokli (SIRD modelis).
    S (susceptible) — cilvēki, kas var saslimst (uzņēmīgie)
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
        self.S = population - initial_infected # veselie ir visi mīnus tie, kas jau slimi
        self.I = initial_infected
        self.R = 0                # sākumā neviens nav izveseļojies
        self.D = 0                # sākumā neviens nav miris

        self.beta = beta          # inficēšanās varbūtība par vienu dienu (lipīgums) (koeficients)
        self.gamma = gamma        # izveseļošanas varbūtība par vienu dienu
        self.mu = mu              # nomiršanas varbūtība par vienu dienu (vēlāk visām pilsētām uzliekam nelielu nejaušu variāciju)

        # pilsētas "veselības sistēmas kapacitāte".
        # Tas ir nepieciešams lai realizētu šo ideju:
        # ja ir pārāk daudz inficēto relatīvi pret dzīviem (I/N), tad palielinas mirstība (μ).
        self.cap_frac = cap_frac

        # cik daudz reižu var palielināties mirstība pie cap_frac pārpildīšanas.
        self.overload_mult = overload_mult

    def step(self, external_infection_force=0):
        """
        Vienas pilsētas dinamika 1 dienas laikā.

        Metode neko neatgriež. Tā atjaunina self.S, self.I, self.R, self.D uz vietas.
        Ja N<=0 (pilsēta izmirusi), aprēķins netiek veikts.
        """
        # N - Dzīvie cilvēki (veselie + inficētie + recovered) (mirušie D netiek skaitīti)
        N = self.S + self.I + self.R
        if N <= 0:
            return None # pilsēta izmirusī, nav ko rēķināt - izejam no funkcijas

        # Iekšējais infekcijas spēks: I / N
        # jo lielāka I/N, jo lielāks risks saslimt
        internal_infection_force = self.I / N

        # Pievienojam ārējo infekcijas ienesumu (kopējais spiediens = iekšējais + ārējais (no tīkla))
        raw_force = internal_infection_force + external_infection_force

        # Drošības pēc ierobežojam robežās [0, 1]
        if raw_force < 0:
            force = 0
        elif raw_force > 1:
            force = 1
        else:
            force = raw_force

        # Inficēšanās un izveseļošanās koeficienti
        p_inf = self.beta * force
        p_rec = self.gamma

        # Veselības sistēmas pārslodze
        # load = inficēto īpatsvars starp dzīvajiem
        load = self.I / N

        # Ja load pārspēj kapacitāti, tad aprēķina pārslodzi.
        # proti, ja ir pārāk daudz inficēto relatīvi pret dzīviem (I/N), tad palielinas mirstība (μ).
        # mirstība pieaug proporcionāli pārslodzei.
        if load <= self.cap_frac:
            overload = 0
        else:
            overload = (load - self.cap_frac) / self.cap_frac

        # efektīva mirstība (jeb gala mirstība)
        mu_eff = self.mu * (1 + self.overload_mult * overload)

        # p_die = mu_eff, bet arī ierobežojam [0, 1]
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

        # Notikumi starp S (uzņemīgiem)
        # Jauno inficēto veidošana (S -> I) vai (S -> S)
        # katram no S ir varbūtība p_inf šodien saslimt
        S_int = int(self.S)
        new_infected = binomial(S_int, p_inf)

        # Notikumi starp I (inficētiem)
        # (I -> R vai I -> D vai I -> I)
        I_int = int(self.I)

        # p_rec + p_die nedrīkst pārsniegt 1
        # (viens cilvēks nevar vienlaikus izveseļoties un nomirt tajā pašā dienā)
        s = p_rec + p_die
        if s > 1: # Ja tomēr ir vairāk par 1, tad samazinam, saglabajot proporcijas starp p_rec un p_die
            p_rec = p_rec / s
            p_die = p_die / s

        # p_stay - varbūtība, ka inficētais paliek inficētais (I -> I)
        p_stay = 1 - p_rec - p_die
        if p_stay < 0:
            p_stay = 0

        # Cik daudz recovered (izveseļojušies) ir šodien
        rec = binomial(I_int, p_rec)

        # Palikušie infected
        remaining_after_rec = I_int - rec
        if remaining_after_rec < 0:
            remaining_after_rec = 0

        # Cik daudz nomira no palikušiem infected
        if p_die + p_stay == 0:
            died = 0
        else:
            died = binomial(remaining_after_rec, p_die / (p_die + p_stay))

        # Cik infected palika (atlikušais skaits paliek kā I)
        stay = remaining_after_rec - died
        if stay < 0:
            stay = 0


        # ATJAUNINĀM STĀVOKLI
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
    Aprēķina, cik liels ārējais infekcijas spiediens iedarbojas uz katru pilsētu no citām pilsētām pa tirdzniecības ceļiem.

    NEMAINA pilsētu iedzīvotāju skaitu (S, I, R, D),
    Funkcija tikai aprēķina papildu koeficientu (external_infection_force), kas vēlāk palielina saslimšanas varbūtību pilsētā.

    IDEJA:
    Katra pilsēta u saņem "ārējo infekcijas spiedienu" no visām kaimiņu pilsētām v (var iedomāties kā infekciju staru)
    Pieņēmums: jo spēcīgāka ir saikne v→u (tirdzniecības/intensitātes svars), un jo lielāka inficēto daļa v (I_v/N_v), jo lielāks ir infekcijas ienesums uz u.
    """

    # --------------------------------------------------
    # 0) Sagatavojam rezultātu vārdnīcu
    # --------------------------------------------------

    # Šeit glabāsim ārējo infekcijas spiedienu katrai pilsētai
    external_infection_force = {}

    # sākumā visām pilsētām ārējais spiediens ir 0 (external_infection_force = 0)
    for city in G.nodes:
        external_infection_force[city] = 0

    # --------------------------------------------------
    # 1) Aprēķinām katras pilsētas savienojumu kopējo svaru
    # --------------------------------------------------

    # Šī vārdnīca glabā, cik “spēcīgi” katra pilsēta ir savienota ar citām
    # (visu tās ceļu svaru summa)
    total_edge_weight_from_city = {}

    for city in G.nodes: # ejam cauri katrai city in G.nodes
        neighbors = list(G[city].items())  # visi kaimiņi

        if len(neighbors) == 0:
            # Ja pilsētai nav savienojumu, tā nevar nodot infekciju citām
            total_edge_weight_from_city[city] = 0
        else:
            total = 0
            for neighbor_city, edge_data in neighbors:
                weight = edge_data.get("weight", 1) # Paņemam ceļa svaru (ja nav, pieņemam 1)
                if weight < 0:
                    weight = 0  # negatīvs svars nav atļauts
                total += weight

            total_edge_weight_from_city[city] = total


    # === 2) APRĒĶINĀM IENĀKOŠO INFEKCIJAS SPIEDIENI KATRAI PILSĒTAI ===
    # target_city - pilsēta, uz kuru nāk infekcija
    for target_city in G.nodes:                                #  target_city - pilsēta, kas SAŅEM infekciju
        for source_city, edge_data in G[target_city].items():  # source_city - pilsēta, kas NODOD infekciju (no kuras infekcija nāk)
            weight_uv = edge_data.get("weight", 1) # Ceļa svars starp source_city un target_city
            if weight_uv < 0:
                weight_uv = 0

            # Summa visiem source_city savienojumiem
            total_w = total_edge_weight_from_city[source_city]

            # Ja source_city nav savienojumu, tā neko nenodod
            if total_w <= 0.0:
                continue

            # Dzīvo iedzīvotāju skaits source_city pilsētā
            N_v = source_city.S + source_city.I + source_city.R

            # Ja pilsēta ir izmirusi, no tās infekcija nenāk
            if N_v <= 0:
                continue

            # Inficēto īpatsvars avota pilsētā
            infected_ratio = source_city.I / N_v  # I_v / N_v

            # Pieskaitām ārējo infekcijas spiedienu mērķa pilsētai
            # Jo lielāks ceļa svars un vairāk slimo avotā, jo lielāks spiediens
            external_infection_force[target_city] += commute_rate * (weight_uv / total_w) * infected_ratio


    # ATGRIEŽAM APRĒĶINĀTOS ĀRĒJOS SPIEDIENA KOEFICIENTUS (kā dict)
    return external_infection_force


def super_commute_spikes(
    G,
    day,
    super_period = 14,      # ik pa N dienām pārbaudām, vai notiek “liels notikums”
    super_prob = 0.50,      # ja super_period diena pienākusi, tad ar varbūtību super_prob "liels notikums" tiešām notieks šajā dienā
    events = 2,             # cik “lēcienus” (no vienas pilsētas uz citu) izspēlējam vienā notikumā
    k_min = 100,            # minimālais kontaktu skaits lielā notikumā
    k_max = 300,            # maksimālais kontaktu skaits lielā notikumā
    spike_rate = 0.0008,    # bāzes stiprums (tāds pats “mērogs” kā commute_rate)
    rate_mult = 80.0,       # cik reizes stiprāks par parasto ienesumu
    rng = None
) -> dict[City_SIRD, float]:
    """
    Aprēķina “lielo notikumu” papildus ārējo infekcijas spiedienu konkrētajā dienā.

    Šī funkcija:
    NEMAINA S/I/R/D (tā neieliek cilvēkus tieši I).
    Funkcija tikai atgriež external_spike[u], ko pēc tam pieskaita pie ārējā spiediena, un tas palielina saslimšanas varbūtību City_SIRD.step() metodē.

    Atgriež: {pilsēta: papildus ārējais infekcijas spiediens} konkrētajā dienā.
    S/I/R/D NETIEK mainīti.

    IDEJA:
    Reti (reizi super_period dienās) un ar varbūtību super_prob notiek 'liels notikums' (piem., kuģa ierašanās), kas izraisa pēkšņu uzliesmojumu kādā pilsētā.
    """

    # Ja nav padots savs rng, tad izmantojam standarta random moduli
    if rng is None:
        rng = random

    # Sākumā visām pilsētām papildus spiediens ir 0
    external_spike = {}
    for city in G.nodes:
        external_spike[city] = 0.0


    # --------------------------------------------------
    # 1) Pārbaudām, vai šodien vispār var notikt “lielais notikums”
    # --------------------------------------------------

    # Ja super_period ir 0 vai mazāks, tad "lielie notikumi" nekad nenotiek
    if super_period <= 0:
        return external_spike

    # Ja šī diena nav dalāma ar super_period, tad šodien "lielais notikums" nav paredzēts
    if (day % super_period) != 0:
        return external_spike

    # Pat ja diena der, tad notikums notiek tikai ar varbūtību super_prob
    if rng.random() >= super_prob:
        return external_spike


    # --------------------------------------------------
    # 2) Savācam kandidātus: no kuras pilsētas uz kuru var trāpīt “lielais notikums”
    # --------------------------------------------------

    candidates = []   # candidates glabā maršrutus (source_city -> target_city) ar svaru w
    weights = []      # weights glabā “izlozes svarus” (kuras pārejas biežāk izvēlēties)

    for source_city in G.nodes:
        N_v = source_city.S + source_city.I + source_city.R # source_city pilsētas dzīvie iedzīvotāji
        if N_v <= 0:
            continue # source_city pilsēta izmirusi

        if source_city.I <= 0: # Ja avotā nav slimo, tad nav jēgas no šī avota taisīt "lielo notikumu"
            continue

        infected_ratio = source_city.I / N_v  # I_v / N_v (Cik liela daļa source_city pilsētā ir slimi)

        # Paskatāmies, uz kurām pilsētām avots ir savienots (pa grafa šķautnēm)
        for target_city, edge_data in G[source_city].items():
            w = edge_data.get("weight", 1)
            w = max(0.0, w)

            # Ja svars 0, tad savienojums faktiski neko nenozīmē
            if w <= 0.0:
                continue

            candidates.append((source_city, target_city, w))

            # Izlozes svars: stiprāks ceļš + vairāk slimo avotā => biežāk izvēlas
            weights.append(w * infected_ratio)

    # Ja nav neviena kandidāta, tad "lielais notikums" neko nevar izdarīt
    if len(candidates) < 1:
        return external_spike

    # Ja visi svari ir 0, tad arī nav ko izlozēt
    total_weights = 0
    for w in weights:
        total_weights += w

    # Ja svars = 0, tad nav iespējams izvēlēties nevienu kandidātu
    if total_weights == 0:
        return external_spike

    # Ja events <= 0, tad neviens “lēciens” vispār nenotiek
    if int(events) <= 0:
        return external_spike

    # Cik lēcienus izspēlēsim (drošībai vismaz 1)
    jumps_to_draw = int(events)
    if jumps_to_draw < 1:
        jumps_to_draw = 1

    # --------------------------------------------------
    # 3) Izlozējam konkrētus “lēciens” un pievienojam spiedienu mērķa pilsētām
    # --------------------------------------------------

    for _ in range(jumps_to_draw):
        # Kopējais izlozes svaru lielums (vajadzīgs ruletes izvēlei)
        total_weight = 0.0
        for w in weights:
            total_weight += w

        # Izvēlamies nejaušu skaitli intervālā [0; total_weight)
        r = rng.random() * total_weight

        # “ruletes” izvēle: ejam cauri svariem, līdz sasniedzam r
        cum = 0.0
        index = 0
        while index < len(weights):
            cum += weights[index]
            if r <= cum:
                break
            index += 1

        # Drošības gadījumam (ja aizgāja ārpus robežas)
        if index >= len(weights):
            index = len(weights) - 1

        # Paņemam izvēlēto maršrutu
        source_city, target_city, w = candidates[index]

        # Mērķa pilsētā jābūt dzīvajiem iedzīvotājiem
        N_u = target_city.S + target_city.I + target_city.R
        if N_u <= 0.0:
            continue # mērķa pilsēta izmirusi

        # Nejaušs kontaktu skaits (cik “cilvēku kontaktu” notika "lielā notikuma" dēļ)
        contacts_k = rng.randint(k_min, k_max)

        # Pievienojam papildus ārējo spiedienu mērķa pilsētai
        # Dalām ar N_u, lai lielās pilsētās tas pats contacts_k dotu mazāku relatīvo efektu
        external_spike[target_city] += rate_mult * spike_rate * (contacts_k / N_u)


    # Atgriežam papildus spiedienu katrai pilsētai
    return external_spike


# ==== IZVEIDOJAM TĪKLU ====

# Vidusjūra (virsotnes, ar populācijas skaitu, initial_infected skaitu un nosaukumu)
feodosia       = City_SIRD(40000,  initial_infected=0, name="Feodosia")
constantinople = City_SIRD(150000, initial_infected=0, name="Constantinople")
thessaloniki   = City_SIRD(120000, initial_infected=0, name="Thessaloniki")
ragusa         = City_SIRD(30000,  initial_infected=0, name="Ragusa")
venice         = City_SIRD(150000, initial_infected=0, name="Venice")
genoa          = City_SIRD(90000,  initial_infected=0, name="Genoa")
marseille      = City_SIRD(31000,  initial_infected=0, name="Marseille")
barcelona      = City_SIRD(48000,  initial_infected=0, name="Barcelona")
palermo        = City_SIRD(51000,  initial_infected=0, name="Palermo")
naples         = City_SIRD(55000,  initial_infected=0, name="Naples")
athens         = City_SIRD(25000,  initial_infected=0, name="Athens")
florence       = City_SIRD(110000, initial_infected=0, name="Florence")
rome           = City_SIRD(40000,  initial_infected=0, name="Rome")
moscow         = City_SIRD(25000,  initial_infected=0, name="Moscow")
sofia          = City_SIRD(20000,  initial_infected=0, name="Sofia")
belgrade       = City_SIRD(20000,  initial_infected=0, name="Belgrade")
lviv           = City_SIRD(10000,  initial_infected=0, name="Lviv")
cnapoca        = City_SIRD(5000,   initial_infected=0, name="Cluj-Napoca")
debrecen       = City_SIRD(3000,   initial_infected=0, name="Debrecen")
durres         = City_SIRD(25000,  initial_infected=0, name="Durres")
burgos         = City_SIRD(25000,  initial_infected=0, name="Burgos")
bordeaux       = City_SIRD(35000,  initial_infected=0, name="Bordeaux")
montpellier    = City_SIRD(35000,  initial_infected=0, name="Montpellier")
toulouse       = City_SIRD(30000,  initial_infected=0, name="Toulouse")
geneve         = City_SIRD(4000,   initial_infected=0, name="Geneve")
granada        = City_SIRD(150000, initial_infected=0, name="Granada")
kyiv           = City_SIRD(20000,  initial_infected=0, name="Kyiv")
lisboa         = City_SIRD(35000,  initial_infected=0, name="Lisboa")
milan          = City_SIRD(120000, initial_infected=0, name="Milan")
pest           = City_SIRD(10000,  initial_infected=0, name="Pest")
suceava        = City_SIRD(20000,  initial_infected=0, name="Suceava")
seville        = City_SIRD(45000,  initial_infected=0, name="Seville")
siena          = City_SIRD(50000,  initial_infected=0, name="Siena")
targoviste     = City_SIRD(25000,  initial_infected=0, name="Târgoviște")
toledo         = City_SIRD(42000,  initial_infected=0, name="Toledo")
trnovo         = City_SIRD(35000,  initial_infected=0, name="Veliko Tarnovo")
vienna         = City_SIRD(35000,  initial_infected=0, name="Vienna")
zagreb         = City_SIRD(5000,   initial_infected=0, name="Zagreb")
zurich         = City_SIRD(6000,   initial_infected=0, name="Zurich")

fes            = City_SIRD(175000, initial_infected=0, name="Fes")
izmir          = City_SIRD(30000,  initial_infected=0, name="Izmir")
jerusalem      = City_SIRD(10000,  initial_infected=0, name="Jerusalem")
tunis          = City_SIRD(40000,  initial_infected=0, name="Tunis")
cairo          = City_SIRD(400000, initial_infected=0, name="Cairo")

sarai          = City_SIRD(100000, initial_infected=100, name="Sarai")


# Baltijas un Ziemeļu jūra jeb sviesta eiropa

riga           = City_SIRD(6500,   initial_infected=0, name="Riga")
vilnius        = City_SIRD(20000,  initial_infected=0, name="Vilnius")
tallinn        = City_SIRD(4000,   initial_infected=0, name="Tallinn")
berlin         = City_SIRD(5500,   initial_infected=0, name="Berlin")
brussels       = City_SIRD(30000,  initial_infected=0, name="Brussels")
danzig         = City_SIRD(20000,  initial_infected=0, name="Danzig")
ghent          = City_SIRD(55000,  initial_infected=0, name="Ghent")
turku          = City_SIRD(5000,   initial_infected=0, name="Turku")
dublin         = City_SIRD(20000,  initial_infected=0, name="Dublin")
oslo           = City_SIRD(5000,   initial_infected=0, name="Oslo")
paris          = City_SIRD(200000, initial_infected=0, name="Paris")
stockholm      = City_SIRD(8000,   initial_infected=0, name="Stockholm")
york           = City_SIRD(23000,  initial_infected=0, name="York")
novgorod       = City_SIRD(50000,  initial_infected=0, name="Novgorod")
nurmberg       = City_SIRD(20000,  initial_infected=0, name="Nurmberg")
prague         = City_SIRD(40000,  initial_infected=0, name="Prague")
lubeck         = City_SIRD(24000,  initial_infected=0, name="Lubeck")
london         = City_SIRD(80000,  initial_infected=0, name="London")
krakow         = City_SIRD(10000,  initial_infected=0, name="Krakow")
bern           = City_SIRD(5000,   initial_infected=0, name="Bern")
erfurt         = City_SIRD(32000,  initial_infected=0, name="Erfurt")
bruges         = City_SIRD(50000,  initial_infected=0, name="Bruges")
bergen         = City_SIRD(7000,   initial_infected=0, name="Bergen")
deventer       = City_SIRD(13000,  initial_infected=0, name="Deventer")
copenhagen     = City_SIRD(3000,   initial_infected=0, name="Copenhagen")
polotsk        = City_SIRD(5000,   initial_infected=0, name="Polotsk")
wroclaw        = City_SIRD(12000,  initial_infected=0, name="Wroclaw")
cologne        = City_SIRD(54000,  initial_infected=0, name="Cologne")
olomouc        = City_SIRD(10000,  initial_infected=0, name="Olomouc")
rouen          = City_SIRD(40000,  initial_infected=0, name="Rouen")
yaroslavl      = City_SIRD(5000,   initial_infected=0, name="Yaroslavl")
smolensk       = City_SIRD(10000,  initial_infected=0, name="Smolensk")

# ==== ADD NODES TO GRAPH ====

G = nx.Graph()

cities = [
    feodosia, constantinople, thessaloniki, ragusa, venice, genoa,
    marseille, barcelona, palermo, naples, tunis, cairo, athens, florence,
    sarai, burgos, bordeaux, montpellier, toulouse, riga, vilnius, tallinn, berlin,
    brussels, danzig, ghent, turku, dublin, oslo, paris, stockholm, york, novgorod, nurmberg,
    prague, lubeck, london, krakow, bern, erfurt, bruges, bergen, deventer, copenhagen,
    polotsk, wroclaw, rome, moscow, sofia, belgrade, lviv , cnapoca, debrecen, durres,
    geneve, granada, kyiv, lisboa, milan, pest, suceava, seville, siena, targoviste, toledo,
    trnovo, vienna, zagreb, zurich, fes, izmir, jerusalem, cologne, olomouc, rouen, yaroslavl, smolensk
]

for c in cities:
    G.add_node(c)

# === EDGES ====
# Vidusjūras pilsētas sakari (šķautnes)

G.add_edge(targoviste, constantinople,    weight=0.3)
G.add_edge(targoviste, belgrade,          weight=0.3)
G.add_edge(targoviste, trnovo,            weight=0.3)
G.add_edge(targoviste, suceava,           weight=0.3)

G.add_edge(feodosia, constantinople,     weight=3.0)
G.add_edge(feodosia, genoa,              weight=1.7)
G.add_edge(feodosia, sarai,              weight=0.4)
G.add_edge(feodosia, suceava,            weight=0.3)

G.add_edge(constantinople, thessaloniki, weight=2.0)
G.add_edge(constantinople, ragusa,       weight=1.0)
G.add_edge(constantinople, venice,       weight=1.2)
G.add_edge(constantinople, athens,       weight=1.2)
G.add_edge(constantinople, genoa,        weight=1.3)
G.add_edge(constantinople, cairo,        weight=0.5)
G.add_edge(constantinople, palermo,      weight=1.2)
G.add_edge(constantinople, izmir,        weight=1.5) 
G.add_edge(constantinople, trnovo,       weight=0.8)
G.add_edge(constantinople, sofia,        weight=0.7)

G.add_edge(cairo, venice,                weight=0.8)
G.add_edge(cairo, genoa,                 weight=0.7)
G.add_edge(cairo, jerusalem,             weight=0.5)

G.add_edge(thessaloniki, athens,         weight=1.2)
G.add_edge(thessaloniki, venice,         weight=0.8)
G.add_edge(thessaloniki, ragusa,         weight=0.6) 
G.add_edge(thessaloniki, durres,         weight=0.5)

G.add_edge(ragusa, venice,               weight=2.0)
G.add_edge(ragusa, genoa,                weight=1.0)
G.add_edge(ragusa, durres,               weight=1.2) 
G.add_edge(ragusa, belgrade,             weight=0.7) 
G.add_edge(ragusa, pest,                 weight=0.5)

G.add_edge(venice, genoa,                weight=2.2)
G.add_edge(venice, marseille,            weight=1.2)
G.add_edge(venice, athens,               weight=0.8)
G.add_edge(venice, florence,             weight=0.6)
G.add_edge(venice, naples,               weight=0.8)
G.add_edge(venice, rome,                 weight=0.5) 
G.add_edge(venice, milan,                weight=0.7) 
G.add_edge(venice, prague,               weight=0.4)

G.add_edge(genoa, marseille,             weight=1.8)
G.add_edge(genoa, barcelona,             weight=1.3)
G.add_edge(genoa, milan,                 weight=1.0) 
G.add_edge(genoa, tunis,                 weight=1.0)

G.add_edge(marseille, tunis,             weight=1.0)
G.add_edge(marseille, montpellier,       weight=1.2)
G.add_edge(marseille, barcelona,         weight=2.0)
G.add_edge(marseille, paris,             weight=0.4)

G.add_edge(barcelona, tunis,             weight=0.8)
G.add_edge(barcelona, granada,           weight=0.7)
G.add_edge(barcelona, seville,           weight=0.6)
G.add_edge(barcelona, montpellier,       weight=1.3)
G.add_edge(barcelona, burgos,            weight=0.8)

G.add_edge(naples, genoa,                weight=0.9)
G.add_edge(naples, rome,                 weight=0.7)
G.add_edge(naples, palermo,              weight=1.0)

G.add_edge(tunis, palermo,               weight=1.3)
G.add_edge(tunis, cairo,                 weight=0.5)
G.add_edge(tunis, fes,                   weight=0.7)

G.add_edge(fes, granada,                 weight=0.7)

G.add_edge(toulouse, bordeaux,           weight=0.9)
G.add_edge(toulouse, granada,            weight=0.3)
G.add_edge(toulouse, montpellier,        weight=0.8)

G.add_edge(burgos, bordeaux,             weight=0.7)
G.add_edge(burgos, toledo,               weight=0.6) 
G.add_edge(burgos, seville,              weight=0.5) 
G.add_edge(burgos, lisboa,               weight=0.4)

G.add_edge(bordeaux, paris,              weight=1.2)
G.add_edge(bordeaux, london,             weight=1.0)
G.add_edge(bordeaux, lisboa,             weight=0.5)

G.add_edge(rome, florence,               weight=0.8) 
G.add_edge(florence, siena,              weight=0.7) 

G.add_edge(geneve, zurich,               weight=0.6)
G.add_edge(geneve, bern,                 weight=0.5)
G.add_edge(geneve, milan,               weight=0.8)

G.add_edge(belgrade, sofia,              weight=0.6)
G.add_edge(belgrade, pest,               weight=0.8) 

G.add_edge(sofia, trnovo,                weight=0.5)

G.add_edge(cnapoca, debrecen,            weight=0.4)
 
G.add_edge(kyiv, lviv,                   weight=0.5) 
G.add_edge(kyiv, moscow,                 weight=0.4) 
G.add_edge(kyiv, polotsk,                weight=0.3)

G.add_edge(lviv, cnapoca,                weight=0.3)
G.add_edge(lviv, krakow,                 weight=0.6) 

G.add_edge(seville, lisboa,              weight=0.7)
G.add_edge(seville, toledo,              weight=0.6) 
G.add_edge(seville, granada,             weight=0.6)

G.add_edge(vienna, pest,                 weight=0.6) 
G.add_edge(vienna, zagreb,               weight=0.4)
G.add_edge(vienna, krakow,               weight=0.8)
G.add_edge(vienna, venice,               weight=0.9)

# Baltic-North Sea pilsētas

G.add_edge(london, bruges,               weight=1.4)
G.add_edge(london, paris,                weight=1.0)
G.add_edge(london, bergen,               weight=0.3)
G.add_edge(london, york,                 weight=0.8) 
G.add_edge(london, dublin,               weight=0.7)

G.add_edge(york, dublin,                 weight=0.5)

G.add_edge(bruges, ghent,                weight=1.4)
G.add_edge(bruges, deventer,             weight=0.7)
G.add_edge(bruges, lubeck,               weight=1.0)
G.add_edge(bruges, brussels,             weight=1.0) 

G.add_edge(paris, brussels,              weight=1.2)
G.add_edge(paris, bern,                  weight=0.5)
G.add_edge(paris, rouen,                 weight=1.0) 

G.add_edge(riga, tallinn,                weight=0.8)
G.add_edge(riga, vilnius,                weight=0.6)
G.add_edge(riga, danzig,                 weight=1.1)
G.add_edge(riga, stockholm,              weight=0.5)
G.add_edge(riga, polotsk,                weight=0.5) 

G.add_edge(danzig, wroclaw,              weight=0.6)
G.add_edge(danzig, lubeck,               weight=1.4)
G.add_edge(danzig, copenhagen,           weight=1.0)
G.add_edge(danzig, krakow,               weight=0.7) 

G.add_edge(lubeck, copenhagen,           weight=1.3)
G.add_edge(lubeck, berlin,               weight=0.7)
G.add_edge(lubeck, deventer,             weight=0.6)

G.add_edge(berlin, nurmberg,             weight=0.6)
G.add_edge(berlin, erfurt,               weight=0.4) 

G.add_edge(nurmberg, prague,             weight=0.8)
G.add_edge(nurmberg, vienna,             weight=0.6) 
G.add_edge(nurmberg, cologne,            weight=0.5) 

G.add_edge(prague, krakow,               weight=0.7)
G.add_edge(prague, olomouc,              weight=0.5) 
G.add_edge(prague, vienna,               weight=0.7) 

G.add_edge(stockholm, bergen,            weight=0.2)
G.add_edge(stockholm, turku,             weight=0.6)
G.add_edge(stockholm, copenhagen,        weight=0.8)
G.add_edge(stockholm, tallinn,           weight=0.8)

G.add_edge(oslo, bergen,                 weight=0.7)
G.add_edge(oslo, copenhagen,             weight=0.4) 

G.add_edge(novgorod, polotsk,           weight=0.7)
G.add_edge(novgorod, riga,              weight=0.9)
G.add_edge(novgorod, smolensk,          weight=0.4) 
G.add_edge(novgorod, yaroslavl,         weight=0.3) 
G.add_edge(novgorod, moscow,            weight=0.5)

G.add_edge(polotsk, vilnius,             weight=0.6)
G.add_edge(polotsk, smolensk,            weight=0.5) 

G.add_edge(erfurt, nurmberg,             weight=0.4)
G.add_edge(erfurt, prague,               weight=0.3)
G.add_edge(erfurt, bruges,               weight=0.3)
G.add_edge(erfurt, cologne,              weight=0.3) 

G.add_edge(brussels, ghent,              weight=0.7)
G.add_edge(brussels, deventer,           weight=0.6)
G.add_edge(brussels, cologne,            weight=0.5) 

G.add_edge(wroclaw, prague,              weight=0.6)
G.add_edge(wroclaw, krakow,              weight=0.6)
G.add_edge(wroclaw, olomouc,             weight=0.4) 

G.add_edge(cologne, deventer,            weight=0.7) 
G.add_edge(cologne, rouen,               weight=0.4) 

G.add_edge(yaroslavl, moscow,            weight=0.4)


initial_pop = {}
for c in G.nodes:
    initial_pop[c] = c.S + c.I + c.R + c.D


# ==== TKINTER KARTE PĒC MODEĻA UN GRAFA ====

# Kartes robežas
MAP_BOUNDS = dict(
    lon_min=-10.0,  # rietumos aiz Ibērijas
    lon_max=50.0,   # austrumos aiz Sarai
    lat_min=30.0,   # nedaudz dienvidos no Kairas
    lat_max=62.0    # ziemeļos virs Bergen / Turku
)

# Reālas (aptuvenas) pilsētu garuma/platuma koordinātas grafā G
COORDS = {
    "Feodosia":      (35.38, 45.03),
    "Constantinople":(28.97, 41.01),
    "Thessaloniki":  (22.94, 40.64),
    "Ragusa":        (18.09, 42.65),
    "Venice":        (12.33, 45.44),
    "Genoa":         (8.93,  44.41),
    "Marseille":     (5.37,  43.30),
    "Barcelona":     (2.17,  41.38),
    "Palermo":       (13.36, 38.12),
    "Naples":        (14.27, 40.85),
    "Tunis":         (10.17, 36.81),
    "Cairo":         (31.24, 35.04),
    "Athens":        (23.73, 37.98),
    "Florence":      (11.25, 43.77),
    "Sarai":         (46.00, 48.00),

    "Burgos":        (-3.70, 42.34),
    "Bordeaux":      (-0.57, 44.84),
    "Montpellier":   (3.88, 43.61),
    "Toulouse":      (1.44, 43.60),

    "Riga":          (24.10, 56.95),
    "Vilnius":       (25.27, 54.68),
    "Tallinn":       (24.75, 59.44),
    "Berlin":        (13.40, 52.52),
    "Brussels":      (4.35, 50.85),
    "Danzig":        (18.65, 54.35),
    "Ghent":         (3.72, 51.05),
    "Turku":         (22.27, 60.45),
    "Dublin":        (-6.26, 53.35),
    "Oslo":          (10.75, 59.91),
    "Paris":         (2.35, 48.86),
    "Stockholm":     (18.07, 59.33),
    "York":          (-1.08, 53.96),
    "Novgorod":      (31.27, 58.52),
    "Nurmberg":      (11.08, 49.45),
    "Prague":        (14.42, 50.08),
    "Lubeck":        (10.69, 53.87),
    "London":        (-0.13, 51.51),
    "Krakow":        (19.94, 50.06),
    "Bern":          (7.45, 46.95),
    "Erfurt":        (11.03, 50.98),
    "Bruges":        (3.22, 51.21),
    "Bergen":        (5.32, 60.39),
    "Deventer":      (6.16, 52.25),
    "Copenhagen":    (12.57, 55.68),
    "Polotsk":       (28.80, 55.48),
    "Wroclaw":       (17.03, 51.11),

    "Rome":          (12.50, 41.90),
    "Moscow":        (37.62, 55.76),
    "Sofia":         (23.32, 42.70),
    "Belgrade":      (20.45, 44.79),
    "Lviv":          (24.03, 49.84),
    "Cluj-Napoca":   (23.60, 46.77),
    "Debrecen":      (21.63, 47.53),
    "Durres":        (19.46, 41.32),
    "Geneve":        (6.12, 46.20),
    "Granada":       (-3.60, 37.18),
    "Kyiv":          (30.52, 50.45),
    "Lisboa":        (-9.14, 38.72),
    "Milan":         (9.19, 45.46),
    "Pest":          (19.04, 47.50),
    "Suceava":       (26.26, 47.65),
    "Seville":       (-5.98, 37.39),
    "Siena":         (11.33, 43.32),
    "Târgoviște":    (25.46, 44.93),
    "Toledo":        (-4.03, 39.86),
    "Veliko Tarnovo":(25.62, 43.08),
    "Vienna":        (16.37, 48.21),
    "Zagreb":        (15.98, 45.81),
    "Zurich":        (8.54, 47.38),
    "Fes":           (-5.01, 34.02),
    "Izmir":         (27.15, 38.42),
    "Jerusalem":     (35.21, 31.77),
    "Cologne":       (6.92, 50.94),
    "Olomouc":       (17.25, 49.53),
    "Rouen":         (1.10, 49.44),
    "Yaroslavl":     (39.88, 57.63),
    "Smolensk":      (32.04, 54.78),
}

def lonlat_to_xy(lon, lat, W, H, bounds=MAP_BOUNDS):
    """Vienkārša lineāra projekcija no garuma/platuma uz Canvas pikseļiem."""
    x = (lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * W
    # y virziens uz leju — lielāka platuma vērtība = tuvāk ekrāna augšai
    y = (bounds["lat_max"] - lat) / (bounds["lat_max"] - bounds["lat_min"]) * H
    return x, y


class MapView:
    """
    Šī klase atbild TIKAI par kartes attēlošanu.
    Canvas tiek zīmēts:
      - pelēkas līnijas — grafa šķautnes (tirdzniecības ceļi)
      - melns aplis — dzīvie iedzīvotāji (N = S+I+R)
      - sarkans aplis — inficētie (I)
    """

    def __init__(self, root, G, width=1920, height=1080, scale_k_alive=0.15, scale_k_inf=0.25, bg="white"):
        self.root = root

        # Grafs ar pilsētām un to savienojumiem
        self.G = G

        # Ekrāna izmēri
        self.W, self.H = width, height

        # Koeficienti, kas nosaka, cik lieli būs apļi
        self.scale_k_alive = scale_k_alive
        self.scale_k_inf = scale_k_inf

        self.canvas = tk.Canvas(root, width=self.W, height=self.H, bg=bg, highlightthickness=0)
        self.canvas.pack()

        # --------------------------------------------------
        # 1) Aprēķinām pilsētu koordinātas pikseļos
        # --------------------------------------------------

        self.xy = {} # Šeit glabāsim pilsētu koordinātas uz ekrāna
        for city in self.G.nodes:
            lon, lat = COORDS[city.name] # Paņemam pilsētas ģeogrāfiskās koordinātas
            self.xy[city] = lonlat_to_xy(lon, lat, self.W, self.H) # Pārveidojam tās uz Canvas koordinātām

        # --------------------------------------------------
        # 2) Uzzīmējam šķautnes starp virsotnēm (pilsētām) (vienu reizi)
        # --------------------------------------------------

        self.edge_ids = []
        for u, v, data in self.G.edges(data=True):
            x1, y1 = self.xy[u]
            x2, y2 = self.xy[v]
            self.edge_ids.append(self.canvas.create_line(x1, y1, x2, y2, fill="#cccccc", width=1)) # pelēka līnija - tirdzniecības ceļš

        # --------------------------------------------------
        # 3) Izveidojam pilsētu apļus un uzrakstus
        # --------------------------------------------------

        # Pilsētu apļi un uzraksti (izveido vienreiz, vēlāk tikai maina coords)
        self.node_art = {}  # city -> (outer_id, inner_id, label_id)
        for city in self.G.nodes:
            x, y = self.xy[city]

            # Melnais aplis (dzīvie)
            outer = self.canvas.create_oval(x-1, y-1, x+1, y+1, outline="black", width=2, fill="")

            # Sarkanais aplis (inficētie)
            inner = self.canvas.create_oval(x-1, y-1, x+1, y+1, outline="", fill="#ff0000")

            # Nosaukums virs pilsētas
            label = self.canvas.create_text(x, y-12, text=city.name, font=("Arial", 9))

            self.node_art[city] = (outer, inner, label)

    @staticmethod
    def _bbox(cx, cy, r):
        """Veido ovāla koordinātas no centra (cx, cy) un rādiusa r."""
        r = max(0.0, r)
        return (cx - r, cy - r, cx + r, cy + r)

    def _r_alive(self, N):
        """
        Aprēķina melnā apļa rādiusu (dzīvie iedzīvotāji).
        Izmantojam sqrt(N), lai aplis neaugtu pārāk strauji.
        """
        if N <= 0:
            return 0.0
        return max(2.0, self.scale_k_alive * sqrt(N))

    def _r_inf(self, I):
        """
        Aprēķina sarkanā apļa rādiusu (inficētie).
        """
        if I <= 0:
            return 0.0
        return max(3.0, self.scale_k_inf * sqrt(I))

    def update_city(self, city):
        """
        Atjauno vienas pilsētas attēlojumu: maina apļu izmērus un redzamību.
        Tas tiek izmantots katra soli.
        """
        outer, inner, label = self.node_art[city]
        x, y = self.xy[city]


        N = city.S + city.I + city.R # Dzīvo iedzīvotāju skaits
        I = city.I # Inficēto skaits

        # Aprēķinām rādiusus
        rN = self._r_alive(N)
        rI = self._r_inf(I)

        # Melnais aplis (dzīvie)
        if N > 0:
            self.canvas.coords(outer, *self._bbox(x, y, rN))
            self.canvas.itemconfigure(outer, state="normal")
        else:
            self.canvas.itemconfigure(outer, state="hidden")

        # Sarkanais aplis (inficētie)
        if I > 0:
            self.canvas.coords(inner, *self._bbox(x, y, rI))
            self.canvas.itemconfigure(inner, state="normal")
            # Sarkanais aplis vienmēr virs melnā
            self.canvas.tag_raise(inner, outer)
        else:
            self.canvas.itemconfigure(inner, state="hidden")

        # Uzraksts virs centrs–12px
        self.canvas.coords(label, x, y - 12)

    # --------------------------------------------------
    # VISAS KARTES ATJAUNOŠANA
    # --------------------------------------------------
    def update_all(self, day=None):
        """
        Atjauno visu pilsētu attēlojumu.
        Tiek izsaukta katrā simulācijas solī.
        """
        for city in self.G.nodes:
            self.update_city(city)

        # Atjaunojam loga virsrakstu ar dienas numuru
        if day is not None:
            self.root.title(f"Black Death map — Day {day}")


# ==== NOKLUSĒJUMA PARAMETRI VISĀM PILSĒTĀM UN TĪKLAM ====

DEFAULT_PARAMS = {
    "beta":          0.099,
    "gamma":         0.0225,
    "mu":            0.009,
    "cap_frac":      0.08,
    "overload_mult": 1.0,
    "commute_rate":  0.00012,
    "super_period":  14,
    "super_prob":    0.50,
    "events":        2,
    "k_min":         80,
    "k_max":         300,
    "spike_rate":    0.0008,
    "rate_mult":     80.0,
    "total_days":    2554,
    "step_ms":       40,
}


def apply_global_params_to_cities(G, beta, gamma, mu, cap_frac, overload_mult):
    """
    Uzliek vienus un tos pašus bāzes parametrus visām pilsētām.
    Katram parametram pieliekam mazu random (piemēram, +-10%), lai pilsētas nav identiskas.
    Tas ir veids, kā imitēt, ka reālajā dzīvē pilsētām atšķiras apstākļi.
    Tāpēc katrai pilsētai ir mazliet savi beta/gamma/mu un kapacitāte.
    """
    for c in G.nodes:
        c.beta = beta * random.uniform(0.90, 1.10) # +-10%
        c.gamma = gamma * random.uniform(0.90, 1.10) # +-10%
        c.mu = mu * random.uniform(0.85, 1.15) # +-15%
        c.cap_frac = cap_frac * random.uniform(0.8, 1.2) # +-20%
        c.overload_mult = overload_mult

# PARAMETRU NOSAUKUMI TKINTER LOGAM
# Šo vārdnīcu izmantojam, lai Tkinter logā nerādītu “beta”, “gamma”, utt., bet rādītu saprotamus aprakstus.
PARAM_LABELS = {
    "beta": "Inficēšanās koeficients (β)",
    "gamma": "Atveseļošanās koeficients (γ)",
    "mu": "Mirstības koeficients (μ)",

    "cap_frac": "Veselības sistēmas kapacitāte",
    "overload_mult": "Pārslodzes mirstības reizinātājs",

    "commute_rate": "Ceļošanas intensitāte",

    "super_period": "Uzliesmojumu biežums (dienas)",
    "super_prob": "Uzliesmojuma varbūtība",
    "events": "Vienlaicīgo uzliesmojumu skaits",

    "k_min": "Inficēto skaits uzliesmojumā (min)",
    "k_max": "Inficēto skaits uzliesmojumā (max)",

    "spike_rate": "Uzliesmojuma bāzes intensitāte",
    "rate_mult": "Uzliesmojuma reizinātājs",

    "total_days": "Simulācijas ilgums (dienas)",
    "step_ms": "Animācijas ātrums (ms)"
}

def run_sim_with_tk(G, compute_commute_forces, super_commute_spikes):
    """
    Galvenā Tkinter funkcija.
    1) Parāda parametru ievades logu (globālie parametri)
    2) Atver atsevišķu logu ar pilsētu sarakstu, kur var iestatīt sākotnējos saslimušos
    3) Palaiž pilnekrāna karti un animāciju (diena pēc dienas)
    """
    # Izveidojam galveno Tkinter logu
    root = tk.Tk()
    root.title("Black Death – parametru iestatīšana")

    form = tk.Frame(root) # Frame ir konteineris, kur saliekam label + entry
    form.pack(padx=10, pady=10, fill="both", expand=True)

    param_entries = {}

    # Parametri un to datu tipi (float/int)
    param_types = {
        "beta": float,
        "gamma": float,
        "mu": float,
        "cap_frac": float,
        "overload_mult": float,
        "commute_rate": float,
        "super_period": int,
        "super_prob": float,
        "events": int,
        "k_min": int,
        "k_max": int,
        "spike_rate": float,
        "rate_mult": float,
        "total_days": int,
        "step_ms": int,
    }

    row = 0
    for name, default_val in DEFAULT_PARAMS.items():
        label_text = PARAM_LABELS.get(name, name)

        tk.Label(form, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)

        e = tk.Entry(form, width=12)
        e.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        e.insert(0, str(default_val))

        param_entries[name] = e
        row += 1


    status_var = tk.StringVar()
    status_label = tk.Label(form, textvariable=status_var, fg="red")
    status_label.grid(row=row + 1, column=0, columnspan=2, pady=(0, 8))

    def parse_param(name):
        """Nolasa vienu parametru no Entry, pārvērš uz vajadzīgo tipu."""
        txt = param_entries[name].get().strip()
        default = DEFAULT_PARAMS[name]
        typ = param_types[name]

        if txt == "":
            return default
        try:
            val = typ(txt)
        except ValueError:
            return default
        return val


    def is_float(s):
        # atļaujam arī komatu kā decimālo atdalītāju
        s = s.strip().replace(",", ".")
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_int(s):
        s = s.strip()
        if s == "":
            return False
        # lai "12.3" netiktu int laukā
        if "." in s or "," in s: # Ja ir punkts/komats, tad tas nav int
            return False
        try:
            int(s)
            return True
        except ValueError:
            return False

    def validate_inputs():
        """
        Pārbauda visus laukus.
        Ja ir kļūda:
        parāda tekstu status_label
        bloķē pogu “Turpināt”

        Ja viss ok:
        notīra statusu
        atbloķē pogu “Turpināt”
        """
        ok = True
        msg = ""

        # (1) pārbaudām float laukus
        float_fields = ["beta", "gamma", "mu", "cap_frac", "overload_mult",
                        "commute_rate", "super_prob", "spike_rate", "rate_mult"]

        for name in float_fields:
            txt = param_entries[name].get().strip()
            if not is_float(txt):
                ok = False
                msg = f"Kļūda: {name} nav decimālskaitlis"
                break

        # (2) pārbaudām int laukus
        if ok:
            int_fields = ["super_period", "events", "k_min", "k_max", "total_days", "step_ms"]
            for name in int_fields:
                txt = param_entries[name].get().strip()
                if not is_int(txt):
                    ok = False
                    msg = f"Kļūda: {name} nav vesels skaitlis"
                    break

        # (3) pārbaudām minimālās robežas, lai nebūtu absurdi parametri
        if ok:
            super_period = int(param_entries["super_period"].get())
            k_min = int(param_entries["k_min"].get())
            k_max = int(param_entries["k_max"].get())
            total_days = int(param_entries["total_days"].get())
            step_ms = int(param_entries["step_ms"].get())

            if super_period < 1:
                ok = False
                msg = "Kļūda: super_period jābūt >= 1"
            elif k_min < 1:
                ok = False
                msg = "Kļūda: k_min jābūt >= 1"
            elif k_max < k_min:
                ok = False
                msg = "Kļūda: k_max nedrīkst būt mazāks par k_min"
            elif total_days < 1:
                ok = False
                msg = "Kļūda: total_days jābūt >= 1"
            elif step_ms < 1:
                ok = False
                msg = "Kļūda: step_ms jābūt >= 1"

        # Rezultāts: atbloķējam vai bloķējam pogu
        if ok:
            status_var.set("")
            next_btn.config(state="normal")
        else:
            status_var.set(msg)
            next_btn.config(state="disabled")

        return ok

    def bind_validation(entry):
        entry.bind("<KeyRelease>", lambda e: validate_inputs())
        entry.bind("<FocusOut>",  lambda e: validate_inputs())

    # piesienam validāciju visiem Entry
    for e in param_entries.values():
        bind_validation(e)

    def show_initial_infected_window(sim_params):
        """
        Atver jaunu logu ar pilsētu sarakstu un ritjoslu, lai iestatītu sākotnējos saslimušos.
        Šeit lietotājs izvēlas, kur mēris sākas (vai cik slimi ir katrā pilsētā).
        """
        inf_win = tk.Toplevel(root)
        inf_win.title("Sākotnējie saslimušie pa pilsētām")

        # Galvenais ietvars
        main_frame = tk.Frame(inf_win)
        main_frame.pack(fill="both", expand=True)

        # Canvas + ritjosla
        canvas = tk.Canvas(main_frame, borderwidth=0)
        vscroll = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        def on_frame_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner_frame.bind("<Configure>", on_frame_config)

        # Saturs: virsraksts + pilsētu saraksts
        row2 = 0
        tk.Label(inner_frame, text="Sākotnējie saslimušie (I) katrā pilsētā:", font=("Arial", 10, "bold")).grid(row=row2, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 8))
        row2 += 1

        infected_entries = {}

        # Kārtojam pilsētas pēc nosaukuma, lai saraksts būtu skaists
        for city in sorted(G.nodes, key=lambda c: c.name):
            tk.Label(inner_frame, text=city.name).grid(row=row2, column=0, sticky="w", padx=5, pady=1)
            e = tk.Entry(inner_frame, width=8)
            e.grid(row=row2, column=1, sticky="w", padx=5, pady=1)

            # Noklusējums — tas, kas jau ir city.I (piemēram, Sarai = 100)
            e.insert(0, str(int(city.I)))

            infected_entries[city] = e
            row2 += 1

        def start_simulation():
            """
            Nolasa visus ievadītos I un uzliek tos uz pilsētām.
            Tad aizver šo logu un palaiž pašu simulāciju + karti.
            """

            # Uzliek sākotnējos saslimušos pa pilsētām
            for city, entry in infected_entries.items():
                txt = entry.get().strip()
                default_I = int(city.I)
                if txt == "":
                    new_I = default_I
                else:
                    try:
                        new_I = int(txt)
                    except ValueError:
                        new_I = default_I

                if new_I < 0:
                    new_I = 0

                # Kopējā sākotnējā populācija šai pilsētai (no initial_pop)
                N0 = initial_pop[city]
                if new_I > N0:
                    new_I = N0

                city.I = new_I
                city.R = 0
                city.D = 0
                city.S = N0 - new_I

            # Aizver logu ar sākotnējiem saslimušajiem
            inf_win.destroy()

            #
            root.title("Black Death – Eiropas karte")
            root.state("zoomed")  # Windows pilnekrāna režīms
            root.update_idletasks()
            W = root.winfo_width()
            H = root.winfo_height()

            view = MapView(root, G, width=W, height=H, scale_k_alive=0.15, scale_k_inf=0.25)

            day_state = {"day": 0}
            rng_shocks = random

            total_days_loc = sim_params["total_days"]
            step_ms_loc = sim_params["step_ms"]

            def one_step():
                """
                Viena simulācijas diena:
                1) aprēķina ārējo infekcijas spiedienu (parasto + uzliesmojumus)
                2) izsauc city.step() katrai pilsētai (SIRD katrai pilsētai)
                3) atjauno karti
                """
                # palielinām dienu par 1
                d = day_state["day"] + 1
                day_state["day"] = d

                # ja pārsniedzam kopējo dienu skaitu, tad apturam simulāciju
                if d > total_days_loc:
                    return

                # parastais (ikdienas) ārējais spiediens no kaimiņiem
                base_ext = compute_commute_forces(G, commute_rate=sim_params["commute_rate"])

                # retie uzliesmojumi (super-spikes), kas reizēm notiek
                spike_ext = super_commute_spikes(
                    G, d,
                    super_period=sim_params["super_period"],
                    super_prob=sim_params["super_prob"],
                    events=sim_params["events"],
                    k_min=sim_params["k_min"], k_max=sim_params["k_max"],
                    spike_rate=sim_params["spike_rate"],
                    rate_mult=sim_params["rate_mult"],
                    rng=rng_shocks
                )

                # saliekam kopējo spiedienu
                ext = {u: base_ext[u] + spike_ext.get(u, 0.0) for u in G.nodes}

                for city in G.nodes:
                    city.step(external_infection_force=ext[city])

                # pārzīmējam karti uz jauno dienu
                view.update_all(day=d)

                if d < total_days_loc:
                    root.after(step_ms_loc, one_step)

            # sākuma stāvoklis (0. diena)
            view.update_all(day=0)
            root.after(step_ms_loc, one_step)

        # Poga, kas palaiž simulāciju (pēc I ievades)
        start_btn2 = tk.Button(inner_frame, text="Sākt simulāciju", command=start_simulation)
        start_btn2.grid(row=row2 + 1, column=0, columnspan=2, pady=10)

    def on_next():
        """
        POGA “TURPINĀT” (NO 1. LOGA UZ 2. LOGU)
        Pēc globālo parametru ievades atver logu ar sākotnējiem saslimušajiem.
        """
        # Ja validācija neiziet, tad neko nedarām
        if not validate_inputs():
            return

        beta = parse_param("beta")
        gamma = parse_param("gamma")
        mu = parse_param("mu")
        cap_frac = parse_param("cap_frac")
        overload_mult = parse_param("overload_mult")

        commute_rate = parse_param("commute_rate")
        super_period = max(1, parse_param("super_period"))
        super_prob = parse_param("super_prob")
        events = max(0, parse_param("events"))
        k_min = max(1, parse_param("k_min"))
        k_max = max(k_min, parse_param("k_max"))
        spike_rate = parse_param("spike_rate")
        rate_mult = parse_param("rate_mult")
        total_days_loc = parse_param("total_days")
        step_ms_loc = parse_param("step_ms")

        # super_prob ieliekam robežās [0,1]
        if super_prob < 0:
            super_prob = 0.0
        elif super_prob > 1:
            super_prob = 1.0

        # Uzliek globālos parametrus visām pilsētām
        apply_global_params_to_cities(G, beta, gamma, mu, cap_frac, overload_mult)

        # Saglabā parametrus vārdnīcā, lai vēlāk izmantotu simulācijā
        sim_params = {
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "cap_frac": cap_frac,
            "overload_mult": overload_mult,
            "commute_rate": commute_rate,
            "super_period": super_period,
            "super_prob": super_prob,
            "events": events,
            "k_min": k_min,
            "k_max": k_max,
            "spike_rate": spike_rate,
            "rate_mult": rate_mult,
            "total_days": total_days_loc,
            "step_ms": step_ms_loc,
        }

        # Paslēpjam pirmo logu, lai tas netraucētu (bet root paliek dzīvs)
        root.withdraw()

        # Noņemam formu (lai vairs nav redzama)
        form.destroy()

        # Atver otro logu ar pilsētu sarakstu
        show_initial_infected_window(sim_params)

    # INFO POGAS FUNKCIONALITĀTE
    info_text =  """=======================================================
          KĀ DARBOJAS "MELNĀS NĀVES" SIMULĀCIJA?
=======================================================

Šī programma modelē mēra izplatību 14. gadsimta Eiropas kartē. 
Simulācija notiek soli pa soli (solis = 1 diena).

1. KAS NOTIEK PILSĒTAS IEKŠIENĒ? (SIRD MODELIS)
Katrā pilsētā iedzīvotāji tiek iedalīti 4 grupās. Katru dienu cilvēki pārvietojas starp tām:

   [S] Veselie (Susceptible)
    |  Vēl nav slimojuši. Viņus apdraud kontakts ar inficētajiem [I].
    v
   [I] Inficētie (Infected)
    |  Šobrīd slimo un izplata mēri tālāk.
    |  Viņiem ir divi iespējamie ceļi:
    |
    +-----> [R] Izveseļojušies (Recovered)
    |           Iegūst imunitāti un vairs nevar saslimst.
    |
    +-----> [D] Mirušie (Dead)
                Diemžēl izstājas no populācijas.

2. VESELĪBAS SISTĒMAS PĀRSLODZE
Simulācijā ir izveidots "Veselības sistēmas pārslodzes" mehānisms.
   - Normālā situācijā: Mirstība ir bāzes līmenī.
   - Krīzes situācijā: Ja slimo skaits pārsniedz "Kapacitātes slieksni", 
     tad mirstība strauji pieaug.

3. KĀ MĒRIS CEĻO?
Pilsētas nav izolētas. Tās savieno tirdzniecības ceļi (līnijas kartē).
   - Infekcija "pārtek" no stipri inficētām pilsētām uz kaimiņiem.
   - Jo lielāka pilsēta un intensīvāka satiksme, jo ātrāk mēris izplatās.

4. ĀRKĀRTAS NOTIKUMI
Papildus parastajai plūsmai, modelis simulē retus, negaidītus notikumus (piemēram, ostā ierodas kuģis ar inficētiem). Tas rada pēkšņus uzliesmojumus vietās, kas likās drošas.

=======================================================
            KO NOZĪMĒ IEVADES PARAMETRI?
=======================================================

--- SLIMĪBAS RAKSTUROJUMS ---

> Inficēšanās varbūtība (β)
  Ko tas dara: Nosaka mēra lipīgumu. Tā ir varbūtība, ka veselais (S) saslims vienas dienas laikā, ja nonāks kontaktā.
  [Robežas]: 0.0 līdz 1.0 (decimālskaitlis).
  (Piemēram: 0.1 nozīmē ļoti lipīgu mēri, 0.01 — mazāk lipīgu mēri).

> Izveseļošanās rādītājs (γ)
  Ko tas dara: Nosaka vidējo slimošanas ilgumu. Matemātiski: 1 / dienu skaits.
  [Robežas]: 0.0 līdz 1.0.
  (Piemēram: 0.1 nozīmē, ka cilvēks slimo vidēji 10 dienas; 0.05 = 20 dienas).

> Mirstības rādītājs (μ)
  Ko tas dara: Bāzes varbūtība nomirt vienas dienas laikā (ja slimnīcās ir vietas).
  [Robežas]: 0.0 līdz 1.0.
  (Piemēram: 0.01 ir augsta mirstība, 0.001 ir zema).

--- VESELĪBAS SISTĒMA (SLIMNĪCAS) ---

> Veselības sistēmas slieksnis (%)
  Ko tas dara: Nosaka, cik lielu daļu no populācijas ārsti spēj aprūpēt vienlaicīgi.
  [Robežas]: 0.0 līdz 1.0.
  (Piemēram: 0.08 nozīmē 8% no iedzīvotājiem. Ja slimo 10%, sākas krīze).

> Mirstības pieauguma faktors
  Ko tas dara: Soda reizinātājs. Cik reizes pieaug mirstība, ja sistēma ir pārslogota.
  [Robežas]: 0.0 (nav soda) līdz ∞ (liels sods).
  (Piemēram: 1.0 nozīmē proporcionālu pieaugumu, 2.0 nozīmē divkāršu mirstību).

--- MOBILITĀTE (ĢEOGRĀFIJA) ---

> Starppilsētu migrācijas plūsma
  Ko tas dara: Cik intensīvi mēris ceļo starp pilsētām ikdienā (bez ārkārtas notikumiem).
  [Robežas]: 0.0 līdz 1.0 (parasti ļoti mazs skaitlis, piem., 0.0001).

--- ĀRKĀRTAS UZLIESMOJUMI (SUPER-SPREADERS) ---

> Uzliesmojumu pārbaudes cikls
  Ko tas dara: Cik bieži (dienās) sistēma "met kauliņu", lai mēģinātu izraisīt notikumu.
  [Robežas]: Vesels skaitlis > 0 (piem., 14 dienas).

> Uzliesmojuma iespējamība
  Ko tas dara: Ja ir pienākusi pārbaudes diena, cik liela ir iespēja, ka notikums tiešām notiks.
  [Robežas]: 0.0 līdz 1.0 (piem., 0.5 = 50% iespēja).

> Vienlaicīgo perēkļu skaits
  Ko tas dara: Cik pilsētās vienlaicīgi parādīsies mēris, ja notikums ir aktivizējies.
  [Robežas]: Vesels skaitlis >= 1.

> Inficēto skaits perēklī (min / max)
  Ko tas dara: Cik jauni slimi cilvēki pēkšņi uzrodas pilsētā (notikuma mērogs).
  [Robežas]: Veseli skaitļi (piem., 80 un 300).

> Uzliesmojuma bāzes ietekme (spike_rate)
  Ko tas dara: Tehniskā infekcijas intensitāte notikuma brīdī (pirms reizināšanas).
  [Robežas]: 0.0 līdz 1.0 (parasti mazs, piem., 0.0008).

> Ietekmes mērogs (reizinātājs)
  Ko tas dara: Cik reizes šis notikums ir spēcīgāks par parastu dienu.
  [Robežas]: Skaitlis > 1.0 (piem., 80.0 padara notikumu ļoti bīstamu).

--- TEHNISKIE IESTATĪJUMI ---

> Simulācijas ilgums
  Ko tas dara: Cik dienas programma darbosies.
  [Robežas]: Vesels skaitlis (piem., 2554 dienas = 7 gadi).

> Animācijas ātrums
  Ko tas dara: Pauze starp dienām milisekundēs.
  [Robežas]: Mazāks skaitlis = ātrāka animācija (piem., 40ms).
"""

    def show_info_window():
        # Uzreiz atslēdzam pogu "info", lai lietotājs nevarētu atvērt vairākus info logus vienlaicīgi
        info_btn.config(state="disabled")

        # Izveidojam jaunu atsevišķu logu (Toplevel)
        info_win = tk.Toplevel(root)
        info_win.title("Informācija")

        # Funkcija, kas izpildās, kad info logs tiek aizvērts
        def on_close():
            # aizveram info logu
            info_win.destroy()
            # atkal ieslēdzam pogu "info"
            info_btn.config(state="normal")

        # noķeram aizvēršanu ar loga krustiņu (WM_DELETE_WINDOW)
        info_win.protocol("WM_DELETE_WINDOW", on_close)

        # vertikālā ritjosla teksta laukam
        text_area = tk.Text(info_win, wrap="word", padx=10, pady=10, width=60, height=20)
        text_area.pack(side="left", fill="both", expand=True)

        scroll = tk.Scrollbar(info_win, command=text_area.yview)
        scroll.pack(side="right", fill="y")
        text_area.config(yscrollcommand=scroll.set)

        # Ieliekam tekstu un padarām lauku readonly
        text_area.insert("1.0", info_text)
        text_area.config(state="disabled")

    next_btn = tk.Button(form, text="Turpināt", command=on_next)
    next_btn.grid(row=row + 2, column=0, columnspan=2, pady=(10, 2))

    # Uzreiz pārbaudām, vai ievade ir korekta
    validate_inputs()

    # Poga "i" (info)
    info_btn = tk.Button(form, text="info", command=show_info_window, fg="blue")
    info_btn.grid(row=row + 3, column=0, columnspan=2, pady=(2, 10))

    root.mainloop()


# ==== PALAIŽAM TKINTER AR DIVIEM LOGIEM ====
run_sim_with_tk(G, compute_commute_forces, super_commute_spikes)