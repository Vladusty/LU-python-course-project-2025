import networkx as nx  # bibliotēka darbam ar grafiem (pilsētu savienojumiem)
import random           # bibliotēka nejaušiem skaitļiem (random izvēlēm)

class City:
    """
    KLASE "City" – pilsēta ar iedzīvotājiem un slimības stāvokli
    """
    def __init__(self, name, population, city_type="town"):
        # Saglabājam pilsētas pamatdatus
        self.name = name                # pilsētas nosaukums
        self.city_type = city_type      # pilsētas tips (city/town/village/rural)
        self.population = population    # kopējais iedzīvotāju skaits

        # SIR modelis: cilvēku sadalījums pa grupām
        self.S = population  # S – veseli cilvēki, kas var saslimst
        self.I = 0           # I – inficētie
        self.R = 0           # R – izveseļojušies
        self.D = 0           # D – mirušie

        # Uzreiz iestatām parametrus atkarībā no pilsētas tipa
        self.set_parameters_by_type()


    def set_parameters_by_type(self):
        """Šī funkcija nosaka slimības un kustības parametrus atkarībā no pilsētas tipa."""

        # Noklusētās vērtības (ja tips nav norādīts)
        #self.beta = 0.10     # inficēšanās ātrums
        #self.gamma = 0.04    # izveseļošanās ātrums
        #self.mu = 0.06       # mirstības rādītājs
        #self.movement = 0.003 # pārvietošanās (cilvēku ceļošana uz citām pilsētām)

        # Atšķirīgi parametri dažādiem pilsētu tipiem:
        if self.city_type == "city":
            self.beta = 0.32     # pilsētās mēris izplatās ļoti ātri
            self.gamma = 0.04
            self.mu = 0.022
            self.movement = 0.045  # daudz tirgotāju, kustība starp pilsētām biežāka
        elif self.city_type == "town":
            self.beta = 0.24
            self.gamma = 0.04
            self.mu = 0.012
            self.movement = 0.03
        elif self.city_type == "village":
            self.beta = 0.15
            self.gamma = 0.04
            self.mu = 0.005
            self.movement = 0.01
        elif self.city_type == "rural":
            self.beta = 0.08
            self.gamma = 0.04
            self.mu = 0.005
            self.movement = 0.005


    def step(self):
        """Vienas dienas simulācija konkrētajā pilsētā."""
        # Aprēķinām, cik cilvēku šobrīd dzīvi
        living = self.S + self.I + self.R

        # Ja pilsēta tukša vai nav slimnieku, neko nedaram
        if living <= 0 or self.I == 0:
            return

        # Aprēķinām jaunos slimniekus (pēc SIR modeļa formulas)
        new_infected = int(self.beta * self.S * self.I / living)

        # Aprēķinām cik izveseļojās un cik nomira
        recoveries = int(self.gamma * self.I)
        deaths = int(self.mu * self.I)

        # Atjaunojam iedzīvotāju grupas
        self.S -= new_infected
        self.I += new_infected - recoveries - deaths
        self.R += recoveries
        self.D += deaths

        # Neļaujam, lai skaits kļūtu negatīvs
        self.S = max(self.S, 0)
        self.I = max(self.I, 0)
        self.R = max(self.R, 0)


    def infect(self, n):
        """Inficē n cilvēkus šajā pilsētā simulācijas sākumā."""
        n = min(n, self.S)   # nevar inficēt vairāk cilvēku nekā ir veselo
        self.S -= n
        self.I += n


    def __repr__(self):
        """Izdrukājot pilsētu, redzams tās stāvoklis."""
        living = self.S + self.I + self.R

        return f"{self.name}: Pop = {living} S={self.S} I={self.I} R={self.R} D={self.D}"



# FUNKCIJA simulate_step – izpilda vienu dienu visā grafā (visās pilsētās)
def simulate_step(G: nx.Graph):
    """Simulē vienu dienu visās pilsētās un starp tām."""

    # 1) Pirmkārt, katra pilsēta atjauno savu iekšējo SIR stāvokli
    for city in G.nodes:
        city.step()

    # 2) Aprēķinām cilvēku kustību starp pilsētām
    movements = {city: {"S": 0, "I": 0, "R": 0} for city in G.nodes}

    for city in G.nodes:
        living = city.S + city.I + city.R
        if living == 0:
            continue

        # cik daļa cilvēku šodien pārvietojas
        move_fraction = city.movement
        movers_total = int(living * move_fraction)
        if movers_total == 0:
            continue

        # atrodam pilsētas kaimiņus
        neighbors = G[city]
        total_weight = sum(data.get("weight", 1.0) for _, data in neighbors.items())
        if total_weight == 0:
            continue

        # proporcija starp S, I, R grupām
        prop_S = city.S / living
        prop_I = city.I / living
        prop_R = city.R / living

        # pārvieto cilvēkus uz kaimiņpilsētām
        for neighbor, data in neighbors.items():
            weight = data.get("weight", 1.0)
            frac = weight / total_weight
            movers = int(round(movers_total * frac))
            if movers == 0:
                continue

            # cik pārvietojas no katras grupas
            mS = min(int(round(movers * prop_S)), city.S)
            mI = min(int(round(movers * prop_I)), city.I)
            mR = min(movers - mS - mI, city.R)

            # atņemam no šīs pilsētas
            city.S -= mS
            city.I -= mI
            city.R -= mR

            # pievienojam citai pilsētai
            movements[neighbor]["S"] += mS
            movements[neighbor]["I"] += mI
            movements[neighbor]["R"] += mR

    # 3) Kad pārvietošanās aprēķināta, pievienojam jaunpienākušos cilvēkus katrā pilsētā
    for city, moved in movements.items():
        city.S += moved["S"]
        city.I += moved["I"]
        city.R += moved["R"]


# GALVENĀ PROGRAMMA – simulācijas piemērs
if __name__ == "__main__":
    random.seed(42)  # lai rezultāti katru reizi būtu vienādi

    # Izveidojam dažas pilsētas
    riga = City("Riga", 6500, "town")
    tallinn = City("Reval", 4000, "town")
    tartu = City("Derpt", 2000, "town")
    novgorod = City("Novgorod", 20000, "city")
    pskov = City("Pskov", 10000, "town")

    # Izveidojam grafu (pilsētu tīklu)
    G = nx.Graph()
    for city in [riga, tallinn, tartu, novgorod, pskov]:
        G.add_node(city)

    # Pievienojam savienojumus (ceļus starp pilsētām)
    # weight – cik spēcīgs savienojums (tirdzniecības ceļš)
    G.add_edge(riga, tartu, weight=0.01)
    G.add_edge(riga, pskov, weight=0.01)
    G.add_edge(riga, tallinn, weight=0.03)
    G.add_edge(riga, novgorod, weight=0.04)
    G.add_edge(tallinn, tartu, weight=0.02)
    G.add_edge(tallinn, novgorod, weight=0.05)
    G.add_edge(tartu, pskov, weight=0.02)
    G.add_edge(pskov, novgorod, weight=0.02)

    # Sākotnējā infekcija: Rīgā 100 cilvēki slimi
    riga.infect(100)

    # Simulējam 50 dienas
    days = 50
    for day in range(1, days + 1):
        simulate_step(G)
        print(f"\n--- {day}. diena ---")
        for city in G.nodes:
            print(city)