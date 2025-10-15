
"""
The Black Death occurred in Europe from 1346 to 1353.

List of cities:
https://www.reddit.com/r/europe/comments/4kdvo1/the_thirty_largest_cities_in_europe_by_population/#lightbox

"""

import random

class Province:
    def __init__(self, province_name: str, population: int, province_type="town"):
        """
        Creates a province (a city) (like a graph)

        name        — name of the province/city
        population  —  total population of the province/city
        province_type   — type ("city", "town", "village", "rural")
        """

        self.province_name = province_name
        self.province_type = province_type
        self.base_population = population

        self.susceptible = population
        self.infected = 0
        self.recovered = 0
        self.dead = 0

        self.neighbors_by_land = {}
        self.neighbors_by_sea = {}

        # buffer for incoming movers during a global step
        self._incoming = {"S": 0, "I": 0, "R": 0}

    def set_parameters_by_type(self):
        # base *per day* parameters
            """Assign β (how fast disease is travelling), γ (recovery speed), μ (death speed) depending on province type."""

            self.beta = 0.10  #  how fast disease is travelling
            self.gamma = 0.04  # recovery speed (not dependent on terrain type)
            self.mu = 0.06  # death fraction per day

            self.movement_base = 0.003  # fraction of living that may move per day

            if self.province_type == "city":
                self.beta = 0.32
                self.gamma = 0.04
                self.mu = 0.022
                self.movement_base = 0.045

            elif self.province_type == "town":
                self.beta = 0.24
                self.gamma = 0.04
                self.mu = 0.012
                self.movement_base = 0.03

            elif self.province_type == "village":
                self.beta = 0.15
                self.gamma = 0.04
                self.mu = 0.005
                self.movement_base = 0.01

            elif self.province_type == "rural":
                self.beta = 0.08
                self.gamma = 0.04
                self.mu = 0.005
                self.movement_base = 0.005


    def add_neighbor(self, neighbor, connection_strength: float, connection_type="land", bidirectional=True):
        """Add relation by sea or land. Optionally make it bidirectional."""
        if connection_type == "land":
            self.neighbors_by_land[neighbor] = connection_strength
            if bidirectional:
                neighbor.neighbors_by_land[self] = connection_strength
        elif connection_type == "sea":
            self.neighbors_by_sea[neighbor] = connection_strength
            if bidirectional:
                neighbor.neighbors_by_sea[self] = connection_strength
        else:
            raise ValueError("connection_type must be 'land' or 'sea'")

    def infect(self, number: int):
        """Manually infect a number of people at the start."""
        number = int(number)
        number = min(number, self.susceptible)
        self.susceptible -= number
        self.infected += number


    @staticmethod
    def infect_province_by_name(provinces, name, number):
        """Find province by name and infect it with `number` people."""
        for p in provinces:
            p.set_parameters_by_type()
        for p in provinces:
            if p.province_name == name:
                p.infect(number)
                print(f"{number} people infected in {p.province_name}")
                return None

        print(f"No province found with name '{name}'")
        return None

    def step_local(self):
        """Simulate local disease dynamics (SIR + deaths) for one step in this province."""
        living = self.susceptible + self.infected + self.recovered
        if living <= 0 or self.infected == 0:
            return None # nothing happens if no infected people or no living people

        # New infections within province
        new_infections = int(self.beta * (self.infected / living) * self.susceptible)
        new_infections = max(0, min(new_infections, self.susceptible))

        recoveries = int(self.gamma * self.infected)
        deaths = int(self.mu * self.infected)

        # Update counts
        self.susceptible -= new_infections
        self.infected += new_infections - recoveries - deaths
        self.recovered += recoveries
        self.dead += deaths

        # Guard against negative (due to rounding)
        if self.susceptible < 0: self.susceptible = 0
        if self.infected < 0: self.infected = 0
        if self.recovered < 0: self.recovered = 0

    def compute_outgoing_moves(self):
        living = self.susceptible + self.infected + self.recovered
        if living <= 0:
            return []

        infected_ratio = self.infected / living

        # realistic quarantine / flee multipliers
        if infected_ratio > 0.3:  # если более 30% заражены, люди меньше двигаются
            quarantine_multiplier = 0.1
        else:
            quarantine_multiplier = 1.0

        flee_multiplier = 1.2 if 0.0 < infected_ratio < 0.3 else 1.0

        base_frac = self.movement_base * quarantine_multiplier * flee_multiplier
        noise = random.uniform(0.9, 1.1)  # меньше шума, более стабильное движение
        base_frac *= noise

        max_movers = int(base_frac * living)
        max_movers = min(max_movers, int(living))

        # gather neighbours
        neighbours = {}
        neighbours.update(self.neighbors_by_land)
        neighbours.update(self.neighbors_by_sea)
        if not neighbours or max_movers <= 0:
            return []

        total_strength = sum(neighbours.values())
        strengths = neighbours if total_strength > 0 else {n: 1.0 for n in neighbours}
        total_strength = total_strength if total_strength > 0 else len(neighbours)

        outgoing = []
        prop_S = self.susceptible / living
        prop_I = self.infected / living
        prop_R = self.recovered / living

        for neighbor, strength in strengths.items():
            frac = strength / total_strength
            movers_total = int(round(max_movers * frac))
            if movers_total <= 0:
                continue

            movers_S = min(int(round(movers_total * prop_S)), self.susceptible)
            movers_I = min(int(round(movers_total * prop_I)), self.infected)
            movers_R = min(movers_total - movers_S - movers_I, self.recovered)

            if movers_S + movers_I + movers_R > 0:
                outgoing.append((neighbor, {"S": movers_S, "I": movers_I, "R": movers_R}))

        return outgoing

    def apply_outgoing(self, movers_list):
        for movers in movers_list:
            mS = movers.get("S", 0)
            mI = movers.get("I", 0)
            mR = movers.get("R", 0)
            self.susceptible = max(0, self.susceptible - mS)
            self.infected = max(0, self.infected - mI)
            self.recovered = max(0, self.recovered - mR)

    def receive_incoming(self):
        """Apply accumulated incoming (from other provinces)"""
        self.susceptible += self._incoming["S"]
        self.infected += self._incoming["I"]
        self.recovered += self._incoming["R"]
        # reset buffer and update population (dead are not moving)
        self._incoming = {"S": 0, "I": 0, "R": 0}

    def __repr__(self):
        living = self.susceptible + self.infected + self.recovered
        return (f"<{self.province_name}: base_pop={self.base_population} living={living} "
                f"S={self.susceptible} I={self.infected} R={self.recovered} D={self.dead}>")


    @staticmethod
    def print_all_provinces(provinces):
        print("=== Province Overview ===")
        for p in provinces:
            living = p.susceptible + p.infected + p.recovered
            print(f"{p.province_name} ({p.province_type}) base_pop: {p.base_population} | living={living} | "
                  f"S={p.susceptible} I={p.infected} R={p.recovered} D={p.dead}")


    @staticmethod
    def print_all_provinces_with_connections(provinces):
        print("=== Province Overview with Connections ===")
        for p in provinces:
            print(f"\n{p.province_name} ({p.province_type}) base_pop: {p.base_population}")
            print(f"  S={p.susceptible} I={p.infected} R={p.recovered} D={p.dead}")
            if p.neighbors_by_land:
                print("  By land:")
                for n, strength in p.neighbors_by_land.items():
                    print(f"    {n.province_name} (strength={strength})")
            if p.neighbors_by_sea:
                print("  By sea:")
                for n, strength in p.neighbors_by_sea.items():
                    print(f"    {n.province_name} (strength={strength})")


def simulate_step(provinces):
    """
    Perform a single simulation step:
    1) Ensure params set
    2) Local disease dynamics step for each province
    3) Compute outgoing movers for each province (based on AFTER local dynamics counts)
    4) Apply outgoing (subtract from source) and accumulate incoming in targets
    5) Apply incoming to each province
    """

    for p in provinces:
        p.set_parameters_by_type()

    # local updates
    for p in provinces:
        p.step_local()

    # compute all outgoing
    all_outgoing = {}
    for p in provinces:
        all_outgoing[p] = p.compute_outgoing_moves()

        # subtract outgoing & accumulate incoming
    for src, outgoing in all_outgoing.items():
        movers_dicts = []
        for (target, movers) in outgoing:
            target._incoming["S"] += movers.get("S", 0)
            target._incoming["I"] += movers.get("I", 0)
            target._incoming["R"] += movers.get("R", 0)
            movers_dicts.append(movers)
        if movers_dicts:
            src.apply_outgoing(movers_dicts)

        # apply incoming
    for p in provinces:
        p.receive_incoming()


if __name__ == "__main__":
    random.seed(42)  # reproducible example

    riga = Province("Riga", 7000, "town")
    tallinn = Province("Reval", 4000, "town")
    tartu = Province("Derpt", 2000, "town")
    novgorod = Province("Novgorod", 20000, "city")
    pskov = Province("Pskov", 10000, "town")

    # connections (example strengths)
    riga.add_neighbor(tartu, 0.01, "land", True)
    riga.add_neighbor(pskov, 0.01, "land", True)
    riga.add_neighbor(tallinn, 0.03, "sea", True)
    riga.add_neighbor(novgorod, 0.04, "sea", True)

    tallinn.add_neighbor(tartu, 0.02, "land", True)
    tallinn.add_neighbor(novgorod, 0.02, "land", True)
    tallinn.add_neighbor(novgorod, 0.03, "sea", True)

    tartu.add_neighbor(pskov, 0.02, "land", True)
    pskov.add_neighbor(novgorod, 0.02, "land", True)



    provinces = [riga, tallinn, tartu, novgorod, pskov]

    Province.print_all_provinces(provinces)
    Province.infect_province_by_name(provinces, "Riga", 100)
    print("\nAfter initial infection:")
    Province.print_all_provinces(provinces)

    # simulate several steps
    steps = 100
    for day in range(1, steps + 1):
        simulate_step(provinces)
        print(f"\n--- Day {day} ---")
        Province.print_all_provinces(provinces)




"""
riga = Province("Riga", 6500, "town") # find Riga pop in 1346.
tallinn = Province("Reval", 4000, "town") # find Reval (Tallin) pop in 1346.
tartu = Province("Derpt", 2000, "town") # find Derpt (Tartu) pop in 1346.
novgorod = Province("Novgorod", 20000, "city") # find Novgorod pop in 1346.
pskov = Province("Pskov", 10000, "city") # find Pskov pop in 1346.


"""