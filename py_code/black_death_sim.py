
"""
The Black Death occurred in Europe from 1346 to 1353.

List of cities:
https://www.reddit.com/r/europe/comments/4kdvo1/the_thirty_largest_cities_in_europe_by_population/#lightbox

"""

class Province:
    def __init__(self, province_name: str, population: int, province_type="town"):
        """
        Creates a province (a city) (like a graph)

        name        — нname of the province/city
        population  —  total population of the province/city
        province_type   — тип ("city", "town", "village", "rural" etc)
        """

        self.province_name = province_name
        self.province_type = province_type
        self.population = population

        self.susceptible = population
        self.infected = 0
        self.recovered = 0
        self.dead = 0

        self.neighbors_by_land = {}
        self.neighbors_by_sea = {}


    def set_parameters_by_type(self):
        """Assign β (how fast disease is travelling), γ (recovery speed), μ (death speed) depending on province type."""

        self.beta = 0.25  # how fast disease is travelling)
        self.gamma = 0.1  # ecovery speed (not dependent on terrain type)
        self.mu = 0.01    # death speed

        if self.province_type == "city":
            self.beta = 0.4     # higher density => faster spread
            self.mu = 0.015     # slightly higher mortality
        elif self.province_type == "town":
            self.beta = 0.3
            self.mu = 0.012
        elif self.province_type == "village":
            self.beta = 0.2
            self.mu = 0.009
        elif self.province_type == "rural":
            self.beta = 0.15
            self.mu = 0.008

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

    def step(self):
        """Simulate one day of infection within this province."""
        if self.infected == 0:
            return  # nothing happens if no infected people

        # New infections within province
        new_infections = int(self.beta * (self.infected / self.population) * self.susceptible)
        new_infections = min(new_infections, self.susceptible)

        # Recoveries and deaths
        recoveries = int(self.gamma * self.infected)
        deaths = int(self.mu * self.infected)

        # Update counts
        self.susceptible -= new_infections
        self.infected += new_infections - recoveries - deaths
        self.recovered += recoveries
        self.dead += deaths

    def __repr__(self):
        return (f"<{self.province_name}: "
                f"S={self.susceptible}, I={self.infected}, "
                f"R={self.recovered}, D={self.dead}>")


    @staticmethod
    def print_all_provinces(provinces):
        print("=== Province Overview ===")
        for p in provinces:
            print(f"{p.province_name} ({p.province_type}) pop: {p.population} | "
                  f"S={p.susceptible} I={p.infected} R={p.recovered} D={p.dead}")


    @staticmethod
    def print_all_provinces_with_connections(provinces):
        print("=== Province Overview with Connections ===")
        for p in provinces:
            print(f"\n{p.province_name} ({p.province_type}) pop: {p.population}")
            print(f"  S={p.susceptible} I={p.infected} R={p.recovered} D={p.dead}")
            if p.neighbors_by_land:
                print("  By land:")
                for n, strength in p.neighbors_by_land.items():
                    print(f"    {n.province_name} (strength={strength})")
            if p.neighbors_by_sea:
                print("  By sea:")
                for n, strength in p.neighbors_by_sea.items():
                    print(f"    {n.province_name} (strength={strength})")


riga = Province("Riga", 7000, "town") # find Riga pop in 1346.
tallinn = Province("Reval", 4000, "town") # find Reval (Tallin) pop in 1346.
tartu = Province("Derpt", 2000, "town") # find Derpt (Tartu) pop in 1346.
novgorod = Province("Novgorod", 20000, "city") # find Novgorod pop in 1346.
pskov = Province("Pskov", 10000, "city") # find Pskov pop in 1346.

riga.add_neighbor(tartu, 0.01, "land")
riga.add_neighbor(tallinn, 0.03, "sea")
riga.add_neighbor(novgorod, 0.04, "sea")
riga.add_neighbor(pskov, 0.01, "land")

tallinn.add_neighbor(tartu, 0.02, "land")

provinces = [riga, tallinn, tartu, novgorod, pskov]

#print_all_provinces(provinces)
#print()
#print_all_provinces_with_connections(provinces)


Province.print_all_provinces(provinces)
Province.infect_province_by_name(provinces, "Riga", 100)
Province.print_all_provinces(provinces)

riga.step()

Province.print_all_provinces(provinces)