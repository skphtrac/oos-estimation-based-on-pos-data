import random
import string


class Store():

    def __init__(self, id, occupancy, opening_time, closing_time):

        self.id = id
        self.occupancy = occupancy
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.shelves = []
        self.create_shelves()

    def create_shelves(self):
        # range indicates the number of shelves in the store
        # range of randint determines shelves capacity
        for _ in range(1):
            cap = random.randint(20, 20)
            self.shelves.append(Shelf(_, cap, Product()))

    def restock_shelves(self):

        for x in self.shelves:
            x.current_stock = x.capacity


class Shelf():

    def __init__(self, id, capacity, product):

        self.id = id
        self.capacity = capacity
        self.current_stock = self.capacity
        self.product = product


class Product():

    def __init__(self):

        self.ean = self.generate_ean()
        self.name = self.generate_name()
        self.demand_factor = self.generate_demand_factor()

    def generate_ean(self):
        ean = 1234500000000 + random.randint(0, 99999999)
        return ean

    def generate_name(self):
        letters = string.ascii_uppercase
        name = ''.join(random.choice(letters) for i in range(5))
        return name

    def generate_demand_factor(self):
        pop = random.uniform(.005, .15)
        return pop
