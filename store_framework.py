import random


class Store():

    def __init__(self, id, occupancy, opening_time, closing_time):

        self.id = id
        self.size = 2
        self.occupancy = occupancy
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.shelfs = []
        self.create_shelfs()

    def create_shelfs(self):
        for _ in range(1):
            cap = random.randint(5, 50)
            self.shelfs.append(Shelf(_, cap, Product()))

    def restock_shelfs(self):

        for x in self.shelfs:
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
        self.name = 'Apple'
        self.popularity = self.generate_popularity()

    def generate_ean(self):
        ean = 1234500000000 + random.randint(0, 99999999)
        return ean

    def generate_popularity(self):
        pop = random.uniform(.005, .15)
        return pop
