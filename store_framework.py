
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
            self.shelfs.append(Shelf(_, 20, Product(False)))

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

    def __init__(self, on_promotion):

        self.ean = 1111111111111
        self.name = 'Apple'
        self.on_promotion = on_promotion
        self.popularity = .01
        self.seasonality = 0
