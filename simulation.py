import datetime
from scipy.interpolate import CubicSpline
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
import random
from scipy.stats import norm, gaussian_kde, expon
import statistics
import math
from sklearn.neighbors import KernelDensity
import collections
from scipy.interpolate import CubicSpline

from sim import Store
from occupancy import Occupancy


class TimeSimulation():

    def __init__(self, store):

        self.store = store
        self.store_occupancy = []
        self.product_graph = []
        self.startdate = None
        self.enddate = None
        self.current_time = None

        self.create_occupancy()

    def simulate_period(self, start, end):

        self.startdate = self.current_time = self.last_time_step = start
        self.enddate = end
        self.scheduled_pos_data = {}
        self.pos_data = []
        self.shelf_data = []
        self.shelf_data_times = []
        self.coc = 0
        time_step = datetime.timedelta(hours=1)

        while self.current_time <= self.enddate:

            if self.current_time.weekday() in {0, 1, 2, 3, 4, 5}:
                time_step = datetime.timedelta(minutes=1)
                if (self.store.opening_time
                    <= self.current_time.time()
                        <= self.store.closing_time):

                    self.time_action()

                elif self.current_time.weekday() != self.last_time_step.weekday() and self.current_time.weekday() in {1, 4}:

                    self.new_day_action()
                else:
                    time_step = datetime.timedelta(minutes=1)

            self.shelf_data.append([
                self.current_time, self.store.shelfs[0].current_stock])

            self.last_time_step = self.current_time
            self.current_time += time_step

    def time_action(self):

        self.coc = self.current_occupancy(self.current_time)

        for shelf in self.store.shelfs:
            if shelf.current_stock > 0:

                self.simulate_shelf_demand(shelf)

                if shelf.current_stock == 0:
                    print(str(shelf.product.name)
                          + ' is OOS at ' + str(self.current_time))

            # if len(self.pos_data) > 1:
            #     self.check_if_oos(shelf)

        if self.current_time in self.scheduled_pos_data:
            max = random.randint(1, 2)
            # for value in self.scheduled_pos_data.get(self.current_time):
            i = 0
            for value in self.scheduled_pos_data[self.current_time]:

                if i < max:
                    self.pos_data.append(
                        PointOfSaleEntry(
                            value[0],
                            value[1],
                            self.current_time
                        )
                    )
                else:
                    purchase_time = self.current_time + \
                        datetime.timedelta(minutes=1)
                    self.schedule_pos_entry(
                            purchase_time,
                            value[0],
                            value[1])
                i += 1

    def simulate_shelf_demand(self, shelf):

        self.product_graph.append([
            self.current_time, shelf.current_stock])

        if random.random() < self.coc * shelf.product.popularity:

            # if random.random() < shelf.product.popularity:

            amount = random.randint(1, 2)
            if amount > shelf.current_stock:
                amount = shelf.current_stock
            shelf.current_stock -= amount

            self.shelf_data_times.append(
                [self.current_time, shelf.current_stock])

            if random.random() > 0.02:  # shrinkage

                for i in range(amount):
                    # product left and retrieved later
                    if random.random() < 0.01:

                        delay = random.randint(0, self.time_til_closing())
                        purchase_time = self.current_time + \
                            datetime.timedelta(minutes=delay)
                        purchase_time = datetime.datetime.combine(
                            self.current_time.date(), purchase_time.time())
                        self.schedule_pos_entry(
                                purchase_time, shelf.product, 1)
                        amount -= 1

                if amount > 0:
                    max_shopping_time = 35
                    ttc = self.time_til_closing()

                    if ttc < max_shopping_time:
                        max_shopping_time = ttc

                    delay = random.randint(0, max_shopping_time)
                    purchase_time = self.current_time + \
                        datetime.timedelta(minutes=delay)
                    purchase_time = datetime.datetime.combine(
                        self.current_time.date(), purchase_time.time())
                    # print(str(self.current_time) + ' + '
                    #       + str(delay) + ' ' + str(purchase_time))
                    self.schedule_pos_entry(
                                            purchase_time,
                                            shelf.product,
                                            amount)

    def check_if_oos(self, shelf):

        time_between_pos = []
        product_pos = []

        # create list only with pos data of given product
        for _ in self.pos_data:
            if shelf.product.ean == _.ean:
                product_pos.append(_)

        # create list of time between product pos entries
        for i in range(len(product_pos)-1):
            time_difference = product_pos[i
                                          + 1].purchase_time - product_pos[i].purchase_time
            time_difference = (datetime.datetime.min + time_difference).time()

            if time_difference < datetime.time(10, 0, 0):
                time_between_pos.append(
                    time_difference.hour*60 + time_difference.minute)

        time_since_last_purchase = self.current_time - \
            product_pos[-1].purchase_time
        time_since_last_purchase = (
            datetime.datetime.min + time_since_last_purchase).time()
        time_since_last_purchase = time_since_last_purchase.hour*60 + \
            time_since_last_purchase.minute

        if len(time_between_pos) > 2:
            time_between_pos.sort()
            c = collections.Counter(time_between_pos)
            x, y = [], []
            for k in c:
                x.append(k)
                y.append(c.get(k)/len(time_between_pos))
            x = np.array(x)
            y = np.array(y)
            f = CubicSpline(x, y, bc_type='natural')
            print(str(f(time_since_last_purchase))
                  + ' ' + str(time_since_last_purchase))

        if len(time_between_pos) > 30:

            scipy_kernel = gaussian_kde(time_between_pos)

            threshold = 0.01
            if scipy_kernel(time_since_last_purchase) < threshold:
                if shelf.current_stock == 0:
                    print(time_since_last_purchase)
                    print('is OO')
                elif shelf.current_stock > 0:
                    print(time_since_last_purchase)
                    print('not OO')

    def new_day_action(self):
        self.store.restock_shelfs()

    def current_occupancy(self, time):
        return self.store_occupancy[time.weekday()](self.time_to_float(time))

    def schedule_pos_entry(self, time, product, amt):

        if time not in self.scheduled_pos_data:
            self.scheduled_pos_data[time] = [[product, amt]]
        else:
            self.scheduled_pos_data[time].append([product, amt])

    def create_occupancy(self):

        for day in self.store.occupancy:

            x, y = [], []

            for _ in self.store.occupancy.get(day):
                x.append(self.time_to_float(_[0]))
                y.append(_[1])

            x, y = np.array(x), np.array(y)

            self.store_occupancy.append(CubicSpline(x, y))

    def time_til_closing(self):
        closing_time = datetime.datetime.combine(
            self.current_time.date(), self.store.closing_time)
        ttc = closing_time - self.current_time
        ttc = (datetime.datetime.min + ttc).time()
        ttc = ttc.hour*60 + ttc.minute + ttc.second / 60
        return round(ttc)

    def float_to_time(self, f):
        str = '{0:02.0f}:{1:02.0f}'.format(*divmod(f * 60, 60))
        return datetime.strptime(str, '%H:%M')

    def time_to_float(self, t):
        return t.hour + (t.minute + t.second / 60) / 60


class PointOfSaleEntry():

    def __init__(self, product, amount, time):

        self.ean = product.ean
        self.amount = amount
        self.purchase_time = time


if __name__ == '__main__':
    exemplary_retailer = Store(
            1,
            Occupancy().weekly_occupancy,
            datetime.time(7),
            datetime.time(22)
    )

    s = TimeSimulation(exemplary_retailer)

    s.simulate_period(datetime.datetime(2022, 1, 3),
                      datetime.datetime(2024, 1, 9))

    # for _ in s.pos_data:
    #     print(str(_.purchase_time) + ' ' + str(_.amount))

    x2, y2 = [], []
    for _ in s.shelf_data:
        x2.append(_[0])
        y2.append(_[1])

    # plt.plot(x2, y2)
    # plt.show()

    time_between_pos = []
    for i in range(len(s.pos_data)-1):

        if s.pos_data[i+1].purchase_time.weekday() == s.pos_data[i].purchase_time.weekday():

            time_difference = s.pos_data[i
                                         + 1].purchase_time - s.pos_data[i].purchase_time
            time_difference = (datetime.datetime.min + time_difference).time()

            time_between_pos.append(
                time_difference.hour*60 + time_difference.minute + time_difference.second / 60)

    c = collections.Counter(time_between_pos)
    print(c)

    # ck = c.keys()
    # cv = c.values()
    # fig = plt.figure(figsize=(10, 5))
    # plt.bar(ck, cv, color='maroon', width=1)
    plt.hist(time_between_pos, bins=round(len(set(time_between_pos))))
    # plt.hist(time_between_pos, bins=100)

    plt.show()

    r = expon.rvs(loc=1, scale=112, size=10000)
    plt.hist(r, bins=50)
    plt.show()

    # fig, ax = plt.subplots(1, 1)
    # print(expon.mean(expon.fit(time_between_pos)))
    # r = expon.rvs(loc=0, scale=110, size=1000)
    # ax.hist(r, density=True, histtype='stepfilled')
    # ax.legend(loc='best', frameon=False)
    # plt.show()

    sx, sy = [], []
    for _ in s.shelf_data_times:
        sx.append(_[0])
        sy.append(_[1])

    tx = []
    for t in s.pos_data:
        tx.append(t.purchase_time)
    plt.scatter(sx, sx, color='red')
    plt.scatter(tx, tx, color='green')

    # plt.show()

    # x = np.linspace(0, 40, len(time_between_pos))
    #
    # scipy_kernel = gaussian_kde(time_between_pos)
    #
    # bw = scipy_kernel.factor * np.std(time_between_pos)
    # print(bw)
    #
    # u = np.linspace(0, max(time_between_pos), len(time_between_pos))
    # v = scipy_kernel.evaluate(u)
    #
    # count, bins, patches = plt.hist(time_between_pos, bins=len(
    #     time_between_pos), density=True, edgecolor='Black')
    # plt.plot(u, v, 'k')
    # plt.show()
