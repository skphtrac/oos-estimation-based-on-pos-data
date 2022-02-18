import datetime
import random
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

from store_framework import Store
from occupancy import Occupancy


class TimeSimulation():

    def __init__(self, store):

        self.store = store
        self.store_occupancy = []
        self.startdate = None
        self.enddate = None
        self.current_time = None

    def simulate_period(self, start, end):

        self.create_occupancy()

        self.open_days = list(self.store.occupancy.keys())
        # days on which store is supposed to be open
        self.restock_days = [0, 1, 2, 3, 4, 5]
        # days to restock shelves on

        # Time Simulation
        self.startdate = self.current_time = self.last_time_step = start
        self.enddate = end
        self.scheduled_pos_data = {}
        self.pos_data = []
        self.coc = 0
        time_step = datetime.timedelta(hours=1)

        # # OOS
        self.ean_pos = {}
        self.threshold = .8

        # OOS Evalution
        self.oos_check = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.tp, self.fp, self.fn, self.tn = [], [], [], []
        self.tpdf = 0
        self.fpdf = 0
        self.prob_time = []

        while self.current_time <= self.enddate:

            if self.current_time.weekday() in self.open_days:

                if (self.store.opening_time
                    <= self.current_time.time()
                        <= self.store.closing_time):

                    time_step = datetime.timedelta(minutes=1)

                    self.time_action()

                elif self.current_time.weekday() != self.last_time_step.weekday() and self.current_time.weekday() in self.restock_days:
                    self.new_day_action()
                else:
                    time_step = datetime.timedelta(hours=1)

            self.last_time_step = self.current_time
            self.current_time += time_step

    def time_action(self):
        # called each timestep within stores opening times

        self.coc = self.current_occupancy(self.current_time)

        for shelf in self.store.shelves:
            if shelf.current_stock > 0:

                self.simulate_shelf_demand(shelf)

                if shelf.current_stock == 0:
                    print(str(shelf.product.name)
                          + ' is OOS at ' + str(self.current_time))

            # calculates the time between pos entries
            self.calculate_purchase_delta(shelf.product.ean)

            # checks whether product could be OOS
            self.check_if_oos(shelf)

        self.process_sheduled_pos(self.current_time)
        # self.check_pos_for_entry(self.current_time)

    def simulate_shelf_demand(self, shelf):
        # using store occupancy and product

        if random.random() > self.coc * shelf.product.demand_factor:
            # uses current store occupancy and product demand_factor to decide
            # whether to take product out of shelf

            return

        amount = random.randint(1, 2)  # amount of products taken out of shelf

        if amount > shelf.current_stock:
            amount = shelf.current_stock

        shelf.current_stock -= amount

        if random.random() < 0.01:  # shrinkage -> will not trigger pos entry
            return

        for i in range(amount):
            # takes into account whether product is left during
            # shopping and is retrieved at some point later

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
            # generates time stamps for when product will be bought

            max_shopping_time = 35
            ttc = self.time_til_closing()

            if ttc < max_shopping_time:
                max_shopping_time = ttc

            delay = random.randint(0, max_shopping_time)

            purchase_time = self.current_time + \
                datetime.timedelta(minutes=delay)
            purchase_time = datetime.datetime.combine(
                self.current_time.date(), purchase_time.time())

            self.schedule_pos_entry(
                                    purchase_time,
                                    shelf.product,
                                    amount)

    def process_sheduled_pos(self, time):
        # checks whether purchases are scheduled for current time to
        # create point-of-sale entries

        if time not in self.scheduled_pos_data:
            return

        max = random.randint(1, 2)
        i = 0
        for value in self.scheduled_pos_data[time]:

            if i < max:
                self.pos_data.append(
                    PointOfSaleEntry(
                        value[0],
                        value[1],
                        time
                    )
                )
            else:  # delays purchases by one minute
                purchase_time = time + \
                    datetime.timedelta(minutes=1)
                self.schedule_pos_entry(
                        purchase_time,
                        value[0],
                        value[1])
            i += 1

    def calculate_purchase_delta(self, ean):

        if ean not in self.ean_pos:

            self.ean_pos[ean] = {
                'pos': [],          # ean's pos entries
                'len': 0,           # length of pos list to check for new entries
                'deltas': [],       # frequency of ean's purchase time deltas
                'lpi': 0,           # last used index of latest pos_data iteration
                'lepi': 0,          # last used index of latest ean_pos iteration
                'curr_delta': -1,   # current time since last purchase
                'pdf': 0,           # probability density function of observed deltas
                'mult': 0           # multiplicator for noramilzaion of pdf values
            }

        start_i = self.ean_pos[ean]['lpi']

        # create list only with pos data of given product
        for i, entry in enumerate(self.pos_data[start_i:], start_i):
            if ean == entry.ean:
                self.ean_pos[ean]['pos'].append(entry)
                self.ean_pos[ean]['lpi'] = i+1

        last_i = self.ean_pos[ean]['lepi']

        for i, entry in enumerate(self.ean_pos[ean]['pos'][last_i:], last_i):

            if (self.current_time - datetime.timedelta(minutes=1)) == entry.purchase_time:
                self.ean_pos[ean]['deltas'].append(
                    self.ean_pos[ean]['curr_delta'])
                self.ean_pos[ean]['curr_delta'] = 0
            self.ean_pos[ean]['lepi'] = i
        self.ean_pos[ean]['curr_delta'] += 1

        if self.current_time.time() == self.store.closing_time:
            self.ean_pos[ean]['deltas'].append(
                self.ean_pos[ean]['curr_delta'])
            self.ean_pos[ean]['curr_delta'] = 0

    def check_if_oos(self, shelf):
        # calculates time since last PoS entry for a product, then creates a
        # kernel density function based on all previous purchase time deltas
        # to calculate how probable the calculates time since the last PoS entry
        # is compared to previous entries

        ean = shelf.product.ean

        if ean not in self.ean_pos or len(self.ean_pos[ean]['pos']) < 20:
            return

        curr_day = self.current_time.weekday()

        # creates kernel density function based on all previous purchase deltas
        if self.ean_pos[ean]['len'] != len(self.ean_pos[ean]['pos']):
            self.ean_pos[ean]['pdf'] = gaussian_kde(
                self.ean_pos[ean]['deltas'])
            self.ean_pos[ean]['len'] = len(self.ean_pos[ean]['pos'])
            self.ean_pos[ean]['mult'] = self.ean_pos[ean]['pdf'].integrate_box_1d(
                0, 1000)

            # if len(self.tp) > 10 and len(self.fp) > 10:
            #     self.threshold = self.calculate_threshold()
            # else:

        # calculate time since last purchase for given product
        if curr_day == self.ean_pos[ean]['pos'][-1].purchase_time.weekday():
            time_since_last_purchase = self.current_time - \
                self.ean_pos[ean]['pos'][-1].purchase_time
        else:
            opening_time = datetime.datetime.combine(
                self.current_time.date(), self.store.opening_time)
            time_since_last_purchase = self.current_time - opening_time

        time_since_last_purchase = (
            datetime.datetime.min + time_since_last_purchase).time()
        time_since_last_purchase = int(time_since_last_purchase.hour*60
                                       + time_since_last_purchase.minute)

        # calculates probability of time since last PoS entry in relation to
        # all previous purchase deltas
        oos_prob = self.ean_pos[ean]['pdf'].integrate_box_1d(
            0, time_since_last_purchase)
        oos_prob = oos_prob * (1/self.ean_pos[ean]['mult'])

        # evaluates relation of the OOS probability and current shelf stock
        if shelf.current_stock == 0:
            self.oos_check['TP'] += 1
            self.tp.append(oos_prob)

        elif shelf.current_stock > 0:
            self.oos_check['FP'] += 1
            self.fp.append(oos_prob)

        # makes estimation if Product is OOS
        if oos_prob > self.threshold:
            # self.oos_result.append(True)
            if shelf.current_stock == 0:
                self.oos_check['TP'] += 1
                return True
                # self.tp.append(oos_prob)

            elif shelf.current_stock > 0:
                self.oos_check['FP'] += 1
                # self.fp.append(oos_prob)
                return False

        elif oos_prob < self.threshold:
            # self.oos_result.append(False)
            if shelf.current_stock == 0:
                self.oos_check['FN'] += 1
                # self.fn.append(oos_prob)
                return False

            elif shelf.current_stock > 0:
                self.oos_check['TN'] += 1
                # self.tn.append(oos_prob)
                return True

    def calculate_threshold(self):
        self.fpdf = gaussian_kde(self.fp)
        self.tpdf = gaussian_kde(self.tp)

        threshold = 0
        intsc = 100
        for i in range(100):
            p = i/100
            difference = abs(self.tpdf(p)-self.fpdf(p))
            if difference < intsc:
                intsc = difference
                threshold = p
        return threshold

    def new_day_action(self):
        # called in the beginning of a new day on specified days

        self.store.restock_shelves()

    def current_occupancy(self, time):
        coc = self.store_occupancy[time.weekday()](self.time_to_float(time))
        return coc

    def schedule_pos_entry(self, time, product, amt):

        if time not in self.scheduled_pos_data:
            self.scheduled_pos_data[time] = [[product, amt]]
        else:
            self.scheduled_pos_data[time].append([product, amt])

    def create_occupancy(self):
        # interpolates between the given occupancy data for each hour
        # on a day to create a interpolation function which returns the
        # occupancy in the store at the given time

        for day in self.store.occupancy:

            x, y = [], []

            for _ in self.store.occupancy.get(day):
                x.append(self.time_to_float(_[0]))
                y.append(_[1])

            x, y = np.array(x), np.array(y)

            self.store_occupancy.append(CubicSpline(x, y))

    def time_til_closing(self):
        # calculates the time between the current time step and
        # the closing time of a store

        closing_time = datetime.datetime.combine(
            self.current_time.date(), self.store.closing_time)
        ttc = closing_time - self.current_time
        ttc = (datetime.datetime.min + ttc).time()
        ttc = ttc.hour*60 + ttc.minute + ttc.second / 60
        return round(ttc)

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
                      datetime.datetime(2022, 1, 9))
