import datetime
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

from store_framework import Store
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

        # Time Simulation
        self.startdate = self.current_time = self.last_time_step = start
        self.enddate = end
        self.scheduled_pos_data = {}
        self.pos_data = []
        self.shelf_data = []
        self.shelf_data_times = []
        self.coc = 0
        time_step = datetime.timedelta(hours=1)

        # OOS
        self.ean_pos = {}
        self.last_ean_index = 0
        self.pos_time_deltas = []
        self.purchase_delta = 0
        self.last_searched_index = 0
        self.pos_length = 0
        self.pdf = 0
        self.mult = 0

        # OOS evalution
        self.oos_check = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.tp, self.fp, self.fn, self.tn = [], [], [], []

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
        # called each timestep within stores opening times

        self.coc = self.current_occupancy(self.current_time)

        for shelf in self.store.shelfs:
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
        self.check_pos_for_entry(self.current_time)

    def simulate_shelf_demand(self, shelf):
        # using store occupancy and product

        self.product_graph.append([
            self.current_time, shelf.current_stock])

        if random.random() > self.coc * shelf.product.popularity:
            # uses current store occupancy and product popularity to decide
            # whether product is taken out of shelf

            return

        amount = random.randint(1, 2)  # amount of products taken out of shelf
        if amount > shelf.current_stock:
            amount = shelf.current_stock

        shelf.current_stock -= amount

        self.shelf_data_times.append(
            [self.current_time, shelf.current_stock])

        if random.random() < 0.02:  # shrinkage -> will not trigger pos entry
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

        if time in self.scheduled_pos_data:
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

    def check_pos_for_entry(self, time):
        # to calculate the time between pos entries it is checked whether
        # there are pos entries at the current time step and the time since
        # the last entry is added to the list of all purchase deltas

        for i, entry in enumerate(self.pos_data[self.last_searched_index:], self.last_searched_index):
            if time == entry.purchase_time:
                self.pos_time_deltas.append(self.purchase_delta)
                self.purchase_delta = 0
            self.last_searched_index = i

        self.purchase_delta += 1

        if time.time() == self.store.closing_time:
            self.pos_time_deltas.append(self.purchase_delta)
            self.purchase_delta = 0

    def calculate_purchase_delta(self, ean):

        if ean not in self.ean_pos:

            self.ean_pos[ean] = {
                'pos': [],      # ean's pos entries
                'deltas': [],   # frequency of ean's purchase time deltas
                'lpi': 0,       # last used index of latest pos_data iteration
                'lepi': 0,       # last used index of latest ean_pos iteration
                'curr_delta': -1
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

        # start_i = self.ean_pos[ean]['lpi']
        #
        # create dict of minute frequencey of ean's purchase time delta
        # for i, pos in enumerate(self.ean_pos[ean]['pos'][start_i:-1], start_i):
        #     current, next = pos, self.ean_pos[ean]['pos'][i+1]
        #
        #     pos_day = current.purchase_time.weekday()
        #     if next.purchase_time.weekday() == pos_day:
        #
        #         purchase_delta = next.purchase_time - current.purchase_time
        #         purchase_delta = (datetime.datetime.min
        #                           + purchase_delta).time()
        #         purchase_delta = int(purchase_delta.hour*60
        #                              + purchase_delta.minute
        #                              + purchase_delta.second / 60)
        #
        #         if purchase_delta > 0:
        #             for t in range(0, purchase_delta):
        #                 if t not in self.ean_pos[ean]['deltas']:
        #                     self.ean_pos[ean]['deltas'][t] = 1
        #                 else:
        #                     self.ean_pos[ean]['deltas'][t] += 1
        #         elif purchase_delta == 0:
        #             t = 0
        #             if t not in self.ean_pos[ean]['deltas']:
        #                 self.ean_pos[ean]['deltas'][t] = 1
        #             else:
        #                 self.ean_pos[ean]['deltas'][t] += 1
        #
        #     self.ean_pos[ean]['lepi'] = i+1

    def check_if_oos(self, shelf):
        # calculates time since last PoS entry for a product, then creates a
        # kernel density function based on all previous purchase time deltas
        # to calculate how probable the calculates time since the last PoS entry
        # is compared to previous entries

        ean = shelf.product.ean

        if ean not in self.ean_pos or len(self.ean_pos[ean]['pos']) < 20:
            return

        curr_day = self.current_time.weekday()

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

        # creates kernel density function based on all previous purchase deltas
        if self.pos_length != len(s.pos_time_deltas):
            self.pdf = gaussian_kde(s.pos_time_deltas)
            self.pos_length = len(s.pos_time_deltas)
            self.mult = self.pdf.integrate_box_1d(0, 1000)

        # calculates probability of time since last PoS entry in relation to
        # all previous purchase deltas
        oos_prob = self.pdf.integrate_box_1d(0, time_since_last_purchase)
        oos_prob = oos_prob * (1/self.mult)

        # evaluates relation of the OOS probability and current shelf stock
        if shelf.current_stock == 0:
            self.oos_check['TP'] += 1  # 0.784
            self.tp.append(oos_prob)

        elif shelf.current_stock > 0:
            self.oos_check['FP'] += 1  # 0.55
            self.fp.append(oos_prob)

        # threshold = .51  # threshold for triggering OOS detection
        #
        # if oos_prob > threshold:
        #     # self.oos_result.append(True)
        #     if shelf.current_stock == 0:
        #         self.oos_check['TP'] += 1  # 0.784
        #         self.tp.append(oos_prob)
        #
        #     elif shelf.current_stock > 0:
        #         self.oos_check['FP'] += 1  # 0.55
        #         self.fp.append(oos_prob)
        #
        # elif oos_prob < threshold:
        #     # self.oos_result.append(False)
        #     if shelf.current_stock == 0:
        #         self.oos_check['FN'] += 1  # 0.752
        #         self.fn.append(oos_prob)
        #
        #     elif shelf.current_stock > 0:
        #         self.oos_check['TN'] += 1  # 0.537
        #         self.tn.append(oos_prob)

    def calculate_threshold(self):
        threshold = 0
        day_sales = []
        counter = 1

        for i, pos in enumerate(s.ean_pos[1111111111111]['pos'][:-1]):

            current, next = pos, s.ean_pos[1111111111111]['pos'][i+1]

            if current.purchase_time.weekday() == next.purchase_time.weekday():
                counter += 1
            else:
                day_sales.append(counter)
                counter = 1

        mean = np.mean(day_sales)

        return threshold

    def less_pos_than_mean(self, pos_data):

        sum_sales = 0
        diff_days = 1
        sum_sales_curr_day = 0

        curr_day = self.current_time.weekday()
        curr_time = self.current_time.time()

        last_date = pos_data[0].purchase_time.date()

        for pos in pos_data:
            if pos.purchase_time.time() < curr_time:
                if pos.purchase_time.weekday() == curr_day:
                    sum_sales += 1
                    if not pos.purchase_time.date() == last_date:
                        diff_days += 1
                    last_date = pos.purchase_time.date()
                if pos.purchase_time.date() == self.current_time.date():
                    sum_sales_curr_day += 1

        # return sum_sales_curr_day, sum_sales/diff_days
        if sum_sales_curr_day < (sum_sales/diff_days):
            return True
        else:
            return False

    def mean_sales(self, ean):
        day_sales = []
        counter = 1
        for i, pos in enumerate(self.ean_pos[ean]['pos'][:-1]):
            current, next = pos, self.ean_pos[ean]['pos'][i+1]

            if current.purchase_time.weekday() == next.purchase_time.weekday():
                counter += 1
            else:
                day_sales.append(counter)
                counter = 1
        return np.mean(day_sales)

    def new_day_action(self):
        # called in the beginning of a new day on specified days

        self.store.restock_shelfs()

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
                      datetime.datetime(2022, 2, 9))

    def show_true_false_results():

        print(s.oos_check)

        sum = 0
        for k in s.oos_check:
            sum += s.oos_check[k]

        for k in s.oos_check:
            print(str(k) + ': ' + str(s.oos_check[k]/sum))

        print('T: ' + str((s.oos_check['TP'] + s.oos_check['TN'])/sum))
        print('F: ' + str((s.oos_check['FP'] + s.oos_check['FN'])/sum))
        # print('TP: ' + str(np.mean(s.tp)))
        # print('FP: ' + str(np.mean(s.fp)))
        # print('FN: ' + str(np.mean(s.fn)))
        # print('TN: ' + str(np.mean(s.tn)))

        plt.hist(s.tp, color='limegreen', bins=100, alpha=.5)
        plt.hist(s.fp, color='firebrick', bins=100, alpha=.5)
        plt.hist(s.fn, color='lightcoral', bins=100, alpha=.5)
        plt.hist(s.tn, color='seagreen', bins=100, alpha=.5)
        plt.show()

    def kde_to_data_comparison(data):
        x = np.array(data).reshape(-1, 1)
        kde = KernelDensity(bandwidth=.01).fit(x)
        kk = kde.sample(round(len(data)))
        # plt.hist(kk, color='firebrick', bins=100, alpha=.5)
        plt.hist(data, color='limegreen', bins=100, alpha=.5)

    def kde_purchase_delta_prob_to_current_stock():
        fppdf = gaussian_kde(s.fp)
        tppdf = gaussian_kde(s.tp)
        k = np.linspace(0, 1, 100)
        plt.plot(k, fppdf(k), color='firebrick')
        plt.fill_between(k, fppdf(k), color='firebrick', alpha=.5)
        plt.plot(k, tppdf(k), color='limegreen')
        plt.fill_between(k, tppdf(k), color='limegreen', alpha=.5)

    # show_true_false_results()
    # kde_purchase_delta_prob_to_current_stock()
    # kde_to_data_comparison(s.tp)
    # kde_to_data_comparison(s.fp)
    # kde_to_data_comparison(s.pos_time_deltas)
    #
    # kde_to_data_comparison(s.ean_pos[s.store.shelfs[0].product.ean]['deltas'])

    plt.show()

    for shelf in s.store.shelfs:
        kde_to_data_comparison(s.ean_pos[shelf.product.ean]['deltas'])
        print(str(shelf.capacity) + ' ' + str(shelf.product.popularity))
        plt.show()

    # print(s.shelf_data)
    #
    x, y = [], []
    for _ in s.shelf_data:
        if (s.store.opening_time <= _[0].time() <= s.store.closing_time):
            x.append(_[1])
            y.append(_[0])

    plt.plot(y, x)
    plt.show()


# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# import numpy as np
#
# x = [13, 13.4, 5, 8, 13, 10.9, 11.3, 11.2, 7.8, 8, 6.4, 6.9, 7.1, 5.1, 5.2, 4, 13.5]
# y = [.185, .22, .9, .6, .19, .4, .34, .3, .54, .52, .74, .68, .65, 0.89, .81, .95, .165]
# x.sort()
# y.sort()
# my_fun = CubicSpline(x, y, bc_type='natural')
#
# print(my_fun(5))
# plt.scatter(y, x)
# plt.show()

# day_sales = []
# counter = 1
# for i, pos in enumerate(s.ean_pos[1111111111111]['pos'][:-1]):
#
#     current, next = pos, s.ean_pos[1111111111111]['pos'][i+1]
#
#     if current.purchase_time.weekday() == next.purchase_time.weekday():
#         counter += 1
#     else:
#         day_sales.append(counter)
#         counter = 1
#
# mean = np.mean(day_sales)
# print(mean)
