import datetime
from scipy.interpolate import CubicSpline
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
import random
from scipy.stats import norm, gaussian_kde, expon, poisson
from scipy.special import ndtr
import statistics
import math
from sklearn.neighbors import KernelDensity
import collections
from scipy.interpolate import CubicSpline
import seaborn as sns
import scipy.optimize
import math

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

        # OOS
        self.ean_pos = {}
        self.last_ean_index = 0
        self.oos_result = []
        self.oos_check = {'TP': 0, 'aTP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        self.tp, self.atp, self.fp, self.fn, self.tn = [], [], [], [], []

        self.pos_time_deltas = []
        self.purchase_delta = 0
        self.last_searched_index = 0
        self.oos_corr = []
        self.thresh_count = [0, 0]
        self.pos_length = 0
        self.pdf = 0
        self.mult = 0

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

            if len(self.pos_data) > 2:
                self.calculate_purchase_delta(shelf.product.ean)
                self.check_if_oos2(shelf)

        self.process_sheduled_pos(self.current_time)
        self.check_pos_for_entry(self.current_time)

    def simulate_shelf_demand(self, shelf):

        self.product_graph.append([
            self.current_time, shelf.current_stock])

        if random.random() > self.coc * shelf.product.popularity:
            return

        amount = random.randint(1, 2)
        if amount > shelf.current_stock:
            amount = shelf.current_stock

        shelf.current_stock -= amount

        self.shelf_data_times.append(
            [self.current_time, shelf.current_stock])

        if random.random() < 0.02:  # shrinkage
            return

        for i in range(amount):
            if random.random() < 0.01:  # product left and retrieved later

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

    def process_sheduled_pos(self, time):

        if time in self.scheduled_pos_data:
            max = random.randint(1, 2)
            # for value in self.scheduled_pos_data.get(self.current_time):
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
                else:
                    purchase_time = time + \
                        datetime.timedelta(minutes=1)
                    self.schedule_pos_entry(
                            purchase_time,
                            value[0],
                            value[1])
                i += 1

    def check_pos_for_entry(self, time):

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
                'deltas': {},     # frequency of ean's purchase time deltas
                'lpi': 0,       # last used index of latest pos_data iteration
                'lepi': 0       # last used index of latest ean_pos iteration
            }

        start_i = self.ean_pos[ean]['lpi']

        # create list only with pos data of given product
        for i, entry in enumerate(self.pos_data[start_i:], start_i):
            if ean == entry.ean:
                self.ean_pos[ean]['pos'].append(entry)
                self.ean_pos[ean]['lpi'] = i+1

        start_i = self.ean_pos[ean]['lepi']

        # create dict of minute frequencey of ean's purchase time delta
        for i, pos in enumerate(self.ean_pos[ean]['pos'][start_i:-1], start_i):
            current, next = pos, self.ean_pos[ean]['pos'][i+1]

            pos_day = current.purchase_time.weekday()
            if next.purchase_time.weekday() == pos_day:

                purchase_delta = next.purchase_time - current.purchase_time
                purchase_delta = (datetime.datetime.min
                                  + purchase_delta).time()
                purchase_delta = int(purchase_delta.hour*60
                                     + purchase_delta.minute
                                     + purchase_delta.second / 60)

                if purchase_delta > 0:
                    for t in range(0, purchase_delta):
                        if t not in self.ean_pos[ean]['deltas']:
                            self.ean_pos[ean]['deltas'][t] = 1
                        else:
                            self.ean_pos[ean]['deltas'][t] += 1
                elif purchase_delta == 0:
                    t = 0
                    if t not in self.ean_pos[ean]['deltas']:
                        self.ean_pos[ean]['deltas'][t] = 1
                    else:
                        self.ean_pos[ean]['deltas'][t] += 1

            self.ean_pos[ean]['lepi'] = i+1

    def check_if_oos(self, shelf):
        ean = shelf.product.ean

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

        # fp = self.fit_to_expon(list(self.ean_pos[ean]['deltas'].values(
        # )), time_since_last_purchase)/self.ean_pos[ean]['deltas'][0]
        #
        # fp = 1-fp

        # if time_since_last_purchase in self.ean_pos[ean]['deltas']:
        #     fp = 1 - \
        #         (self.ean_pos[ean]['deltas'][time_since_last_purchase]
        #          / self.ean_pos[ean]['deltas'][0])
        # time_since_last_purchase += 10
        if time_since_last_purchase in self.ean_pos[ean]['deltas']:
            if self.ean_pos[ean]['deltas'][time_since_last_purchase] < 2:
                fp = 1
            else:
                fp = 1 - \
                    (self.ean_pos[ean]['deltas'][time_since_last_purchase]
                     / self.ean_pos[ean]['deltas'][0])
            # fp = 0
            # if self.ean_pos[ean]['deltas'][time_since_last_purchase] <= 1:
            #     fp = 1
            # else:
            #     fp = 0
        else:
            fp = 1

        upper_threshold = .98
        lower_threshold = .55

        # fdfm = self.less_pos_than_mean(self.ean_pos[ean]['pos'])
        # curr, mean = self.less_pos_than_mean(self.ean_pos[ean]['pos'])

        if fp > upper_threshold:
            self.oos_result.append(True)
        elif fp < lower_threshold:
            self.oos_result.append(False)

        if len(self.oos_result) > 4:
            for i, result in enumerate(self.oos_result[:-5], 4):
                if (self.oos_result[i] == self.oos_result[i-1] == self.oos_result[i-2] == self.oos_result[i-3] == self.oos_result[i-4]) is True:
                    print(shelf.current_stock)

        if fp >= upper_threshold:
            self.oos_result.append(True)
            if shelf.current_stock == 0 or shelf.current_stock/shelf.capacity < .2:
                self.oos_check['TP'] += 1  # 0.784
                self.tp.append(fp)

            elif shelf.current_stock > 0:
                self.oos_check['FP'] += 1  # 0.55
                self.fp.append(fp)

        elif fp < lower_threshold:
            self.oos_result.append(False)
            if shelf.current_stock == 0:
                self.oos_check['FN'] += 1  # 0.752
                self.fn.append(fp)

            elif shelf.current_stock > 0:
                self.oos_check['TN'] += 1  # 0.537
                self.tn.append(fp)

    def check_if_oos2(self, shelf):
        ean = shelf.product.ean

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

        if self.pos_length != len(s.pos_time_deltas):
            self.pdf = gaussian_kde(s.pos_time_deltas)
            self.pos_length = len(s.pos_time_deltas)
            self.mult = self.pdf.integrate_box_1d(0, 1000)

        oos_prob = self.pdf.integrate_box_1d(0, time_since_last_purchase)
        oos_prob = oos_prob * (1/self.mult)

        upper_threshold = .8

        if oos_prob >= upper_threshold:
            self.thresh_count[0] += 1
        else:
            self.thresh_count[1] += 1

        # if oos_prob > upper_threshold:
        #     self.oos_result.append(True)
        if shelf.current_stock == 0:
            self.oos_check['TP'] += 1  # 0.784
            self.tp.append(oos_prob)

        elif shelf.current_stock > 0:
            self.oos_check['FP'] += 1  # 0.55
            self.fp.append(oos_prob)

        # elif oos_prob < upper_threshold:
        #     self.oos_result.append(False)
        #     if shelf.current_stock == 0:
        #         self.oos_check['FN'] += 1  # 0.752
        #         self.fn.append(oos_prob)
        #
        #     elif shelf.current_stock > 0:
        #         self.oos_check['TN'] += 1  # 0.537
        #         self.tn.append(oos_prob)

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

    def monoExp(self, x, m, t, b):
        return m * np.exp(-t * x) + b

    def fit_to_expon(self, data, point):

        xs = np.arange(len(data))
        ys = np.array(data)

        # perform the fit
        p0 = (2000, .1, 50)  # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(self.monoExp, xs, ys, p0)
        m, t, b = params
        sampleRate = 20_000  # Hz
        tauSec = (1 / t) / sampleRate

        # determine quality of the fit
        squaredDiffs = np.square(ys - self.monoExp(xs, m, t, b))
        squaredDiffsFromMean = np.square(ys - np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        return self.monoExp(point, m, t, b)

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
                      datetime.datetime(2022, 2, 9))

    def show_true_false_results():

        print(s.oos_check)

        sum = 0
        for k in s.oos_check:
            sum += s.oos_check[k]

        for k in s.oos_check:
            print(str(k) + ': ' + str(s.oos_check[k]/sum))

        print(
            'T: ' + str((s.oos_check['TP'] + s.oos_check['TN'] + s.oos_check['aTP'])/sum))
        print('F: ' + str(s.oos_check['FP']/sum + s.oos_check['FN']/sum))
        # a = np.array(s.tp)
        # print(statistics.mode(a))
        print('TP: ' + str(np.mean(s.tp)))
        # print('aTP: ' + str(np.mean(s.atp)))
        print('FP: ' + str(np.mean(s.fp)))
        # print('FN: ' + str(np.mean(s.fn)))
        # print('TN: ' + str(np.mean(s.tn)))

        plt.hist(s.tp, color='limegreen', bins=100, alpha=.5)
        # plt.show()
        plt.hist(s.fp, color='firebrick', bins=100, alpha=.5)
        plt.show()
        # plt.hist(s.fn, color='lightcoral', bins=100, alpha=.5)
        # # plt.show()
        # plt.hist(s.tn, color='seagreen', bins=100, alpha=.5)
        # plt.show()

    def kde_to_data_comparison(data):
        x = np.array(data).reshape(-1, 1)
        kde = KernelDensity(bandwidth=.01).fit(x)
        kk = kde.sample(round(len(data)))
        plt.hist(kk, color='firebrick', bins=100, alpha=.5)
        plt.hist(data, color='limegreen', bins=100, alpha=.5)

    def kde_purchase_delta_prob_to_current_stock():
        fppdf = gaussian_kde(s.fp)
        tppdf = gaussian_kde(s.tp)
        k = np.linspace(0, 1, 100)
        plt.plot(k, fppdf(k), color='firebrick')
        plt.fill_between(k, fppdf(k), color='firebrick', alpha=.5)
        plt.plot(k, tppdf(k), color='limegreen')
        plt.fill_between(k, tppdf(k), color='limegreen', alpha=.5)
        plt.show()

    # show_true_false_results()
    kde_purchase_delta_prob_to_current_stock()
    # kde_to_data_comparison(s.tp)
    # kde_to_data_comparison(s.fp)
    # kde_to_data_comparison(s.pos_time_deltas)

    # plt.hist(s.tp, color='limegreen', bins=100, alpha=.5)
    # plt.hist(s.fp, color='firebrick', bins=100, alpha=.5)

    plt.show()

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # x, y = [], []
    # for k in s.ean_pos[1111111111111]['deltas'].keys():
    #     x.append(k)
    #     y.append(s.ean_pos[1111111111111]['deltas'][k])
    #
    # # p = y[0]
    # # for i, v in enumerate(y):
    # #     y[i] = y[i]/p
    # # print(y)
    # #
    # xs = np.arange(len(y))
    # ys = np.array(y)
    #
    # def monoExp(x, m, t, b):
    #     return m * np.exp(-t * x) + b
    #
    # # perform the fit
    # p0 = (2000, .1, 50)  # start with values near those we expect
    # params, cv = scipy.optimize.curve_fit(monoExp, xs, ys, p0)
    # m, t, b = params
    # sampleRate = 20_000  # Hz
    # tauSec = (1 / t) / sampleRate
    #
    # # determine quality of the fit
    # squaredDiffs = np.square(ys - monoExp(xs, m, t, b))
    # squaredDiffsFromMean = np.square(ys - np.mean(ys))
    # rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    # print(f"R² = {rSquared}")
    #
    # # plot the results
    # plt.plot(xs, ys, '.', label="data")
    # plt.plot(xs, monoExp(xs, m, t, b), '--', label="fitted")
    # plt.title("Fitted Exponential Curve")
    #
    # # inspect the parameters
    # print(f"Y = {m} * e^(-{t} * x) + {b}")
    # print(f"Tau = {tauSec * 1e6} µs")
    #
    # def expo(x):
    #     return m * math.exp((-t) * x) + b
    #
    # print(expo(100))
    # print(s.ean_pos[1111111111111]['deltas'][100])

    # xs2 = np.arange(25)
    # ys2 = monoExp(xs2, m, t, b)
    #
    # plt.plot(xs, ys, '.', label="data")
    # plt.plot(xs2, ys2, '--', label="fitted")
    # plt.title("Extrapolated Exponential Curve")

    # plt.show()
