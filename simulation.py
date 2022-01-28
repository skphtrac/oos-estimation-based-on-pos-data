import datetime
from scipy.interpolate import CubicSpline
import numpy as np
from numpy import cov
import matplotlib.pyplot as plt
import random
from scipy.stats import norm, gaussian_kde, expon, poisson
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

        self.coc = self.current_occupancy(self.current_time)

        for shelf in self.store.shelfs:
            if shelf.current_stock > 0:

                self.simulate_shelf_demand(shelf)

                if shelf.current_stock == 0:
                    print(str(shelf.product.name)
                          + ' is OOS at ' + str(self.current_time))

            if len(self.pos_data) > 20:
                self.calculate_purchase_delta(shelf.product.ean)
                self.check_if_oos3(shelf)

        self.process_sheduled_pos(self.current_time)

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

    def check_if_oos2(self, shelf):

        product_pos = []

        # create list only with pos data of given product
        for _ in self.pos_data:
            if shelf.product.ean == _.ean:
                product_pos.append(_)

        time_between_pos = []
        for i in range(len(product_pos)-1):

            if product_pos[i+1].purchase_time.weekday() == product_pos[i].purchase_time.weekday():

                time_difference = product_pos[i
                                              + 1].purchase_time - product_pos[i].purchase_time
                time_difference = (datetime.datetime.min
                                   + time_difference).time()

                time_between_pos.append(
                    time_difference.hour*60 + time_difference.minute + time_difference.second / 60)

        #calculate probability of time differences
        c = collections.Counter(time_between_pos)
        for key in c:
            c[key] = [c[key]/(len(time_between_pos))]

        #calculate time since last purchase for given product
        time_since_last_purchase = self.current_time - \
            product_pos[-1].purchase_time
        time_since_last_purchase = (
            datetime.datetime.min + time_since_last_purchase).time()
        time_since_last_purchase = time_since_last_purchase.hour*60 + \
            time_since_last_purchase.minute

        threshold = 0.8

        if shelf.current_stock < 20:

            i = time_since_last_purchase
            cpb = 0  # prob that shelf is OOS
            while i >= 0:
                if i in c:
                    cpb += c[i][0]
                i -= 1

            if cpb > threshold and shelf.current_stock == 0:
                self.oos_check['TP'] += 1
            elif cpb > threshold and shelf.current_stock > 0:
                self.oos_check['FP'] += 1
            elif cpb < threshold and shelf.current_stock == 0:
                self.oos_check['FN'] += 1
            elif cpb < threshold and shelf.current_stock > 0:
                self.oos_check['TN'] += 1
            # print(str(cpb) + ' ' + str(shelf.current_stock))

        # if time_since_last_purchase in c:
        #     if c[time_since_last_purchase][0] < threshold and shelf.current_stock == 0:
        #         self.oos_check[0] += 1
        #     else:
        #         self.oos_check[1] += 1

        # mu = expon.mean(expon.fit(time_between_pos))
        # prob = expon.pdf(time_since_last_purchase, scale=mu)
        # if prob[0] < threshold and shelf.current_stock == 0:
        #     self.oos_check[0] += 1
        # elif prob[0] < threshold and shelf.current_stock != 0:
        #     self.oos_check[1] += 1

    def calculate_purchase_delta(self, ean):

        if ean not in self.ean_pos:

            self.ean_pos[ean] = {
                'pos': [],      # ean's pos entries
                'deltas': {},   # frequency of ean's purchase time deltas
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

            if next.purchase_time.weekday() == current.purchase_time.weekday():

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

    def check_if_oos3(self, shelf):
        ean = shelf.product.ean

        # calculate time since last purchase for given product
        if self.current_time.weekday() == self.ean_pos[ean]['pos'][-1].purchase_time.weekday():
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

        if time_since_last_purchase in self.ean_pos[ean]['deltas']:
            fp = 1 - \
                (self.ean_pos[ean]['deltas'][time_since_last_purchase]
                 / self.ean_pos[ean]['deltas'][0])
        else:
            fp = 1

        threshold = .9

        if fp >= threshold and shelf.current_stock == 0:
            self.oos_check['TP'] += 1  # 0.784
            self.tp.append(fp)
            # threshold += .01
        elif fp >= threshold and shelf.current_stock > 0:
            self.oos_check['FP'] += 1  # 0.55
            self.fp.append(fp)
            # threshold -= .01
        elif fp < threshold and shelf.current_stock == 0:
            self.oos_check['FN'] += 1  # 0.752
            self.fn.append(fp)
            # threshold -= .01
            # print(str(fp) + ' ' + str(shelf.current_stock))
        elif fp < threshold and shelf.current_stock > 0:
            self.oos_check['TN'] += 1  # 0.537
            self.tn.append(fp)
            # threshold += .01
        # if shelf.current_stock == 0:
        #     self.tp.append(fp)
        # elif shelf.current_stock > 0:
        #     self.fp.append(fp)

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
                      datetime.datetime(2022, 6, 9))

    # for _ in s.pos_data:
    #     print(str(_.purchase_time) + ' ' + str(_.amount))

    print(s.ean_pos[1111111111111]['deltas'])
    print(s.oos_check)
    sum = 0
    for k in s.oos_check:
        sum += s.oos_check[k]

    for k in s.oos_check:
        print(str(k) + ': ' + str(s.oos_check[k]/sum))

    print('T: ' + str(s.oos_check['TP']/sum + s.oos_check['TN']/sum))
    print('F: ' + str(s.oos_check['FP']/sum + s.oos_check['FN']/sum))

    # print('TP: ' + str(np.mean(s.tp)))
    # print('FP: ' + str(np.mean(s.fp)))
    # print('FN: ' + str(np.mean(s.fn)))
    # print('TN: ' + str(np.mean(s.tn)))

    # plt.hist(s.tp, color='limegreen', bins=10, alpha=.5)
    # # plt.show()
    # plt.hist(s.fp, color='firebrick', bins=10, alpha=.5)
    # # plt.show()
    # plt.hist(s.fn, color='lightcoral', bins=90, alpha=.5)
    # # plt.show()
    # plt.hist(s.tn, color='seagreen', bins=90, alpha=.5)
    # plt.show()
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
    # print(s.ean_pos[1111111111111]['deltas'][100]/p)

    # xs2 = np.arange(25)
    # ys2 = monoExp(xs2, m, t, b)
    #
    # plt.plot(xs, ys, '.', label="data")
    # plt.plot(xs2, ys2, '--', label="fitted")
    # plt.title("Extrapolated Exponential Curve")

    # plt.show()

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
    # print(c)

    # ck = c.keys()
    # cv = c.values()
    # fig = plt.figure(figsize=(10, 5))
    # plt.bar(ck, cv, color='maroon', width=1)
    # plt.hist(s.time_between_pos, bins=round(len(set(s.time_between_pos))))
    # plt.hist(time_between_pos, bins=500)

    # plt.show()
    #
    # xy = []
    # xx = []
    # for key in c:
    #     xx.append(key)
    #     c[key] = [c[key]/(len(time_between_pos))]
    #
    #     # xy.append([c[key], c[key]/(len(time_between_pos))])
    # xx.sort()
    # yy = []
    # for x in xx:
    #     yy.append(c[x][0])
    # print(xx, yy)

    # plt.hist(c, bins=round(len(set(xx))))
    # plt.show()

    # r = expon.rvs(loc=1, scale=112, size=10000)
    # plt.hist(r, bins=50)
    # plt.show()

    # fig, ax = plt.subplots(1, 1)
    # print(expon.mean(expon.fit(time_between_pos)))
    # r = expon.rvs(loc=0, scale=110, size=1000)
    # ax.hist(r, density=True, histtype='stepfilled')
    # ax.legend(loc='best', frameon=False)
    # plt.show()

    # sx, sy = [], []
    # for _ in s.shelf_data_times:
    #     sx.append(_[0])
    #     sy.append(_[1])
    #
    # tx = []
    # for t in s.pos_data:
    #     tx.append(t.purchase_time)
    # plt.scatter(sx, sx, color='red')
    # plt.scatter(tx, tx, color='green')

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

    # mu = statistics.mean(time_between_pos)
    # # x = np.random.poisson(mu, len(time_between_pos))
    # # x = poisson.rvs(mu, loc=0, size=(len(time_between_pos)))
    # # print(x)
    # plt.hist(x, bins=round(len(set(x))))
    # plt.show()
