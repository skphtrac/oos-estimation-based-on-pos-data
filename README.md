# Estimating Out-of-Shelf Situations In Real-Time From Point of Sales Data

## General info
This repository consists of the code for a bachelor thesis presented to and conducted for the faculty of economics and business administration of university of duisburg-essen. The aim was to research a cost effective method for retailers to estimate Out-of-Shelf events in real time by using Point of Sale data. The approach for this was to  simulate an exemplary store (*store_framework.py & occupancy.py*) and its assortment over a period of time (*simulation.py*), thus generating realistic Point of Sale data as a base. This data was then used to calculate the time difference between Point of Sale entries of a product and comparing it to the current time since that product has been purchased last in order to estimate a probability of the product being Out-of-Shelf.

## Technologies
The project is created with:
* Python version: 3.10.1
* Scipy version: 1.7.3
* Numpy version: 1.22.1
* Matplotlib version: 3.5.1

## Setup
Before you run this project make sure to have the above mentioned versions of Python, Scipy and Numpy. The files *store_framework.py* and *occupancy.py* make up the framework of the exemplary store. The simulation of Point of Sale data and the Out-of-Shelf estimation described in the bachelor thesis can be executed using the *simulation.py* file. For a more customized simulation the list below shows the variables which influence the simulations outcome and can be modified if necessary. 

* Store
  * Business hours: store_framework.py -> Store() -> opening_time/closing_time
  * Occupancy: occupancy.py -> dictionary with days as keys & time with desired occupancy at this time in percent as values

* Shelves
  * Quantity in store: store_framework.py -> Store().create_shelves() -> range
  * Capacity: store_framework.py -> Store().create_shelves() -> cap

* Products
  * Demand multiplicator: store_framework.py -> Product().generate_demand_factor() -> pop (realistic results with range in 0.005 - 0.15)
  * EAN: store_framework.py -> Product().generate_ean

* Simulation
  * Time period: simulation.py -> TimeSimulation() -> startdate/enddate
