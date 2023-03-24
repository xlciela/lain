# orderMgt module

class OrderManagement:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.orders = []

    def add_order(self, order):
        self.orders.append(order)

    def remove_order(self, order):
        self.orders.remove(order)

    def get_orders(self):
        return self.orders
    
    # create stop orders
    def 