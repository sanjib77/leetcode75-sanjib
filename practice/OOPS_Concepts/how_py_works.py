class Money:
    def __init__(self, amount):
        self.amount = amount
    
    def __add__(self, other):
        return Money(self.amount + other.amount)
    
    def __str__(self):
        return f"${self.amount}"

wallet1 = Money(50)
wallet2 = Money(30)
total = wallet1 + wallet2

# CHECK THE TYPE!
print(f"Type of wallet1: {type(wallet1)}")
print(f"Type of wallet2: {type(wallet2)}")
print(f"Is wallet1 a Money object? {isinstance(wallet1, Money)}")
print(f"Is wallet1 a string? {isinstance(wallet1, str)}")

# What's inside?
print(f"\nwallet1.amount = {wallet1.amount}")
print(f"wallet2.amount = {wallet2.amount}")
print(f"total.amount = {total.amount}")

# The difference
print(f"\nWhen you PRINT wallet1: {wallet1}")  # Calls __str__
print(f"wallet1 itself is: {repr(wallet1)}")   # Shows real object