class BankAccount:
    def __init__(self, name, balance):
        self.name = name
        self.__balance = balance  # Double underscore makes it "private"
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.__balance
    
    def __str__(self):
        return f"{self.name}: ${self.__balance}"

# Create account
acc = BankAccount("Arpit", 1000)
print(acc)

# Try to access balance
print(f"\nUsing method: {acc.get_balance()}")

# Try to access directly
try:
    print(f"Direct access: {acc.__balance}")
except AttributeError as e:
    print(f"Error: {e}")

# Check what's in __dict__
print(f"\nacc.__dict__: {acc.__dict__}")

# The secret way to access it (name mangling)
print(f"Secret access: {acc._BankAccount__balance}")


#The truth about privacy in Python
class BankAccount1:
    def __init__(self, balance):
        self.__balance = 1  # "Private"
        self._balance2 = 2  # "Protected" (single underscore)
        self.balance3 = balance   # Public

acc = BankAccount1(1000)

# All three are accessible!
print(f"Single underscore: {acc._balance2}")
print(f"No underscore: {acc.balance3}")

#to access private variables without getter
print(f"Double underscore (mangled): {acc._BankAccount1__balance}")

"""
Python's Philosophy: "We're all consenting adults here"
Unlike Java/C++ where private truly blocks access, Python says:

"I'll make it hard to access by accident, but if you REALLY need it, you can access it."

Name mangling (__balance) prevents:

✅ Accidental overwrites in inheritance (like you saw with Parent/Child)
✅ Accidental external access (acc.__balance fails)
Name mangling does NOT prevent:

❌ Intentional access (acc._BankAccount__balance still works)
"""

#Best Practice in Python:

class BankAccount2:
    def __init__(self, balance):
        self._balance = balance  # Single underscore (convention)
    
    def get_balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
