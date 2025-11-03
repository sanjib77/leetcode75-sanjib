class BankAccount:
    bank_name = "Python Bank"  # CLASS ATTRIBUTE
    interest_rate = 0.05       # CLASS ATTRIBUTE
    
    def __init__(self, name, balance):
        self.name = name           # INSTANCE ATTRIBUTE
        self.balance = balance     # INSTANCE ATTRIBUTE
    
    def show_info(self):
        print(f"Owner: {self.name}")
        print(f"Balance: ${self.balance}")
        print(f"Bank: {self.bank_name}")
        print(f"Interest: {self.interest_rate}")
        print(f"Memory of bank_name: {id(self.bank_name)}")

# Create two accounts
acc1 = BankAccount("Arpit", 1000)
acc2 = BankAccount("Rohan", 2000)

print("Account 1:")
acc1.show_info()
print(f"\nAccount 2:")
acc2.show_info()

print("\n" + "="*50)
print(f"BankAccount.bank_name memory: {id(BankAccount.bank_name)}")
print(f"Are they sharing the same bank_name? {id(acc1.bank_name) == id(acc2.bank_name)}")

# Change class attribute
BankAccount.interest_rate = 0.07

print(f"\nAfter changing class attribute:")
print(f"acc1 interest: {acc1.interest_rate}")
print(f"acc2 interest: {acc2.interest_rate}")

# Change instance attribute
acc1.interest_rate = 0.10  # Creates NEW instance attribute!

print(f"\nAfter changing acc1's interest:")
print(f"acc1 interest: {acc1.interest_rate}")
print(f"acc2 interest: {acc2.interest_rate}")
print(f"BankAccount interest: {BankAccount.interest_rate}")

# Check __dict__
print(f"\nacc1.__dict__: {acc1.__dict__}")
print(f"acc2.__dict__: {acc2.__dict__}")