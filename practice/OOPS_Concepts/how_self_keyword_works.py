class Dog:
    def __init__(self, name):
        self.name = name
        print(f"Inside __init__, self is: {id(self)}")
    
    def bark(self):
        print(f"Inside bark, self is: {id(self)}")
        print(f"{self.name} says: Woof!")

# Create two dogs
dog1 = Dog("Buddy")
print(f"dog1 memory address: {id(dog1)}\n")

dog2 = Dog("Max")
print(f"dog2 memory address: {id(dog2)}\n")

# Call bark
dog1.bark()
print(f"dog1 memory address: {id(dog1)}\n")

dog2.bark()
print(f"dog2 memory address: {id(dog2)}")