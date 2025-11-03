class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic sound"
    
    def info(self):
        return f"{self.name} is a {self.species}"

class Dog(Animal):  # Dog inherits from Animal
    def __init__(self, name):
        self.name = name
        self.species = "Canis familiaris"
    
    def make_sound(self):  # Override parent method
        return "Woof!"

class Cat(Animal):
    def __init__(self, name):
        self.name = name
        self.species = "Felis catus"
    
    def make_sound(self):
        return "Meow!"

# Create objects
dog = Dog("Buddy")
cat = Cat("Whiskers")

print(dog.info())  # Method from Animal!
print(dog.make_sound())

print(cat.info())  # Method from Animal!
print(cat.make_sound())

# Check inheritance
print(f"\nDog's parent: {Dog.__bases__}")
print(f"Cat's parent: {Cat.__bases__}")
print(f"Is dog an Animal? {isinstance(dog, Animal)}")
print(f"Is dog a Dog? {isinstance(dog, Dog)}")