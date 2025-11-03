class Dog:
    species = "Canis familiaris"

dog1 = Dog()

# Explore what Python stores internally
print("Dunder attributes of dog1:")
print(f"__class__: {dog1.__class__}")
print(f"__dict__: {dog1.__dict__}")
print(f"__sizeof__: {dog1.__sizeof__()} bytes")

print("\nDunder attributes of Dog class:")
print(f"__name__: {Dog.__name__}")
print(f"__dict__: {Dog.__dict__}")
print(f"__bases__: {Dog.__bases__}")
print(f"__module__: {Dog.__module__}")

# Even integers have dunders!
x = 42
print("\nInteger dunders:")
print(f"(42).__class__: {x.__class__}")
print(f"(42).__sizeof__(): {x.__sizeof__()} bytes")

# See ALL dunders
print("\nAll attributes of an integer:")
print([attr for attr in dir(x) if attr.startswith('__')])