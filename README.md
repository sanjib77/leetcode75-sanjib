# LeetCode 75 Solutions

This repository contains solutions to the LeetCode 75 study plan problems, implemented in Python.

## Problems and Solutions

### String Problems
- [1071. Greatest Common Divisor of Strings](practice/1071%20gcdStrings.py)
- [1768. Merge Strings Alternately](practice/MergeAlternately.py)
- [345. Reverse Vowels of a String](practice/345.py)

### Array Problems
- [1431. Kids With the Greatest Number of Candies](practice/1431.py)

### Linked List Problems
- [234. Palindrome Linked List](practice/PalidromeLinkedList.py)
- [Singly Linked List Implementation](practice/SInglyLinkedListBasics.py)

## Repository Structure

```
leetcode75-sanjib/
└── practice/
    ├── 1071 gcdStrings.py
    ├── 1431.py
    ├── 345.py
    ├── MergeAlternately.py
    ├── PalidromeLinkedList.py
    └── SInglyLinkedListBasics.py
```

## Problem Categories

### Arrays
- [1431. Kids With the Greatest Number of Candies](src/arrays/kids_with_candies.py)

### Linked Lists
- [234. Palindrome Linked List](src/linked_list/palindrome_linked_list.py)
- Basic Singly Linked List Implementation with common operations

### Strings
- [1071. Greatest Common Divisor of Strings](src/strings/gcd_strings.py)
- [1768. Merge Strings Alternately](src/strings/merge_alternately.py)
- [345. Reverse Vowels of a String](src/strings/reverse_vowels.py)

## Testing

Each solution includes comprehensive unit tests. To run the tests:

```bash
# Run all tests
python -m unittest discover tests

# Run tests for a specific category
python -m unittest discover tests/arrays
python -m unittest discover tests/linked_list
python -m unittest discover tests/strings
```

## Time and Space Complexity

Each solution includes documentation about its time and space complexity in the source code comments.

## Contributing

Feel free to contribute by:
1. Creating a new branch
2. Making your changes
3. Writing appropriate tests
4. Creating a pull request

## License

This project is open source and available under the MIT License.
