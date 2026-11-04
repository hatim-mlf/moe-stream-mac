```json
[
  {
    "id": "python-implement-1",
    "prompt": "Write a Python function that takes a list of integers and returns the two numbers that add up to a specific target sum. Assume there is exactly one valid pair and return their indices."
  },
  {
    "id": "python-fix-1",
    "prompt": "Fix the off-by-one error in this Python loop that processes a list: for i in range(len(data)): print(data[i+1]). The loop should print every element without raising an IndexError."
  },
  {
    "id": "python-refactor-1",
    "prompt": "Refactor this Python snippet to use list comprehensions instead of explicit loops: result = []; for x in nums: if x > 0: result.append(x * 2). Return the cleaned version."
  },
  {
    "id": "python-test-1",
    "prompt": "Write unit tests for a Python function that parses a CSV string into a list of dictionaries. Include test cases for empty strings, single rows, and rows with missing values."
  },
  {
    "id": "python-explain-1",
    "prompt": "Explain why this Python code raises a 'TypeError: 'int' object is not iterable' when trying to unpack a single integer into multiple variables: a, b = 5."
  },
  {
    "id": "python-optimize-1",
    "prompt": "Optimize this slow Python loop that calculates the sum of squares for a large list of numbers. The current implementation uses a standard for-loop and is too slow for lists over 1 million elements."
  },
  {
    "id": "c-implement-1",
    "prompt": "Implement a C function that reverses a null-terminated string in place. The function should take a char pointer and return void, modifying the original string directly."
  },
  {
    "id": "c-fix-1",
    "prompt": "Fix the memory leak in this C code snippet: char *buf = malloc(100); strcpy(buf, \"hello\"); return buf; without adding free() calls in the caller."
  },
  {
    "id": "swift-implement-1",
    "prompt": "Write a Swift function that checks if a given string is a valid palindrome, ignoring case and non-alphanumeric characters. Return a boolean value."
  },
  {
    "id": "swift-refactor-1",
    "prompt": "Refactor this Swift code to use modern optionals and guard statements: var name = getUser(); if name != nil { print(name!) }. Make it crash-safe and concise."
  },
  {
    "id": "js-implement-1",
    "prompt": "Write a JavaScript function that debounces another function by a specified delay. The debounced function should cancel any pending execution when called again before the delay expires."
  },
  {
    "id": "sql-design-1",
    "prompt": "Design a small SQL schema for a library system with tables for Books, Authors, and Loans. Include primary keys, foreign keys, and an index on the loan date for fast queries."
  }
]
```