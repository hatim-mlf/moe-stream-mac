# Task: write 12 realistic coding requests as a JSON array

We need 12 DIFFERENT coding requests that a developer would send to an AI coding assistant, to study which parts of
a model are used for coding. Make them varied:
- languages: Python (3), C (2), Swift (2), JavaScript/TypeScript (2), SQL (1), Bash (1), Rust (1)
- kinds: implement a function, fix a given bug, refactor a short snippet, write unit tests, explain an error message,
  optimise a slow loop, design a small data structure
- each 1 to 3 sentences; when a snippet is part of the request, include it (max 8 lines, plain text, no markdown fences).

Output format, exactly: a JSON array of 12 objects {"id": "<lang>-<kind>-<n>", "prompt": "<text>"}.
Return only the JSON array in one fenced json code block.
