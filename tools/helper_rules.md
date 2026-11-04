Project rules for every coding task (they come from mistakes already made on this project; follow all of them):

1. Implement EXACTLY the spec in the brief. Same names, same signatures, same file layout. Do not add features,
   options, prints or extra functions that were not asked for.
2. Quantized values are UNSIGNED (0..15 for 4-bit, 0..255 for 8-bit). Never sign-extend them.
3. No hidden allocation: do not call malloc/calloc/free (C) or allocate large arrays (Python) unless the brief
   says so. Use the buffers you are given.
4. Never write past the end of an output buffer. Check every index against its bound, especially in top-k,
   ring buffers and loops that insert or shift.
5. Two buffers passed to one function must not overlap unless the brief says they may.
6. Remove every unused variable and helper. The C build uses -Wall -Wextra -Werror: an unused variable is a
   build failure. Wrap mixed && and || in parentheses.
7. Check the return value of every function that can fail, and return the error code the spec gives.
8. No "thinking aloud" comments ("Let's assume...", "Actually..."). Comments only where the code is not obvious.
9. When you receive review notes or test failures, apply EVERY note exactly as written and change nothing else.
   If a note quotes a line to delete, delete that exact line.
10. When the brief gives a skeleton, fill only the TODOs and return the whole file otherwise unchanged.
11. Reply with ONE fenced code block containing the complete file (or exactly the definitions requested), and
    nothing else.
