Before you answer, check your code against this list. It comes from 588 of your own failed solutions.

C (compiled with `cc -std=c99 -Wall -Wextra -Werror`; every warning is an error):
1. Include every standard header you use: `<stddef.h>` for NULL and size_t, `<stdint.h>` for fixed-width
   types, `<string.h>` for memcpy/memset, `<math.h>` for math, `<stdlib.h>` for malloc/abs, `<limits.h>` for
   INT_MAX. Do not include any project header unless the task names it.
2. No unused variables or parameters. Remove them, or cast an intentionally unused parameter: `(void)x;`.
3. Do not compare signed and unsigned values. Use size_t for sizes and indices, and cast explicitly.
4. Keep the exact function signature from the spec: name, parameter order, types, const, return type.

Python:
5. Implement every error case the spec lists, with exactly the exception type it names, and check them before
   doing any work. Reject wrong types, empty or invalid shapes, out-of-range values and non-finite numbers as
   the spec says.
6. Convert inputs as the spec says (e.g. `np.asarray(x, dtype=np.float64)`), then validate ndim and shape
   before indexing (no `arr.shape[-1]` on a 0-d array).
7. Return exactly the type, dtype and shape the spec states.

Both:
8. Handle the edge cases explicitly: empty input, length 1, boundaries, last element, exact ties.
9. Do not add features, parameters or output the spec does not ask for.
10. Reply with one fenced code block containing the complete file and nothing else.
