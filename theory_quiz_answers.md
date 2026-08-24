# Week 05 — Theory Quiz Answers (Part 2)

**Q1. What are the two separate pieces that make up a running notebook, and which one holds your variables in memory?**

The **cells** (where code/markdown is written and displayed) and the **kernel** (the live Python process that actually executes them). The kernel holds all variables in memory — that's why deleting a cell does not delete its variables, and why "Restart Kernel" wipes everything while cells stay on screen.

**Q2. `arr.mean(axis=0)` returns one number instead of per-column means — what does that say about `arr`'s shape, and how to check?**

It means `arr` has only **one row** (shape like `(n,)` or `(1, m)`), so axis 0 collapses everything into a single value. Check with `arr.shape` or `arr.ndim` — if `ndim == 1`, there are no columns to average across; reshape first (e.g., `arr.reshape(-1, m)`) to get per-column means.

**Q3. Why does `df[(df['a'] > 5) and (df['b'] < 10)]` fail instead of working element-wise?**

Python's `and` calls `bool()` on its operands to get ONE truth value, but a pandas Series of booleans has no single truth value — hence *"The truth value of a Series is ambiguous"*. Element-wise logic needs the bitwise operator `&`, with each comparison wrapped in parentheses because `&` binds tighter than `>` / `<`.

**Q4. Reshape a NumPy array, change a value in the reshaped version — why does the original change too?**

`.reshape()` returns a **view**, not a copy: a new array object that reads and writes through the *same underlying memory buffer*, just with different shape metadata. Writing through either array edits the shared buffer, so both views reflect the change. Use `.reshape(...).copy()` (or `np.array(orig)`) when independence is needed.

**Q5. A column is 40% missing — one argument for `dropna()`, one for `fillna()`, and what decides it?**

*For dropping:* if rows with missing values are a biased subsample anyway (e.g., sensor dead during failures), imputing invents data and dropna keeps every remaining number truly observed. *For filling:* deleting 40% of the dataset throws away the other columns' valid information and can badly bias results. The decider is **why the values are missing** (MCAR vs informative missingness) and how much signal the other columns carry — mechanism matters more than percentage.

**Q6. Why is `df['price'] * df['qty']` faster than `.apply(lambda row: ..., axis=1)`?**

The vectorized form dispatches one operation over whole columns as contiguous memory blocks, executed in optimized C/NumPy loops with no Python-level overhead per row. `.apply(axis=1)` iterates row by row in pure Python, constructing a Series for each row and calling the lambda once per row — thousands of interpreted function calls doing what C does natively, typically 10–100× slower at this scale.

**Q7. Comparing average resolution time across four categories — correct plot type, and what would a line chart wrongly imply?**

A **bar chart** — bar heights compare discrete, unrelated groups directly. A line chart would connect categories whose order is arbitrary, wrongly implying a continuous trend between them ("resolution rises from Chat→Email→Phone"), inviting slope interpretations that mean nothing for nominal categories.

**Q8. One concrete way to make a real chart misleading without changing any data?**

Truncate the y-axis to start just below the maximum bar (e.g., ylim starting at 11.6 h instead of 0): tiny differences then fill the entire visual field and look like multi-fold disparities. Same numbers, opposite story — demonstrated concretely on our own Chart-2 data in `05_misleading_vs_honest.png`.
