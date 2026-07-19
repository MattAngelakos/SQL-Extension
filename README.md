# Ad-Hoc OLAP Query Engine — MF/EMF SQL Extension

A query processing engine for **Multi-Feature (MF)** and **Extended Multi-Feature (EMF)** queries — an
extended SQL syntax for ad-hoc OLAP analysis that expresses in a single pass what standard SQL requires
multiple correlated subqueries and view joins to compute.

Rather than interpreting a query, the engine **generates a standalone executable Python program**
implementing the multi-scan algorithm for that query. The generated program connects to PostgreSQL,
materializes the result through pandas, and prints a formatted table.

CS 562 Final Project (Fall 2024, Prof. Kim) — Matthew Angelakos and Christopher Arias.

## How it works

A query is specified by the six-parameter **Phi operator**:

| Parameter | Meaning |
|---|---|
| `SELECT ATTRIBUTE(S)` | Attributes and aggregates to project |
| `NUMBER OF GROUPING VARIABLES(n)` | Count of grouping variables |
| `GROUPING ATTRIBUTES(V)` | Attributes to group by |
| `F-VECT([F])` | Aggregate functions, as `<var>_<agg>_<attribute>` |
| `SELECT CONDITION-VECT([σ])` | Per-grouping-variable selection conditions |
| `HAVING_CONDITION(G)` | Predicate applied to the aggregated result |

Conditions are parsed into abstract syntax trees using Python's `ast` module with a custom SQL
operator-precedence table, then translated into evaluable predicates. Regex-based attribute resolution
binds each grouping variable's references to the correct aggregate or grouping source.

## Requirements

- PostgreSQL
- Python 3 with the packages in [requirements.txt](requirements.txt): `psycopg2`, `python-dotenv`, `pandas`, `tabulate`

```bash
pip install -r requirements.txt
```

Create the `sales` table the queries run against:

```bash
psql -d <your_db> -f "load_sales_10000_table (NEW).sql"
```

Copy [env.example](env.example) to `.env` and fill in your database credentials:

```
USER=
PASSWORD=
DBNAME=
```

## Running

```bash
python generator.py
```

The program prompts for either an input file name or the Phi operator parameters entered inline. It
writes the generated algorithm to `_generated.py` and runs it automatically. To generate without
executing, remove the `subprocess.run` call at [generator.py:584](generator.py).

## Example

Input — [q1.txt](q1.txt), five grouping variables with chained dependencies and a having clause:

```
SELECT ATTRIBUTE(S):
prod, 1_sum_quant, 2_avg_quant, 3_min_quant, 4_max_quant, 5_count_quant
NUMBER OF GROUPING VARIABLES(n):
5
GROUPING ATTRIBUTES(V):
prod
F-VECT([F]):
1_sum_quant, 2_avg_quant, 3_min_quant, 4_max_quant, 5_count_quant
SELECT CONDITION-VECT([σ]):
1.month=1 and 1.year=2020 and 1.prod=prod and 1.day>=2
2.month=2 and 2.year=2020 and 2.prod=prod and 2.day<=30
3.month=3 and 3.year=2020 and 3.prod=prod and 3.day!=16 and 3.quant>2_avg_quant
4.month=4 and 4.year=2020 and 4.prod=prod and 4.day<20 and 4.quant<3_min_quant
5.month=5 and 5.prod=prod and 5.day>2 and 5.state='NJ'
HAVING_CONDITION(G):
5_count_quant>11
```

The equivalent standard SQL — five views, two levels of correlated subquery, and a four-way join — is in
[q1_standard.txt](q1_standard.txt) for comparison.

## Test queries

Each `q*.txt` / `test*.txt` file is an input; the matching `*_standard.txt` file is the hand-written
standard SQL used to validate the engine's output against a 10,000-row sales dataset.

| Input | Covers |
|---|---|
| [q1.txt](q1.txt) | Five grouping variables, chained aggregate dependencies, having clause |
| [q2.txt](q2.txt) | Multiple grouping attributes |
| [q4.txt](q4.txt) | Aggregate arithmetic in the select list |
| [testnoF.txt](testnoF.txt) | No aggregates |
| [testnoV.txt](testnoV.txt) | No grouping variables |
| [testnosigma.txt](testnosigma.txt) | No selection conditions |
| [testnohaving.txt](testnohaving.txt) | No having clause |

## Files

- [generator.py](generator.py) — the engine; parses the Phi operator and emits the query program
- [sql.py](sql.py) — runs the hand-written standard SQL used to validate engine output
- [test_generator.py](test_generator.py) — tests for the generator
- `*_generated.py` — disposable output, not tracked in git
