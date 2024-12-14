import os
import psycopg2
import psycopg2.extras
import tabulate
from dotenv import load_dotenv

def query():
    """
    Used for testing standard queries in SQL.
    """
    load_dotenv()

    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    dbname = os.getenv('DBNAME')

    conn = psycopg2.connect("dbname="+dbname+" user="+user+" password="+password,
                            cursor_factory=psycopg2.extras.DictCursor)
    cur = conn.cursor()
    #q1
    #cur.execute("CREATE VIEW G1 AS SELECT prod, SUM(quant) AS sum_quant_1 FROM Sales WHERE month = 1 AND year = 2020 AND day >= 2 GROUP BY prod; CREATE VIEW G2 AS SELECT prod, AVG(quant) AS avg_quant_2 FROM Sales WHERE month = 2 AND year = 2020 AND day <= 30 GROUP BY prod; CREATE VIEW G3 AS SELECT prod, MIN(quant) AS min_quant_3 FROM Sales x WHERE month = 3 AND year = 2020 AND day != 16 AND quant > (SELECT AVG(quant) FROM Sales y WHERE y.prod = x.prod AND month = 2 AND year = 2020 AND day <= 30) GROUP BY prod; CREATE VIEW G4 AS SELECT prod, MAX(quant) AS max_quant_4 FROM Sales x WHERE month = 4 AND year = 2020 AND day < 20 AND quant < (SELECT MIN(quant) FROM Sales y WHERE y.prod = x.prod AND month = 3 AND year = 2020 AND day != 16 AND quant > (SELECT AVG(quant) FROM Sales z WHERE z.prod = y.prod AND month = 2 AND year = 2020 AND day <= 30)) GROUP BY prod; CREATE VIEW G5 AS SELECT prod, COUNT(*) AS count_quant_5 FROM Sales WHERE month = 5 AND day > 2 AND state = 'NJ' GROUP BY prod; SELECT G1.prod, G1.sum_quant_1, G2.avg_quant_2, G3.min_quant_3, G4.max_quant_4, G5.count_quant_5 FROM G1 JOIN G2 ON G1.prod = G2.prod JOIN G3 ON G1.prod = G3.prod JOIN G4 ON G1.prod = G4.prod JOIN G5 ON G1.prod = G5.prod WHERE G5.count_quant_5 > 11;")
    #cur.execute("SELECT G1.prod, G1.sum_quant_1, G2.avg_quant_2, G3.min_quant_3, G4.max_quant_4, G5.count_quant_5 FROM G1 JOIN G2 ON G1.prod = G2.prod JOIN G3 ON G1.prod = G3.prod JOIN G4 ON G1.prod = G4.prod JOIN G5 ON G1.prod = G5.prod WHERE G5.count_quant_5 > 11;")
    #q2
    #cur.execute("CREATE VIEW B1 AS SELECT x.prod, x.month, AVG(y.quant) AS avg_x FROM Sales x JOIN Sales y ON x.prod = y.prod AND x.month > y.month GROUP BY x.prod, x.month; CREATE VIEW B2 AS SELECT x.prod, x.month, AVG(y.quant) AS avg_y FROM Sales x JOIN Sales y ON x.prod = y.prod AND x.month < y.month GROUP BY x.prod, x.month; SELECT B1.prod, B1.month, B1.avg_x, B2.avg_y FROM B1 JOIN B2 ON B1.prod = B2.prod AND B1.month = B2.month;")
    #cur.execute("SELECT B1.prod, B1.month, B1.avg_x, B2.avg_y FROM B1 JOIN B2 ON B1.prod = B2.prod AND B1.month = B2.month;")
    #q4
    #cur.execute("SELECT cust, prod, COUNT(quant) AS total_count, SUM(quant) AS total_sum, AVG(quant) AS average_quantity, MIN(quant) AS min_quantity, MAX(quant) AS max_quantity FROM sales GROUP BY cust, prod;")
    #noF
    #cur.execute("SELECT cust, prod FROM sales GROUP BY cust, prod;")
    #noV
    #cur.execute("SELECT cust, prod, SUM(SUM(quant)) OVER () AS total_sum_quant FROM sales GROUP BY cust, prod;")
    return tabulate.tabulate(cur.fetchall(),
                             headers="keys", tablefmt="psql")


def main():
    print(query())


if "__main__" == __name__:
    main()
