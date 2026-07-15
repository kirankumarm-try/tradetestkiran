=== raw dataframe tail ===
Price             Close         High          Low         Open   Volume
Ticker            LT.NS        LT.NS        LT.NS        LT.NS    LT.NS
Date                                                                   
2026-07-02  4059.399902  4110.000000  4012.000000  4100.000000  2481554
2026-07-03  4026.600098  4094.000000  4019.000000  4080.000000  1505428
2026-07-06  4041.000000  4068.000000  4016.500000  4027.699951  1252054
2026-07-07  3991.899902  4046.300049  3975.000000  4044.500000  2442514
2026-07-08  3892.100098  3988.000000  3871.000000  3950.000000  2252308
2026-07-09  3886.000000  3947.300049  3867.800049  3916.000000  5117653
2026-07-10  3945.800049  3953.800049  3905.000000  3905.000000  1283990
2026-07-13  3928.500000  3939.000000  3875.600098  3895.000000  1744547
2026-07-14  3848.699951  3897.899902  3843.000000  3892.000000  2409089
2026-07-15          NaN          NaN          NaN          NaN  3602031

columns: [('Close', 'LT.NS'), ('High', 'LT.NS'), ('Low', 'LT.NS'), ('Open', 'LT.NS'), ('Volume', 'LT.NS')]

last index: 2026-07-15 00:00:00
last Close (raw): Ticker
LT.NS   NaN
Name: 2026-07-15 00:00:00, dtype: float64
last Adj Close (raw): None

last non-null Close: None
last non-null Adj Close: None

Using price column: Close
Traceback (most recent call last):
  File "C:\Users\Nazeer\Desktop\try\debug_yf.py", line 39, in <module>
    print("last RSI (computed):", float(rsi.ffill().bfill().iloc[-1]) if not rsi.empty else None)
                                  ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: float() argument must be a string or a real number, not 'Series'
