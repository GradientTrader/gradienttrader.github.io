
# Cryptocurrency Trader Agent

### UCB MIDS 2017 Winter Capstone Project
#### Ramsey Aweti, Shuang Chan, GuangZhi(Frank) Xie, Jason Xie

### Global Variables


```python
import random 
random.seed(3456)
```


```python
num_coins_per_order = 0 #0 means buy/sell all per order
recent_k = 150
```

External States: ["current_price", "rolling_mean", "rolling_std", "cross_upper_band", "cross_lower_band", "upper_band", "lower_band", "price_over_sma"]

Internal States: ["coin", "cash", "total_value", "is_holding_coin", "return_since_entry"]

## ETH

### Benchmarks


```python
from v2 import run_benchmarks
```


```python
run_benchmarks.run_bollingerband_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k)
```




    53.089268317694724




```python
run_benchmarks.run_random_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k)
```




    -11.87166091088876




```python
run_benchmarks.run_alwaysbuy_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k)
```




    -11.87166091088876



### DDQN Agent


```python
from v2.ddqn_agent import DDQNAgent
```

    Using TensorFlow backend.



```python
eth_agent = DDQNAgent(recent_k = 150, num_coins_per_order = num_coins_per_order, epsilon_min = 0,
                     external_states = ["upper_band", "lower_band", "price_over_sma"],
                     internal_states = ["is_holding_coin", "return_since_entry"])
```


```python
#eth_agent.plot_external_states()
```


```python
eth_agent.train(num_episodes=800)
```

    episode: 1/800, returns: -56.6307707724, epsilon: 1.0
    episode: 2/800, returns: -33.9865558191, epsilon: 0.99
    episode: 3/800, returns: 26.8041787907, epsilon: 0.98
    episode: 4/800, returns: 70.4954971432, epsilon: 0.97
    episode: 5/800, returns: -54.3652996011, epsilon: 0.96
    episode: 6/800, returns: -50.4580543133, epsilon: 0.95
    episode: 7/800, returns: 3.18558751923, epsilon: 0.94
    episode: 8/800, returns: -23.2968386276, epsilon: 0.93
    episode: 9/800, returns: -53.3790984996, epsilon: 0.92
    episode: 10/800, returns: -35.9988668831, epsilon: 0.91
    episode: 11/800, returns: -31.9435569414, epsilon: 0.9
    episode: 12/800, returns: 68.7318447255, epsilon: 0.9
    episode: 13/800, returns: 9.78873313137, epsilon: 0.89
    episode: 14/800, returns: 3.37938086377, epsilon: 0.88
    episode: 15/800, returns: -25.1648565875, epsilon: 0.87
    episode: 16/800, returns: -44.5434089438, epsilon: 0.86
    episode: 17/800, returns: -50.9297465487, epsilon: 0.85
    episode: 18/800, returns: 7.35112271979, epsilon: 0.84
    episode: 19/800, returns: -44.6340574753, epsilon: 0.83
    episode: 20/800, returns: -17.8967976193, epsilon: 0.83
    episode: 21/800, returns: -2.36308358162, epsilon: 0.82
    episode: 22/800, returns: -23.7411818197, epsilon: 0.81
    episode: 23/800, returns: 1.91215990561, epsilon: 0.8
    episode: 24/800, returns: 159.854860039, epsilon: 0.79
    episode: 25/800, returns: -10.9879870386, epsilon: 0.79
    episode: 26/800, returns: -11.7538276048, epsilon: 0.78
    episode: 27/800, returns: -24.9198591352, epsilon: 0.77
    episode: 28/800, returns: -54.4140059917, epsilon: 0.76
    episode: 29/800, returns: 31.158348332, epsilon: 0.75
    episode: 30/800, returns: -26.422335557, epsilon: 0.75
    episode: 31/800, returns: -10.1641084176, epsilon: 0.74
    episode: 32/800, returns: -51.5277328043, epsilon: 0.73
    episode: 33/800, returns: -3.13618558699, epsilon: 0.72
    episode: 34/800, returns: -40.2161920106, epsilon: 0.72
    episode: 35/800, returns: -21.9154542969, epsilon: 0.71
    episode: 36/800, returns: 12.5070332919, epsilon: 0.7
    episode: 37/800, returns: -13.5869288234, epsilon: 0.7
    episode: 38/800, returns: -9.67328345483, epsilon: 0.69
    episode: 39/800, returns: -42.3642998474, epsilon: 0.68
    episode: 40/800, returns: -2.71065541304, epsilon: 0.68
    episode: 41/800, returns: -27.8924652756, epsilon: 0.67
    episode: 42/800, returns: 31.8400492058, epsilon: 0.66
    episode: 43/800, returns: -23.78129137, epsilon: 0.66
    episode: 44/800, returns: 43.905260286, epsilon: 0.65
    episode: 45/800, returns: 77.9776086262, epsilon: 0.64
    episode: 46/800, returns: -31.0063209563, epsilon: 0.64
    episode: 47/800, returns: -28.5583402419, epsilon: 0.63
    episode: 48/800, returns: -27.7814052022, epsilon: 0.62
    episode: 49/800, returns: 25.4713195506, epsilon: 0.62
    episode: 50/800, returns: -16.6112515004, epsilon: 0.61
    episode: 51/800, returns: 28.6667635101, epsilon: 0.61
    episode: 52/800, returns: 11.5615458849, epsilon: 0.6
    episode: 53/800, returns: -8.74488257814, epsilon: 0.59
    episode: 54/800, returns: 9.26614790826, epsilon: 0.59
    episode: 55/800, returns: -16.8053213007, epsilon: 0.58
    episode: 56/800, returns: -12.1279975529, epsilon: 0.58
    episode: 57/800, returns: -8.77390952675, epsilon: 0.57
    episode: 58/800, returns: 22.5791219953, epsilon: 0.56
    episode: 59/800, returns: -59.4621785423, epsilon: 0.56
    episode: 60/800, returns: -22.6519193352, epsilon: 0.55
    episode: 61/800, returns: -6.6611086758, epsilon: 0.55
    episode: 62/800, returns: 60.1448015641, epsilon: 0.54
    episode: 63/800, returns: 83.9389103459, epsilon: 0.54
    episode: 64/800, returns: 0.716766345392, epsilon: 0.53
    episode: 65/800, returns: 16.2550850159, epsilon: 0.53
    episode: 66/800, returns: 27.478931885, epsilon: 0.52
    episode: 67/800, returns: 35.0178472674, epsilon: 0.52
    episode: 68/800, returns: -31.8791089179, epsilon: 0.51
    episode: 69/800, returns: 15.0937622221, epsilon: 0.5
    episode: 70/800, returns: -28.7173548721, epsilon: 0.5
    episode: 71/800, returns: -36.7096192939, epsilon: 0.49
    episode: 72/800, returns: -17.7889196519, epsilon: 0.49
    episode: 73/800, returns: -3.36472971444, epsilon: 0.48
    episode: 74/800, returns: 3.06928942629, epsilon: 0.48
    episode: 75/800, returns: -31.0142956406, epsilon: 0.48
    episode: 76/800, returns: -31.7278156147, epsilon: 0.47
    episode: 77/800, returns: -28.5038931894, epsilon: 0.47
    episode: 78/800, returns: -10.0146127676, epsilon: 0.46
    episode: 79/800, returns: -4.67314944958, epsilon: 0.46
    episode: 80/800, returns: 24.7821397935, epsilon: 0.45
    episode: 81/800, returns: 29.9622020875, epsilon: 0.45
    episode: 82/800, returns: 16.980048038, epsilon: 0.44
    episode: 83/800, returns: -47.1216255559, epsilon: 0.44
    episode: 84/800, returns: 30.6153065463, epsilon: 0.43
    episode: 85/800, returns: 84.3212662132, epsilon: 0.43
    episode: 86/800, returns: -25.8251481924, epsilon: 0.43
    episode: 87/800, returns: 6.29668943912, epsilon: 0.42
    episode: 88/800, returns: 45.5395062904, epsilon: 0.42
    episode: 89/800, returns: -8.39856215784, epsilon: 0.41
    episode: 90/800, returns: -27.1087591987, epsilon: 0.41
    episode: 91/800, returns: -25.1119795074, epsilon: 0.4
    episode: 92/800, returns: -12.1824674703, epsilon: 0.4
    episode: 93/800, returns: 12.3067512402, epsilon: 0.4
    episode: 94/800, returns: 43.3773818106, epsilon: 0.39
    episode: 95/800, returns: -24.5130820997, epsilon: 0.39
    episode: 96/800, returns: -28.3627982763, epsilon: 0.38
    episode: 97/800, returns: 15.3514793783, epsilon: 0.38
    episode: 98/800, returns: -29.2118609607, epsilon: 0.38
    episode: 99/800, returns: -33.4421427386, epsilon: 0.37
    episode: 100/800, returns: -7.37495358936, epsilon: 0.37
    episode: 101/800, returns: -31.7788915138, epsilon: 0.37
    episode: 102/800, returns: 53.1336387074, epsilon: 0.36
    episode: 103/800, returns: -59.339008546, epsilon: 0.36
    episode: 104/800, returns: 24.9691730562, epsilon: 0.36
    episode: 105/800, returns: 16.6910749326, epsilon: 0.35
    episode: 106/800, returns: 22.3392965084, epsilon: 0.35
    episode: 107/800, returns: 31.5599517581, epsilon: 0.34
    episode: 108/800, returns: -47.6120980102, epsilon: 0.34
    episode: 109/800, returns: 27.5792302911, epsilon: 0.34
    episode: 110/800, returns: -36.0470973836, epsilon: 0.33
    episode: 111/800, returns: 1.70522588679, epsilon: 0.33
    episode: 112/800, returns: -12.2606027476, epsilon: 0.33
    episode: 113/800, returns: 2.98232955022, epsilon: 0.32
    episode: 114/800, returns: 0.232791446738, epsilon: 0.32
    episode: 115/800, returns: -51.4039701557, epsilon: 0.32
    episode: 116/800, returns: 33.1988195974, epsilon: 0.31
    episode: 117/800, returns: -16.7650093129, epsilon: 0.31
    episode: 118/800, returns: 67.7213114372, epsilon: 0.31
    episode: 119/800, returns: -38.6061916726, epsilon: 0.31
    episode: 120/800, returns: -45.0220986722, epsilon: 0.3
    episode: 121/800, returns: -24.3584852844, epsilon: 0.3
    episode: 122/800, returns: 73.501860568, epsilon: 0.3
    episode: 123/800, returns: 28.0069166438, epsilon: 0.29
    episode: 124/800, returns: -24.6341057824, epsilon: 0.29
    episode: 125/800, returns: -16.3676255659, epsilon: 0.29
    episode: 126/800, returns: -29.5182051213, epsilon: 0.28
    episode: 127/800, returns: 35.8977008157, epsilon: 0.28
    episode: 128/800, returns: -34.2728933557, epsilon: 0.28
    episode: 129/800, returns: -51.576440556, epsilon: 0.28
    episode: 130/800, returns: 9.79265846341, epsilon: 0.27
    episode: 131/800, returns: 1.99603933972, epsilon: 0.27
    episode: 132/800, returns: 54.2511210166, epsilon: 0.27
    episode: 133/800, returns: -9.50805201063, epsilon: 0.27
    episode: 134/800, returns: -9.63687314202, epsilon: 0.26
    episode: 135/800, returns: 13.9835109158, epsilon: 0.26
    episode: 136/800, returns: -20.0335782782, epsilon: 0.26
    episode: 137/800, returns: 10.6435789339, epsilon: 0.25
    episode: 138/800, returns: -26.592125727, epsilon: 0.25
    episode: 139/800, returns: -48.97577672, epsilon: 0.25
    episode: 140/800, returns: 3.1296963806, epsilon: 0.25
    episode: 141/800, returns: -24.040081551, epsilon: 0.24
    episode: 142/800, returns: -12.9583766466, epsilon: 0.24
    episode: 143/800, returns: -49.3346580997, epsilon: 0.24
    episode: 144/800, returns: 15.110206817, epsilon: 0.24
    episode: 145/800, returns: -38.0095290844, epsilon: 0.24
    episode: 146/800, returns: -5.0271919687, epsilon: 0.23
    episode: 147/800, returns: -21.9285932606, epsilon: 0.23
    episode: 148/800, returns: 33.2233512095, epsilon: 0.23
    episode: 149/800, returns: -35.3119901941, epsilon: 0.23
    episode: 150/800, returns: -19.7687424471, epsilon: 0.22
    episode: 151/800, returns: -28.2822793748, epsilon: 0.22
    episode: 152/800, returns: -29.2832650979, epsilon: 0.22
    episode: 153/800, returns: 9.13933255292, epsilon: 0.22
    episode: 154/800, returns: -16.8712675551, epsilon: 0.21
    episode: 155/800, returns: -28.2270240196, epsilon: 0.21
    episode: 156/800, returns: -32.8481616546, epsilon: 0.21
    episode: 157/800, returns: 7.46206829721, epsilon: 0.21
    episode: 158/800, returns: 6.26969815748, epsilon: 0.21
    episode: 159/800, returns: -2.73226795243, epsilon: 0.2
    episode: 160/800, returns: -7.81296849868, epsilon: 0.2
    episode: 161/800, returns: 9.33281321329, epsilon: 0.2
    episode: 162/800, returns: -28.6404741334, epsilon: 0.2
    episode: 163/800, returns: -11.0186447235, epsilon: 0.2
    episode: 164/800, returns: 9.45409664922, epsilon: 0.19
    episode: 165/800, returns: -24.5348331286, epsilon: 0.19
    episode: 166/800, returns: 46.5223163657, epsilon: 0.19
    episode: 167/800, returns: 5.41154706943, epsilon: 0.19
    episode: 168/800, returns: -15.5225574677, epsilon: 0.19
    episode: 169/800, returns: -15.0172853117, epsilon: 0.18
    episode: 170/800, returns: -19.8440940531, epsilon: 0.18
    episode: 171/800, returns: -12.1059778166, epsilon: 0.18
    episode: 172/800, returns: -9.96951110822, epsilon: 0.18
    episode: 173/800, returns: -25.5199105979, epsilon: 0.18
    episode: 174/800, returns: 6.8135166397, epsilon: 0.18
    episode: 175/800, returns: -7.4596321225, epsilon: 0.17
    episode: 176/800, returns: -10.8305198884, epsilon: 0.17
    episode: 177/800, returns: -11.1727336847, epsilon: 0.17
    episode: 178/800, returns: 21.9884697705, epsilon: 0.17
    episode: 179/800, returns: 31.6026498017, epsilon: 0.17
    episode: 180/800, returns: -15.3813228267, epsilon: 0.17
    episode: 181/800, returns: 15.4649708518, epsilon: 0.16
    episode: 182/800, returns: 15.3504064829, epsilon: 0.16
    episode: 183/800, returns: 11.1745279343, epsilon: 0.16
    episode: 184/800, returns: -10.9590629015, epsilon: 0.16
    episode: 185/800, returns: 8.20143522138, epsilon: 0.16
    episode: 186/800, returns: 21.6892828512, epsilon: 0.16
    episode: 187/800, returns: -1.11747797003, epsilon: 0.15
    episode: 188/800, returns: 9.14116501233, epsilon: 0.15
    episode: 189/800, returns: -32.4684981739, epsilon: 0.15
    episode: 190/800, returns: 3.65306544451, epsilon: 0.15
    episode: 191/800, returns: 17.3717965644, epsilon: 0.15
    episode: 192/800, returns: 7.18931126067, epsilon: 0.15
    episode: 193/800, returns: -39.472590777, epsilon: 0.15
    episode: 194/800, returns: -0.548923907035, epsilon: 0.14
    episode: 195/800, returns: 90.9983793067, epsilon: 0.14
    episode: 196/800, returns: 17.502671941, epsilon: 0.14
    episode: 197/800, returns: 1.54514701452, epsilon: 0.14
    episode: 198/800, returns: 5.20331008504, epsilon: 0.14
    episode: 199/800, returns: 14.7979916285, epsilon: 0.14
    episode: 200/800, returns: 32.7842324769, epsilon: 0.14
    episode: 201/800, returns: -13.59738087, epsilon: 0.13
    episode: 202/800, returns: 49.3789740533, epsilon: 0.13
    episode: 203/800, returns: -35.2230492091, epsilon: 0.13
    episode: 204/800, returns: -45.1808391505, epsilon: 0.13
    episode: 205/800, returns: -1.67593579909, epsilon: 0.13
    episode: 206/800, returns: -4.1026944414, epsilon: 0.13
    episode: 207/800, returns: 6.29498714689, epsilon: 0.13
    episode: 208/800, returns: -7.67363394164, epsilon: 0.12
    episode: 209/800, returns: -4.02830890476, epsilon: 0.12
    episode: 210/800, returns: -1.43969094466, epsilon: 0.12
    episode: 211/800, returns: -24.1420175877, epsilon: 0.12
    episode: 212/800, returns: 1.39915441031, epsilon: 0.12
    episode: 213/800, returns: -3.73430534369, epsilon: 0.12
    episode: 214/800, returns: -35.4470817712, epsilon: 0.12
    episode: 215/800, returns: 10.3386664459, epsilon: 0.12
    episode: 216/800, returns: 6.67065343307, epsilon: 0.12
    episode: 217/800, returns: 4.74044096509, epsilon: 0.11
    episode: 218/800, returns: -11.9005112812, epsilon: 0.11
    episode: 219/800, returns: -8.34641809518, epsilon: 0.11
    episode: 220/800, returns: -7.71627583619, epsilon: 0.11
    episode: 221/800, returns: 9.44436195815, epsilon: 0.11
    episode: 222/800, returns: 26.6777632248, epsilon: 0.11
    episode: 223/800, returns: -5.05934967335, epsilon: 0.11
    episode: 224/800, returns: -8.44964914845, epsilon: 0.11
    episode: 225/800, returns: 2.95126839015, epsilon: 0.11
    episode: 226/800, returns: -17.8065891082, epsilon: 0.1
    episode: 227/800, returns: -3.94075969857, epsilon: 0.1
    episode: 228/800, returns: -12.5867060703, epsilon: 0.1
    episode: 229/800, returns: -5.04176236079, epsilon: 0.1
    episode: 230/800, returns: -9.28636422375, epsilon: 0.1
    episode: 231/800, returns: -20.1536505525, epsilon: 0.099
    episode: 232/800, returns: -0.0221507656965, epsilon: 0.098
    episode: 233/800, returns: -0.994828210956, epsilon: 0.097
    episode: 234/800, returns: 4.70908785151, epsilon: 0.096
    episode: 235/800, returns: 16.2094173591, epsilon: 0.095
    episode: 236/800, returns: -9.03347777607, epsilon: 0.094
    episode: 237/800, returns: -26.88725791, epsilon: 0.093
    episode: 238/800, returns: -16.8825022088, epsilon: 0.092
    episode: 239/800, returns: 3.97778878731, epsilon: 0.091
    episode: 240/800, returns: 25.6524846059, epsilon: 0.091
    episode: 241/800, returns: 11.7053067622, epsilon: 0.09
    episode: 242/800, returns: -15.9679718833, epsilon: 0.089
    episode: 243/800, returns: 4.49744068118, epsilon: 0.088
    episode: 244/800, returns: 16.7158180539, epsilon: 0.087
    episode: 245/800, returns: -2.97595772975, epsilon: 0.086
    episode: 246/800, returns: -5.33269320826, epsilon: 0.085
    episode: 247/800, returns: 13.7253496579, epsilon: 0.084
    episode: 248/800, returns: -10.1969012074, epsilon: 0.084
    episode: 249/800, returns: 20.7694731588, epsilon: 0.083
    episode: 250/800, returns: -1.26575811595, epsilon: 0.082
    episode: 251/800, returns: 3.20286211967, epsilon: 0.081
    episode: 252/800, returns: -5.84709650506, epsilon: 0.08
    episode: 253/800, returns: -12.3447325999, epsilon: 0.079
    episode: 254/800, returns: 0.0, epsilon: 0.079
    episode: 255/800, returns: 8.16704459562, epsilon: 0.078
    episode: 256/800, returns: -3.17932409049, epsilon: 0.077
    episode: 257/800, returns: -16.8915699715, epsilon: 0.076
    episode: 258/800, returns: 29.4740336632, epsilon: 0.076
    episode: 259/800, returns: -5.96648208589, epsilon: 0.075
    episode: 260/800, returns: -18.3611808171, epsilon: 0.074
    episode: 261/800, returns: -7.35222613641, epsilon: 0.073
    episode: 262/800, returns: 34.6583977168, epsilon: 0.073
    episode: 263/800, returns: -1.1992840392, epsilon: 0.072
    episode: 264/800, returns: -2.6705511568, epsilon: 0.071
    episode: 265/800, returns: 4.83061936609, epsilon: 0.07
    episode: 266/800, returns: 0.0, epsilon: 0.07
    episode: 267/800, returns: -1.07357306785, epsilon: 0.069
    episode: 268/800, returns: -9.89810108733, epsilon: 0.068
    episode: 269/800, returns: 20.4635184331, epsilon: 0.068
    episode: 270/800, returns: 0.491620654646, epsilon: 0.067
    episode: 271/800, returns: 1.87972857031, epsilon: 0.066
    episode: 272/800, returns: 7.12873058401, epsilon: 0.066
    episode: 273/800, returns: 6.05785167117, epsilon: 0.065
    episode: 274/800, returns: 10.0477071366, epsilon: 0.064
    episode: 275/800, returns: 15.4786299547, epsilon: 0.064
    episode: 276/800, returns: 8.21779695166, epsilon: 0.063
    episode: 277/800, returns: 0.0, epsilon: 0.062
    episode: 278/800, returns: 21.0072642889, epsilon: 0.062
    episode: 279/800, returns: -12.9127390429, epsilon: 0.061
    episode: 280/800, returns: -8.17543789561, epsilon: 0.061
    episode: 281/800, returns: 0.744749100444, epsilon: 0.06
    episode: 282/800, returns: -10.3034387067, epsilon: 0.059
    episode: 283/800, returns: -0.660009235115, epsilon: 0.059
    episode: 284/800, returns: 1.40911825256, epsilon: 0.058
    episode: 285/800, returns: -4.75331114232, epsilon: 0.058
    episode: 286/800, returns: -2.75517652698, epsilon: 0.057
    episode: 287/800, returns: 9.05429166802, epsilon: 0.056
    episode: 288/800, returns: -9.25900857173, epsilon: 0.056
    episode: 289/800, returns: -19.3967414722, epsilon: 0.055
    episode: 290/800, returns: 4.61824921297, epsilon: 0.055
    episode: 291/800, returns: 18.5087120001, epsilon: 0.054
    episode: 292/800, returns: -10.6499305282, epsilon: 0.054
    episode: 293/800, returns: 15.6413765557, epsilon: 0.053
    episode: 294/800, returns: -1.34969325153, epsilon: 0.053
    episode: 295/800, returns: -1.64644769522, epsilon: 0.052
    episode: 296/800, returns: 0.155310959552, epsilon: 0.052
    episode: 297/800, returns: 5.45568510164, epsilon: 0.051
    episode: 298/800, returns: 23.7943046418, epsilon: 0.051
    episode: 299/800, returns: 4.53065995501, epsilon: 0.05
    episode: 300/800, returns: -15.3156964839, epsilon: 0.05
    episode: 301/800, returns: -1.86748916909, epsilon: 0.049
    episode: 302/800, returns: 20.2608991327, epsilon: 0.049
    episode: 303/800, returns: 7.28043446777, epsilon: 0.048
    episode: 304/800, returns: -10.1535007887, epsilon: 0.048
    episode: 305/800, returns: -0.709749163085, epsilon: 0.047
    episode: 306/800, returns: -14.4505807929, epsilon: 0.047
    episode: 307/800, returns: -7.14028762495, epsilon: 0.046
    episode: 308/800, returns: 1.29449838188, epsilon: 0.046
    episode: 309/800, returns: 18.7064952865, epsilon: 0.045
    episode: 310/800, returns: 0.0, epsilon: 0.045
    episode: 311/800, returns: -7.70156438026, epsilon: 0.044
    episode: 312/800, returns: 0.432243397828, epsilon: 0.044
    episode: 313/800, returns: 4.97435534694, epsilon: 0.043
    episode: 314/800, returns: -8.13015594386, epsilon: 0.043
    episode: 315/800, returns: 6.77653125218, epsilon: 0.043
    episode: 316/800, returns: 0.0, epsilon: 0.042
    episode: 317/800, returns: 5.16845680712, epsilon: 0.042
    episode: 318/800, returns: -10.3360164356, epsilon: 0.041
    episode: 319/800, returns: -29.6668289077, epsilon: 0.041
    episode: 320/800, returns: -5.16075584176, epsilon: 0.041
    episode: 321/800, returns: -12.0687942492, epsilon: 0.04
    episode: 322/800, returns: 0.691470802804, epsilon: 0.04
    episode: 323/800, returns: -0.140731805388, epsilon: 0.039
    episode: 324/800, returns: 6.02764396323, epsilon: 0.039
    episode: 325/800, returns: -4.33074971001, epsilon: 0.039
    episode: 326/800, returns: -16.9113510934, epsilon: 0.038
    episode: 327/800, returns: 0.0, epsilon: 0.038
    episode: 328/800, returns: -2.64359484563, epsilon: 0.037
    episode: 329/800, returns: 7.88896907193, epsilon: 0.037
    episode: 330/800, returns: -4.21842633158, epsilon: 0.037
    episode: 331/800, returns: -6.06608444123, epsilon: 0.036
    episode: 332/800, returns: -1.19582761717, epsilon: 0.036
    episode: 333/800, returns: -15.2507040085, epsilon: 0.036
    episode: 334/800, returns: 0.0, epsilon: 0.035
    episode: 335/800, returns: -4.25045884275, epsilon: 0.035
    episode: 336/800, returns: -10.7654920979, epsilon: 0.034
    episode: 337/800, returns: -1.55254485536, epsilon: 0.034
    episode: 338/800, returns: 0.0, epsilon: 0.034
    episode: 339/800, returns: -0.225111715889, epsilon: 0.033
    episode: 340/800, returns: 1.44261036926, epsilon: 0.033
    episode: 341/800, returns: 12.2023183742, epsilon: 0.033
    episode: 342/800, returns: 0.0, epsilon: 0.032
    episode: 343/800, returns: -6.09806260986, epsilon: 0.032
    episode: 344/800, returns: -5.52245940043, epsilon: 0.032
    episode: 345/800, returns: 0.0, epsilon: 0.032
    episode: 346/800, returns: -1.54844695207, epsilon: 0.031
    episode: 347/800, returns: 0.0, epsilon: 0.031
    episode: 348/800, returns: 5.27427612758, epsilon: 0.031
    episode: 349/800, returns: 8.58555826874, epsilon: 0.03
    episode: 350/800, returns: -2.42233359327, epsilon: 0.03
    episode: 351/800, returns: -4.68667971174, epsilon: 0.03
    episode: 352/800, returns: 0.0, epsilon: 0.029
    episode: 353/800, returns: -1.07682650056, epsilon: 0.029
    episode: 354/800, returns: -1.09182903555, epsilon: 0.029
    episode: 355/800, returns: 0.537721486184, epsilon: 0.029
    episode: 356/800, returns: 15.9837026615, epsilon: 0.028
    episode: 357/800, returns: -1.12205190401, epsilon: 0.028
    episode: 358/800, returns: -11.258357011, epsilon: 0.028
    episode: 359/800, returns: 1.80307258287, epsilon: 0.027
    episode: 360/800, returns: 0.0, epsilon: 0.027
    episode: 361/800, returns: 0.0, epsilon: 0.027
    episode: 362/800, returns: 1.52753425065, epsilon: 0.027
    episode: 363/800, returns: -1.55254485536, epsilon: 0.026
    episode: 364/800, returns: -14.5411811742, epsilon: 0.026
    episode: 365/800, returns: -4.70570727816, epsilon: 0.026
    episode: 366/800, returns: 11.4903229909, epsilon: 0.026
    episode: 367/800, returns: 0.0, epsilon: 0.025
    episode: 368/800, returns: -4.58296307999, epsilon: 0.025
    episode: 369/800, returns: 0.0, epsilon: 0.025
    episode: 370/800, returns: 1.82520590275, epsilon: 0.025
    episode: 371/800, returns: -14.2770719903, epsilon: 0.024
    episode: 372/800, returns: 0.0, epsilon: 0.024
    episode: 373/800, returns: -1.16334852483, epsilon: 0.024
    episode: 374/800, returns: 1.13437817821, epsilon: 0.024
    episode: 375/800, returns: 0.0, epsilon: 0.023
    episode: 376/800, returns: 0.0, epsilon: 0.023
    episode: 377/800, returns: 0.0, epsilon: 0.023
    episode: 378/800, returns: -3.49853069653, epsilon: 0.023
    episode: 379/800, returns: 0.0, epsilon: 0.022
    episode: 380/800, returns: -0.16965245335, epsilon: 0.022
    episode: 381/800, returns: 17.9326106919, epsilon: 0.022
    episode: 382/800, returns: 0.0, epsilon: 0.022
    episode: 383/800, returns: 0.0, epsilon: 0.022
    episode: 384/800, returns: 10.2365617104, epsilon: 0.021
    episode: 385/800, returns: 0.853074348711, epsilon: 0.021
    episode: 386/800, returns: 0.0, epsilon: 0.021
    episode: 387/800, returns: 20.4635184331, epsilon: 0.021
    episode: 388/800, returns: 2.32991189027, epsilon: 0.02
    episode: 389/800, returns: 2.62147107387, epsilon: 0.02
    episode: 390/800, returns: -0.263594650005, epsilon: 0.02
    episode: 391/800, returns: -3.97589973258, epsilon: 0.02
    episode: 392/800, returns: 0.537721486184, epsilon: 0.02
    episode: 393/800, returns: -4.53998777275, epsilon: 0.019
    episode: 394/800, returns: -11.8716609109, epsilon: 0.019
    episode: 395/800, returns: 0.812924821532, epsilon: 0.019
    episode: 396/800, returns: 19.9284103755, epsilon: 0.019
    episode: 397/800, returns: 10.0477071366, epsilon: 0.019
    episode: 398/800, returns: 0.691470802804, epsilon: 0.019
    episode: 399/800, returns: -0.768340513084, epsilon: 0.018
    episode: 400/800, returns: 0.0, epsilon: 0.018
    episode: 401/800, returns: 0.0, epsilon: 0.018
    episode: 402/800, returns: 0.0, epsilon: 0.018
    episode: 403/800, returns: 0.0, epsilon: 0.018
    episode: 404/800, returns: 7.31490848725, epsilon: 0.017
    episode: 405/800, returns: 0.195654770884, epsilon: 0.017
    episode: 406/800, returns: 0.0, epsilon: 0.017
    episode: 407/800, returns: 0.0, epsilon: 0.017
    episode: 408/800, returns: 4.48209233317, epsilon: 0.017
    episode: 409/800, returns: 0.0, epsilon: 0.017
    episode: 410/800, returns: -3.23272307439, epsilon: 0.016
    episode: 411/800, returns: 21.9013688356, epsilon: 0.016
    episode: 412/800, returns: 2.13924435716, epsilon: 0.016
    episode: 413/800, returns: 2.32991189027, epsilon: 0.016
    episode: 414/800, returns: 2.83419933869, epsilon: 0.016
    episode: 415/800, returns: 0.0, epsilon: 0.016
    episode: 416/800, returns: 0.0, epsilon: 0.015
    episode: 417/800, returns: 0.0, epsilon: 0.015
    episode: 418/800, returns: 0.0, epsilon: 0.015
    episode: 419/800, returns: 0.0, epsilon: 0.015
    episode: 420/800, returns: 0.744749100444, epsilon: 0.015
    episode: 421/800, returns: 1.52753425065, epsilon: 0.015
    episode: 422/800, returns: 0.0, epsilon: 0.015
    episode: 423/800, returns: 0.200678766416, epsilon: 0.014
    episode: 424/800, returns: 0.0, epsilon: 0.014
    episode: 425/800, returns: -1.65300185136, epsilon: 0.014
    episode: 426/800, returns: 0.0, epsilon: 0.014
    episode: 427/800, returns: 0.0, epsilon: 0.014
    episode: 428/800, returns: 0.0, epsilon: 0.014
    episode: 429/800, returns: 0.0, epsilon: 0.014
    episode: 430/800, returns: 0.480173873487, epsilon: 0.013
    episode: 431/800, returns: 4.48209233317, epsilon: 0.013
    episode: 432/800, returns: 0.0, epsilon: 0.013
    episode: 433/800, returns: -1.59002285868, epsilon: 0.013
    episode: 434/800, returns: 0.0, epsilon: 0.013
    episode: 435/800, returns: 0.0, epsilon: 0.013
    episode: 436/800, returns: 0.0, epsilon: 0.013
    episode: 437/800, returns: 0.0, epsilon: 0.013
    episode: 438/800, returns: 0.0, epsilon: 0.012
    episode: 439/800, returns: 0.0, epsilon: 0.012
    episode: 440/800, returns: 0.0, epsilon: 0.012
    episode: 441/800, returns: 0.0, epsilon: 0.012
    episode: 442/800, returns: -11.0328013516, epsilon: 0.012
    episode: 443/800, returns: 0.0, epsilon: 0.012
    episode: 444/800, returns: 0.812924821532, epsilon: 0.012
    episode: 445/800, returns: 0.0, epsilon: 0.012
    episode: 446/800, returns: 0.0, epsilon: 0.011
    episode: 447/800, returns: 0.0, epsilon: 0.011
    episode: 448/800, returns: 1.87972857031, epsilon: 0.011
    episode: 449/800, returns: 0.0, epsilon: 0.011
    episode: 450/800, returns: -1.82963208485, epsilon: 0.011
    episode: 451/800, returns: 0.0, epsilon: 0.011
    episode: 452/800, returns: 0.0, epsilon: 0.011
    episode: 453/800, returns: 0.0, epsilon: 0.011
    episode: 454/800, returns: -1.65300185136, epsilon: 0.011
    episode: 455/800, returns: -0.830559891615, epsilon: 0.01
    episode: 456/800, returns: 0.0, epsilon: 0.01
    episode: 457/800, returns: 0.69954839281, epsilon: 0.01
    episode: 458/800, returns: 21.9013688356, epsilon: 0.01
    episode: 459/800, returns: -20.3137649648, epsilon: 0.01
    episode: 460/800, returns: 11.4344909234, epsilon: 0.0099
    episode: 461/800, returns: -0.622138385349, epsilon: 0.0098
    episode: 462/800, returns: 0.0, epsilon: 0.0097
    episode: 463/800, returns: -4.91562009419, epsilon: 0.0096
    episode: 464/800, returns: 3.68775644598, epsilon: 0.0095
    episode: 465/800, returns: 0.0, epsilon: 0.0094
    episode: 466/800, returns: 0.0, epsilon: 0.0093
    episode: 467/800, returns: 15.0229001677, epsilon: 0.0092
    episode: 468/800, returns: 0.0, epsilon: 0.0092
    episode: 469/800, returns: 0.72192423169, epsilon: 0.0091
    episode: 470/800, returns: 4.83061936609, epsilon: 0.009
    episode: 471/800, returns: 0.0, epsilon: 0.0089
    episode: 472/800, returns: 0.0, epsilon: 0.0088
    episode: 473/800, returns: 0.0, epsilon: 0.0087
    episode: 474/800, returns: 6.93079548981, epsilon: 0.0086
    episode: 475/800, returns: 0.0, epsilon: 0.0085
    episode: 476/800, returns: 0.0, epsilon: 0.0084
    episode: 477/800, returns: 0.0, epsilon: 0.0084
    episode: 478/800, returns: 0.0, epsilon: 0.0083
    episode: 479/800, returns: 0.0, epsilon: 0.0082
    episode: 480/800, returns: 3.58939834172, epsilon: 0.0081
    episode: 481/800, returns: 0.0, epsilon: 0.008
    episode: 482/800, returns: 0.0, epsilon: 0.008
    episode: 483/800, returns: 0.0, epsilon: 0.0079
    episode: 484/800, returns: 0.0, epsilon: 0.0078
    episode: 485/800, returns: 0.0, epsilon: 0.0077
    episode: 486/800, returns: -1.65300185136, epsilon: 0.0076
    episode: 487/800, returns: 0.0, epsilon: 0.0076
    episode: 488/800, returns: 20.4635184331, epsilon: 0.0075
    episode: 489/800, returns: -1.78080398062, epsilon: 0.0074
    episode: 490/800, returns: 0.69954839281, epsilon: 0.0073
    episode: 491/800, returns: 0.69954839281, epsilon: 0.0073
    episode: 492/800, returns: 0.0, epsilon: 0.0072
    episode: 493/800, returns: 0.0, epsilon: 0.0071
    episode: 494/800, returns: -3.97589973258, epsilon: 0.007
    episode: 495/800, returns: -12.1323260018, epsilon: 0.007
    episode: 496/800, returns: 0.0, epsilon: 0.0069
    episode: 497/800, returns: -0.516917293233, epsilon: 0.0068
    episode: 498/800, returns: 0.0, epsilon: 0.0068
    episode: 499/800, returns: 0.0, epsilon: 0.0067
    episode: 500/800, returns: 0.0, epsilon: 0.0066
    episode: 501/800, returns: 0.0, epsilon: 0.0066
    episode: 502/800, returns: 0.0, epsilon: 0.0065
    episode: 503/800, returns: 0.0, epsilon: 0.0064
    episode: 504/800, returns: 0.0, epsilon: 0.0064
    episode: 505/800, returns: 0.0, epsilon: 0.0063
    episode: 506/800, returns: 0.0, epsilon: 0.0062
    episode: 507/800, returns: 0.0, epsilon: 0.0062
    episode: 508/800, returns: 16.5644456835, epsilon: 0.0061
    episode: 509/800, returns: 10.9042942987, epsilon: 0.0061
    episode: 510/800, returns: -1.65300185136, epsilon: 0.006
    episode: 511/800, returns: 2.40877124979, epsilon: 0.0059
    episode: 512/800, returns: -1.23952513966, epsilon: 0.0059
    episode: 513/800, returns: -2.3525243408, epsilon: 0.0058
    episode: 514/800, returns: 2.42780198663, epsilon: 0.0058
    episode: 515/800, returns: 0.0, epsilon: 0.0057
    episode: 516/800, returns: 0.0, epsilon: 0.0057
    episode: 517/800, returns: 0.0, epsilon: 0.0056
    episode: 518/800, returns: 0.0, epsilon: 0.0055
    episode: 519/800, returns: 0.0, epsilon: 0.0055
    episode: 520/800, returns: -0.140731805388, epsilon: 0.0054
    episode: 521/800, returns: -3.0086780318, epsilon: 0.0054
    episode: 522/800, returns: 0.0, epsilon: 0.0053
    episode: 523/800, returns: 0.0, epsilon: 0.0053
    episode: 524/800, returns: 0.0, epsilon: 0.0052
    episode: 525/800, returns: 10.602006689, epsilon: 0.0052
    episode: 526/800, returns: 0.0, epsilon: 0.0051
    episode: 527/800, returns: 0.0, epsilon: 0.0051
    episode: 528/800, returns: 0.0, epsilon: 0.005
    episode: 529/800, returns: 0.0, epsilon: 0.005
    episode: 530/800, returns: 0.0, epsilon: 0.0049
    episode: 531/800, returns: 0.0, epsilon: 0.0049
    episode: 532/800, returns: 0.0, epsilon: 0.0048
    episode: 533/800, returns: 0.0, epsilon: 0.0048
    episode: 534/800, returns: 45.7434345407, epsilon: 0.0047
    episode: 535/800, returns: 0.0, epsilon: 0.0047
    episode: 536/800, returns: 0.702515458577, epsilon: 0.0046
    episode: 537/800, returns: 0.0, epsilon: 0.0046
    episode: 538/800, returns: 0.0, epsilon: 0.0045
    episode: 539/800, returns: 0.0, epsilon: 0.0045
    episode: 540/800, returns: 0.0, epsilon: 0.0044
    episode: 541/800, returns: 0.0, epsilon: 0.0044
    episode: 542/800, returns: 0.0, epsilon: 0.0044
    episode: 543/800, returns: 0.0, epsilon: 0.0043
    episode: 544/800, returns: -3.49853069653, epsilon: 0.0043
    episode: 545/800, returns: 0.0, epsilon: 0.0042
    episode: 546/800, returns: 0.0, epsilon: 0.0042
    episode: 547/800, returns: 0.0, epsilon: 0.0041
    episode: 548/800, returns: 0.0, epsilon: 0.0041
    episode: 549/800, returns: 22.0495566865, epsilon: 0.0041
    episode: 550/800, returns: 0.0, epsilon: 0.004
    episode: 551/800, returns: 0.0, epsilon: 0.004
    episode: 552/800, returns: 0.0, epsilon: 0.0039
    episode: 553/800, returns: 0.0, epsilon: 0.0039
    episode: 554/800, returns: 0.0, epsilon: 0.0039
    episode: 555/800, returns: -9.6113196152, epsilon: 0.0038
    episode: 556/800, returns: 0.0, epsilon: 0.0038
    episode: 557/800, returns: 10.602006689, epsilon: 0.0037
    episode: 558/800, returns: 0.0, epsilon: 0.0037
    episode: 559/800, returns: 0.0, epsilon: 0.0037
    episode: 560/800, returns: 3.68775644598, epsilon: 0.0036
    episode: 561/800, returns: 0.0, epsilon: 0.0036
    episode: 562/800, returns: 0.0, epsilon: 0.0036
    episode: 563/800, returns: 0.0, epsilon: 0.0035
    episode: 564/800, returns: 0.0, epsilon: 0.0035
    episode: 565/800, returns: 0.0, epsilon: 0.0035
    episode: 566/800, returns: 0.0, epsilon: 0.0034
    episode: 567/800, returns: 0.0, epsilon: 0.0034
    episode: 568/800, returns: 0.0, epsilon: 0.0034
    episode: 569/800, returns: 0.0, epsilon: 0.0033
    episode: 570/800, returns: 0.0, epsilon: 0.0033
    episode: 571/800, returns: 0.0, epsilon: 0.0033
    episode: 572/800, returns: 3.45478248936, epsilon: 0.0032
    episode: 573/800, returns: 0.0, epsilon: 0.0032
    episode: 574/800, returns: 0.0, epsilon: 0.0032
    episode: 575/800, returns: 0.0, epsilon: 0.0031
    episode: 576/800, returns: 0.0, epsilon: 0.0031
    episode: 577/800, returns: 0.0, epsilon: 0.0031
    episode: 578/800, returns: 6.93079548981, epsilon: 0.003
    episode: 579/800, returns: 0.0, epsilon: 0.003
    episode: 580/800, returns: 0.308406566075, epsilon: 0.003
    episode: 581/800, returns: 0.0, epsilon: 0.0029
    episode: 582/800, returns: 0.0, epsilon: 0.0029
    episode: 583/800, returns: 0.0, epsilon: 0.0029
    episode: 584/800, returns: 0.0, epsilon: 0.0029
    episode: 585/800, returns: 0.0, epsilon: 0.0028
    episode: 586/800, returns: 0.0, epsilon: 0.0028
    episode: 587/800, returns: 0.0, epsilon: 0.0028
    episode: 588/800, returns: -12.5649101898, epsilon: 0.0027
    episode: 589/800, returns: 0.0, epsilon: 0.0027
    episode: 590/800, returns: 0.0, epsilon: 0.0027
    episode: 591/800, returns: 0.0, epsilon: 0.0027
    episode: 592/800, returns: 0.0, epsilon: 0.0026
    episode: 593/800, returns: 0.0, epsilon: 0.0026
    episode: 594/800, returns: 0.0, epsilon: 0.0026
    episode: 595/800, returns: 0.0, epsilon: 0.0026
    episode: 596/800, returns: 0.0, epsilon: 0.0025
    episode: 597/800, returns: 0.0, epsilon: 0.0025
    episode: 598/800, returns: 0.0, epsilon: 0.0025
    episode: 599/800, returns: 0.69954839281, epsilon: 0.0025
    episode: 600/800, returns: 0.0, epsilon: 0.0024
    episode: 601/800, returns: 0.0, epsilon: 0.0024
    episode: 602/800, returns: 0.0, epsilon: 0.0024
    episode: 603/800, returns: 0.0, epsilon: 0.0024
    episode: 604/800, returns: 0.0, epsilon: 0.0023
    episode: 605/800, returns: 0.0, epsilon: 0.0023
    episode: 606/800, returns: 0.0, epsilon: 0.0023
    episode: 607/800, returns: 0.0, epsilon: 0.0023
    episode: 608/800, returns: 0.0, epsilon: 0.0022
    episode: 609/800, returns: -14.2770719903, epsilon: 0.0022
    episode: 610/800, returns: 0.0, epsilon: 0.0022
    episode: 611/800, returns: 5.45417884867, epsilon: 0.0022
    episode: 612/800, returns: 0.0, epsilon: 0.0022
    episode: 613/800, returns: 0.0, epsilon: 0.0021
    episode: 614/800, returns: 0.0, epsilon: 0.0021
    episode: 615/800, returns: 0.0, epsilon: 0.0021
    episode: 616/800, returns: 0.0, epsilon: 0.0021
    episode: 617/800, returns: -1.19132471235, epsilon: 0.002
    episode: 618/800, returns: 0.0, epsilon: 0.002
    episode: 619/800, returns: 0.0, epsilon: 0.002
    episode: 620/800, returns: 0.0, epsilon: 0.002
    episode: 621/800, returns: 0.0, epsilon: 0.002
    episode: 622/800, returns: 0.0, epsilon: 0.0019
    episode: 623/800, returns: 0.0, epsilon: 0.0019
    episode: 624/800, returns: 0.0, epsilon: 0.0019
    episode: 625/800, returns: 0.0, epsilon: 0.0019
    episode: 626/800, returns: -9.6113196152, epsilon: 0.0019
    episode: 627/800, returns: 0.0, epsilon: 0.0019
    episode: 628/800, returns: 0.0, epsilon: 0.0018
    episode: 629/800, returns: 0.0, epsilon: 0.0018
    episode: 630/800, returns: 0.0, epsilon: 0.0018
    episode: 631/800, returns: 20.4635184331, epsilon: 0.0018
    episode: 632/800, returns: 0.0, epsilon: 0.0018
    episode: 633/800, returns: 0.0, epsilon: 0.0017
    episode: 634/800, returns: 0.0, epsilon: 0.0017
    episode: 635/800, returns: 0.0, epsilon: 0.0017
    episode: 636/800, returns: -4.51291477218, epsilon: 0.0017
    episode: 637/800, returns: 0.0, epsilon: 0.0017
    episode: 638/800, returns: 0.0, epsilon: 0.0017
    episode: 639/800, returns: 0.0, epsilon: 0.0016
    episode: 640/800, returns: 0.0, epsilon: 0.0016
    episode: 641/800, returns: -8.87876862247, epsilon: 0.0016
    episode: 642/800, returns: 0.0, epsilon: 0.0016
    episode: 643/800, returns: 0.0, epsilon: 0.0016
    episode: 644/800, returns: 0.0, epsilon: 0.0016
    episode: 645/800, returns: 0.0, epsilon: 0.0015
    episode: 646/800, returns: 0.0, epsilon: 0.0015
    episode: 647/800, returns: 0.0, epsilon: 0.0015
    episode: 648/800, returns: 0.0, epsilon: 0.0015
    episode: 649/800, returns: 0.0, epsilon: 0.0015
    episode: 650/800, returns: 0.0, epsilon: 0.0015
    episode: 651/800, returns: 0.0, epsilon: 0.0015
    episode: 652/800, returns: 0.0, epsilon: 0.0014
    episode: 653/800, returns: 0.0, epsilon: 0.0014
    episode: 654/800, returns: 0.0, epsilon: 0.0014
    episode: 655/800, returns: 0.0, epsilon: 0.0014
    episode: 656/800, returns: 0.0, epsilon: 0.0014
    episode: 657/800, returns: 0.0, epsilon: 0.0014
    episode: 658/800, returns: 0.0, epsilon: 0.0014
    episode: 659/800, returns: 0.0, epsilon: 0.0013
    episode: 660/800, returns: 0.0, epsilon: 0.0013
    episode: 661/800, returns: 0.0, epsilon: 0.0013
    episode: 662/800, returns: 0.0, epsilon: 0.0013
    episode: 663/800, returns: 0.0, epsilon: 0.0013
    episode: 664/800, returns: 0.0, epsilon: 0.0013
    episode: 665/800, returns: 0.0, epsilon: 0.0013
    episode: 666/800, returns: 0.0, epsilon: 0.0013
    episode: 667/800, returns: 0.0, epsilon: 0.0012
    episode: 668/800, returns: 0.0, epsilon: 0.0012
    episode: 669/800, returns: 0.0, epsilon: 0.0012
    episode: 670/800, returns: 0.0, epsilon: 0.0012
    episode: 671/800, returns: 0.0, epsilon: 0.0012
    episode: 672/800, returns: 0.0, epsilon: 0.0012
    episode: 673/800, returns: 0.0, epsilon: 0.0012
    episode: 674/800, returns: 0.0, epsilon: 0.0012
    episode: 675/800, returns: 0.0, epsilon: 0.0011
    episode: 676/800, returns: 0.0, epsilon: 0.0011
    episode: 677/800, returns: 0.0, epsilon: 0.0011
    episode: 678/800, returns: 0.0, epsilon: 0.0011
    episode: 679/800, returns: 0.0, epsilon: 0.0011
    episode: 680/800, returns: 0.0, epsilon: 0.0011
    episode: 681/800, returns: 0.0, epsilon: 0.0011
    episode: 682/800, returns: 0.0, epsilon: 0.0011
    episode: 683/800, returns: 0.0, epsilon: 0.0011
    episode: 684/800, returns: 0.0, epsilon: 0.001
    episode: 685/800, returns: 0.0, epsilon: 0.001
    episode: 686/800, returns: 0.0, epsilon: 0.001
    episode: 687/800, returns: 0.0, epsilon: 0.001
    episode: 688/800, returns: 0.0, epsilon: 0.001
    episode: 689/800, returns: 0.0, epsilon: 0.00099
    episode: 690/800, returns: 0.0, epsilon: 0.00098
    episode: 691/800, returns: 0.0, epsilon: 0.00097
    episode: 692/800, returns: 0.0, epsilon: 0.00096
    episode: 693/800, returns: 0.0, epsilon: 0.00095
    episode: 694/800, returns: 0.0, epsilon: 0.00094
    episode: 695/800, returns: 0.0, epsilon: 0.00094
    episode: 696/800, returns: 0.0, epsilon: 0.00093
    episode: 697/800, returns: 0.0, epsilon: 0.00092
    episode: 698/800, returns: 0.0, epsilon: 0.00091
    episode: 699/800, returns: 0.0, epsilon: 0.0009
    episode: 700/800, returns: 0.0, epsilon: 0.00089
    episode: 701/800, returns: 0.0, epsilon: 0.00088
    episode: 702/800, returns: 0.0, epsilon: 0.00087
    episode: 703/800, returns: 0.0, epsilon: 0.00086
    episode: 704/800, returns: 0.0, epsilon: 0.00085
    episode: 705/800, returns: 0.0, epsilon: 0.00085
    episode: 706/800, returns: 0.0, epsilon: 0.00084
    episode: 707/800, returns: 0.0, epsilon: 0.00083
    episode: 708/800, returns: 0.0, epsilon: 0.00082
    episode: 709/800, returns: 0.0, epsilon: 0.00081
    episode: 710/800, returns: 0.0, epsilon: 0.0008
    episode: 711/800, returns: 0.0, epsilon: 0.0008
    episode: 712/800, returns: 0.0, epsilon: 0.00079
    episode: 713/800, returns: 0.0, epsilon: 0.00078
    episode: 714/800, returns: 0.0, epsilon: 0.00077
    episode: 715/800, returns: 0.0, epsilon: 0.00076
    episode: 716/800, returns: 0.0, epsilon: 0.00076
    episode: 717/800, returns: 0.0, epsilon: 0.00075
    episode: 718/800, returns: 0.0, epsilon: 0.00074
    episode: 719/800, returns: 0.0, epsilon: 0.00073
    episode: 720/800, returns: 0.0, epsilon: 0.00073
    episode: 721/800, returns: 0.0, epsilon: 0.00072
    episode: 722/800, returns: 0.0, epsilon: 0.00071
    episode: 723/800, returns: 0.0, epsilon: 0.00071
    episode: 724/800, returns: 0.0, epsilon: 0.0007
    episode: 725/800, returns: 0.0, epsilon: 0.00069
    episode: 726/800, returns: 0.0, epsilon: 0.00068
    episode: 727/800, returns: 0.0, epsilon: 0.00068
    episode: 728/800, returns: 0.0, epsilon: 0.00067
    episode: 729/800, returns: 0.0, epsilon: 0.00066
    episode: 730/800, returns: 0.0, epsilon: 0.00066
    episode: 731/800, returns: 0.0, epsilon: 0.00065
    episode: 732/800, returns: 0.0, epsilon: 0.00064
    episode: 733/800, returns: 0.0, epsilon: 0.00064
    episode: 734/800, returns: 0.0, epsilon: 0.00063
    episode: 735/800, returns: 0.0, epsilon: 0.00063
    episode: 736/800, returns: 0.0, epsilon: 0.00062
    episode: 737/800, returns: 0.0, epsilon: 0.00061
    episode: 738/800, returns: 0.0, epsilon: 0.00061
    episode: 739/800, returns: 0.0, epsilon: 0.0006
    episode: 740/800, returns: 0.0, epsilon: 0.00059
    episode: 741/800, returns: 0.0, epsilon: 0.00059
    episode: 742/800, returns: 0.0, epsilon: 0.00058
    episode: 743/800, returns: 0.0, epsilon: 0.00058
    episode: 744/800, returns: 0.0, epsilon: 0.00057
    episode: 745/800, returns: 0.0, epsilon: 0.00057
    episode: 746/800, returns: 0.0, epsilon: 0.00056
    episode: 747/800, returns: 0.0, epsilon: 0.00055
    episode: 748/800, returns: 0.0, epsilon: 0.00055
    episode: 749/800, returns: 0.0, epsilon: 0.00054
    episode: 750/800, returns: 0.0, epsilon: 0.00054
    episode: 751/800, returns: 0.0, epsilon: 0.00053
    episode: 752/800, returns: 0.0, epsilon: 0.00053
    episode: 753/800, returns: 0.0, epsilon: 0.00052
    episode: 754/800, returns: 0.0, epsilon: 0.00052
    episode: 755/800, returns: 0.0, epsilon: 0.00051
    episode: 756/800, returns: 0.0, epsilon: 0.00051
    episode: 757/800, returns: 0.0, epsilon: 0.0005
    episode: 758/800, returns: 10.602006689, epsilon: 0.0005
    episode: 759/800, returns: 0.0, epsilon: 0.00049
    episode: 760/800, returns: 0.0, epsilon: 0.00049
    episode: 761/800, returns: 0.0, epsilon: 0.00048
    episode: 762/800, returns: 0.0, epsilon: 0.00048
    episode: 763/800, returns: 0.0, epsilon: 0.00047
    episode: 764/800, returns: 0.0, epsilon: 0.00047
    episode: 765/800, returns: 0.0, epsilon: 0.00046
    episode: 766/800, returns: 0.0, epsilon: 0.00046
    episode: 767/800, returns: 0.0, epsilon: 0.00045
    episode: 768/800, returns: 0.0, epsilon: 0.00045
    episode: 769/800, returns: 0.0, epsilon: 0.00044
    episode: 770/800, returns: 0.0, epsilon: 0.00044
    episode: 771/800, returns: 0.0, epsilon: 0.00044
    episode: 772/800, returns: 0.0, epsilon: 0.00043
    episode: 773/800, returns: 0.0, epsilon: 0.00043
    episode: 774/800, returns: 0.0, epsilon: 0.00042
    episode: 775/800, returns: 0.0, epsilon: 0.00042
    episode: 776/800, returns: 0.0, epsilon: 0.00041
    episode: 777/800, returns: 0.0, epsilon: 0.00041
    episode: 778/800, returns: 0.0, epsilon: 0.00041
    episode: 779/800, returns: 0.0, epsilon: 0.0004
    episode: 780/800, returns: 0.0, epsilon: 0.0004
    episode: 781/800, returns: 0.0, epsilon: 0.00039
    episode: 782/800, returns: 0.0, epsilon: 0.00039
    episode: 783/800, returns: 0.0, epsilon: 0.00039
    episode: 784/800, returns: 0.0, epsilon: 0.00038
    episode: 785/800, returns: 0.0, epsilon: 0.00038
    episode: 786/800, returns: 0.0, epsilon: 0.00037
    episode: 787/800, returns: 0.0, epsilon: 0.00037
    episode: 788/800, returns: 0.0, epsilon: 0.00037
    episode: 789/800, returns: 0.0, epsilon: 0.00036
    episode: 790/800, returns: 0.0, epsilon: 0.00036
    episode: 791/800, returns: 0.0, epsilon: 0.00036
    episode: 792/800, returns: 0.0, epsilon: 0.00035
    episode: 793/800, returns: 0.0, epsilon: 0.00035
    episode: 794/800, returns: 0.0, epsilon: 0.00035
    episode: 795/800, returns: 0.0, epsilon: 0.00034
    episode: 796/800, returns: 0.0, epsilon: 0.00034
    episode: 797/800, returns: 0.0, epsilon: 0.00034
    episode: 798/800, returns: 0.0, epsilon: 0.00033
    episode: 799/800, returns: 0.0, epsilon: 0.00033
    episode: 800/800, returns: 0.0, epsilon: 0.00033



```python
eth_agent.plot_cum_returns()
```


![png](output_17_0.png)



```python
eth_agent.test(epsilon=0)
```

    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    0.0


## Numeraire

### Bench Marks


```python
from v2 import run_benchmarks
```


```python
run_benchmarks.run_bollingerband_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                                      coin_name = "numeraire")
```




    -49.976776590803539




```python
run_benchmarks.run_random_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                               coin_name = "numeraire")
```




    -89.564964635209762




```python
run_benchmarks.run_alwaysbuy_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                                  coin_name = "numeraire")
```




    -69.272467902995714



### DDQN Agent


```python
from v2.ddqn_agent import DDQNAgent
```


```python
nmr_agent = DDQNAgent(recent_k = 150, num_coins_per_order = num_coins_per_order, coin_name = "numeraire",
                     external_states = ["upper_band", "lower_band", "price_over_sma"],
                     internal_states = ["is_holding_coin", "return_since_entry"])
```


```python
#nmr_agent.plot_external_states()
```


```python
nmr_agent.train(num_episodes=800)
```

    episode: 1/800, returns: -89.5649646352, epsilon: 1.0
    episode: 2/800, returns: -72.0984455959, epsilon: 0.99
    episode: 3/800, returns: -79.2565485362, epsilon: 0.98
    episode: 4/800, returns: -79.2565485362, epsilon: 0.97
    episode: 5/800, returns: -87.784960871, epsilon: 0.96
    episode: 6/800, returns: -79.2565485362, epsilon: 0.95
    episode: 7/800, returns: -69.272467903, epsilon: 0.94
    episode: 8/800, returns: -69.272467903, epsilon: 0.93
    episode: 9/800, returns: -75.7868705036, epsilon: 0.92
    episode: 10/800, returns: -69.272467903, epsilon: 0.91
    episode: 11/800, returns: -69.272467903, epsilon: 0.9
    episode: 12/800, returns: -69.272467903, epsilon: 0.9
    episode: 13/800, returns: -89.5649646352, epsilon: 0.89
    episode: 14/800, returns: -69.272467903, epsilon: 0.88
    episode: 15/800, returns: -89.5649646352, epsilon: 0.87
    episode: 16/800, returns: -87.784960871, epsilon: 0.86
    episode: 17/800, returns: -69.272467903, epsilon: 0.85
    episode: 18/800, returns: -81.9959879639, epsilon: 0.84
    episode: 19/800, returns: -79.2565485362, epsilon: 0.83
    episode: 20/800, returns: -75.7868705036, epsilon: 0.83
    episode: 21/800, returns: -75.7868705036, epsilon: 0.82
    episode: 22/800, returns: -75.7868705036, epsilon: 0.81
    episode: 23/800, returns: -74.5149077142, epsilon: 0.8
    episode: 24/800, returns: -69.272467903, epsilon: 0.79
    episode: 25/800, returns: -75.7868705036, epsilon: 0.79
    episode: 26/800, returns: -74.5149077142, epsilon: 0.78
    episode: 27/800, returns: -89.5649646352, epsilon: 0.77
    episode: 28/800, returns: -81.9959879639, epsilon: 0.76
    episode: 29/800, returns: -87.784960871, epsilon: 0.75
    episode: 30/800, returns: -79.7784453624, epsilon: 0.75
    episode: 31/800, returns: -79.2565485362, epsilon: 0.74
    episode: 32/800, returns: -75.7868705036, epsilon: 0.73
    episode: 33/800, returns: -75.7868705036, epsilon: 0.72
    episode: 34/800, returns: -72.6580350343, epsilon: 0.72
    episode: 35/800, returns: -69.272467903, epsilon: 0.71
    episode: 36/800, returns: -69.272467903, epsilon: 0.7
    episode: 37/800, returns: -69.272467903, epsilon: 0.7
    episode: 38/800, returns: -89.5649646352, epsilon: 0.69
    episode: 39/800, returns: -68.4995612752, epsilon: 0.68
    episode: 40/800, returns: -69.272467903, epsilon: 0.68
    episode: 41/800, returns: -60.8932461874, epsilon: 0.67
    episode: 42/800, returns: -75.7868705036, epsilon: 0.66
    episode: 43/800, returns: -68.4995612752, epsilon: 0.66
    episode: 44/800, returns: -89.5649646352, epsilon: 0.65
    episode: 45/800, returns: -81.9959879639, epsilon: 0.64
    episode: 46/800, returns: -74.5149077142, epsilon: 0.64
    episode: 47/800, returns: -75.7868705036, epsilon: 0.63
    episode: 48/800, returns: -68.4995612752, epsilon: 0.62
    episode: 49/800, returns: -81.9959879639, epsilon: 0.62
    episode: 50/800, returns: -74.5149077142, epsilon: 0.61
    episode: 51/800, returns: -74.5149077142, epsilon: 0.61
    episode: 52/800, returns: -87.784960871, epsilon: 0.6
    episode: 53/800, returns: -81.9959879639, epsilon: 0.59
    episode: 54/800, returns: -69.272467903, epsilon: 0.59
    episode: 55/800, returns: -87.784960871, epsilon: 0.58
    episode: 56/800, returns: -81.9959879639, epsilon: 0.58
    episode: 57/800, returns: -87.784960871, epsilon: 0.57
    episode: 58/800, returns: -69.272467903, epsilon: 0.56
    episode: 59/800, returns: -69.272467903, epsilon: 0.56
    episode: 60/800, returns: -75.7868705036, epsilon: 0.55
    episode: 61/800, returns: -74.5149077142, epsilon: 0.55
    episode: 62/800, returns: -69.272467903, epsilon: 0.54
    episode: 63/800, returns: -87.784960871, epsilon: 0.54
    episode: 64/800, returns: -79.7784453624, epsilon: 0.53
    episode: 65/800, returns: -69.272467903, epsilon: 0.53
    episode: 66/800, returns: -74.5149077142, epsilon: 0.52
    episode: 67/800, returns: -75.7868705036, epsilon: 0.52
    episode: 68/800, returns: -69.272467903, epsilon: 0.51
    episode: 69/800, returns: -79.7784453624, epsilon: 0.5
    episode: 70/800, returns: -72.6580350343, epsilon: 0.5
    episode: 71/800, returns: -79.2565485362, epsilon: 0.49
    episode: 72/800, returns: -81.9959879639, epsilon: 0.49
    episode: 73/800, returns: -69.272467903, epsilon: 0.48
    episode: 74/800, returns: -62.4083769634, epsilon: 0.48
    episode: 75/800, returns: -81.9959879639, epsilon: 0.48
    episode: 76/800, returns: -75.7868705036, epsilon: 0.47
    episode: 77/800, returns: -87.784960871, epsilon: 0.47
    episode: 78/800, returns: -79.7784453624, epsilon: 0.46
    episode: 79/800, returns: -75.0521195274, epsilon: 0.46
    episode: 80/800, returns: -53.5775862069, epsilon: 0.45
    episode: 81/800, returns: -74.1105769231, epsilon: 0.45
    episode: 82/800, returns: -69.272467903, epsilon: 0.44
    episode: 83/800, returns: -74.5149077142, epsilon: 0.44
    episode: 84/800, returns: -75.7868705036, epsilon: 0.43
    episode: 85/800, returns: -89.5649646352, epsilon: 0.43
    episode: 86/800, returns: -79.2565485362, epsilon: 0.43
    episode: 87/800, returns: -89.5649646352, epsilon: 0.42
    episode: 88/800, returns: -75.7868705036, epsilon: 0.42
    episode: 89/800, returns: -75.0521195274, epsilon: 0.41
    episode: 90/800, returns: -69.272467903, epsilon: 0.41
    episode: 91/800, returns: -75.0521195274, epsilon: 0.4
    episode: 92/800, returns: -89.5649646352, epsilon: 0.4
    episode: 93/800, returns: -87.784960871, epsilon: 0.4
    episode: 94/800, returns: -75.7868705036, epsilon: 0.39
    episode: 95/800, returns: -62.4083769634, epsilon: 0.39
    episode: 96/800, returns: -74.5149077142, epsilon: 0.38
    episode: 97/800, returns: -75.0231910946, epsilon: 0.38
    episode: 98/800, returns: -87.784960871, epsilon: 0.38
    episode: 99/800, returns: -89.5649646352, epsilon: 0.37
    episode: 100/800, returns: -72.6580350343, epsilon: 0.37
    episode: 101/800, returns: -79.2565485362, epsilon: 0.37
    episode: 102/800, returns: -75.7868705036, epsilon: 0.36
    episode: 103/800, returns: -74.5149077142, epsilon: 0.36
    episode: 104/800, returns: -69.272467903, epsilon: 0.36
    episode: 105/800, returns: -72.6580350343, epsilon: 0.35
    episode: 106/800, returns: -69.272467903, epsilon: 0.35
    episode: 107/800, returns: -62.849258365, epsilon: 0.34
    episode: 108/800, returns: -69.272467903, epsilon: 0.34
    episode: 109/800, returns: -69.272467903, epsilon: 0.34
    episode: 110/800, returns: -89.5649646352, epsilon: 0.33
    episode: 111/800, returns: -87.784960871, epsilon: 0.33
    episode: 112/800, returns: -89.5649646352, epsilon: 0.33
    episode: 113/800, returns: -59.8134328358, epsilon: 0.32
    episode: 114/800, returns: -75.7868705036, epsilon: 0.32
    episode: 115/800, returns: -87.784960871, epsilon: 0.32
    episode: 116/800, returns: -79.7784453624, epsilon: 0.31
    episode: 117/800, returns: -68.4995612752, epsilon: 0.31
    episode: 118/800, returns: -74.1105769231, epsilon: 0.31
    episode: 119/800, returns: -74.1105769231, epsilon: 0.31
    episode: 120/800, returns: -62.4083769634, epsilon: 0.3
    episode: 121/800, returns: -79.2565485362, epsilon: 0.3
    episode: 122/800, returns: -89.5649646352, epsilon: 0.3
    episode: 123/800, returns: -53.396797923, epsilon: 0.29
    episode: 124/800, returns: -87.784960871, epsilon: 0.29
    episode: 125/800, returns: -75.0521195274, epsilon: 0.29
    episode: 126/800, returns: -81.9959879639, epsilon: 0.28
    episode: 127/800, returns: -65.066493675, epsilon: 0.28
    episode: 128/800, returns: -65.066493675, epsilon: 0.28
    episode: 129/800, returns: -75.0231910946, epsilon: 0.28
    episode: 130/800, returns: -79.7784453624, epsilon: 0.27
    episode: 131/800, returns: -43.2859399684, epsilon: 0.27
    episode: 132/800, returns: -75.0521195274, epsilon: 0.27
    episode: 133/800, returns: -62.849258365, epsilon: 0.27
    episode: 134/800, returns: -43.2859399684, epsilon: 0.26
    episode: 135/800, returns: -68.4995612752, epsilon: 0.26
    episode: 136/800, returns: -75.0521195274, epsilon: 0.26
    episode: 137/800, returns: -68.453427065, epsilon: 0.25
    episode: 138/800, returns: -79.2565485362, epsilon: 0.25
    episode: 139/800, returns: -68.4995612752, epsilon: 0.25
    episode: 140/800, returns: -74.5149077142, epsilon: 0.25
    episode: 141/800, returns: -46.9980314961, epsilon: 0.24
    episode: 142/800, returns: -89.5649646352, epsilon: 0.24
    episode: 143/800, returns: -68.4995612752, epsilon: 0.24
    episode: 144/800, returns: -72.0984455959, epsilon: 0.24
    episode: 145/800, returns: -87.784960871, epsilon: 0.24
    episode: 146/800, returns: -89.5649646352, epsilon: 0.23
    episode: 147/800, returns: -74.5149077142, epsilon: 0.23
    episode: 148/800, returns: -87.784960871, epsilon: 0.23
    episode: 149/800, returns: -75.0231910946, epsilon: 0.23
    episode: 150/800, returns: -54.2286442839, epsilon: 0.22
    episode: 151/800, returns: -53.5775862069, epsilon: 0.22
    episode: 152/800, returns: -75.0521195274, epsilon: 0.22
    episode: 153/800, returns: -68.4995612752, epsilon: 0.22
    episode: 154/800, returns: -68.4995612752, epsilon: 0.21
    episode: 155/800, returns: -89.5649646352, epsilon: 0.21
    episode: 156/800, returns: -81.9959879639, epsilon: 0.21
    episode: 157/800, returns: -53.396797923, epsilon: 0.21
    episode: 158/800, returns: -87.784960871, epsilon: 0.21
    episode: 159/800, returns: -79.2565485362, epsilon: 0.2
    episode: 160/800, returns: -38.3161512027, epsilon: 0.2
    episode: 161/800, returns: -89.5649646352, epsilon: 0.2
    episode: 162/800, returns: -81.9959879639, epsilon: 0.2
    episode: 163/800, returns: -75.7868705036, epsilon: 0.2
    episode: 164/800, returns: -53.396797923, epsilon: 0.19
    episode: 165/800, returns: -81.9959879639, epsilon: 0.19
    episode: 166/800, returns: -75.0521195274, epsilon: 0.19
    episode: 167/800, returns: -79.2565485362, epsilon: 0.19
    episode: 168/800, returns: -54.2286442839, epsilon: 0.19
    episode: 169/800, returns: -72.6580350343, epsilon: 0.18
    episode: 170/800, returns: -68.4995612752, epsilon: 0.18
    episode: 171/800, returns: -81.9959879639, epsilon: 0.18
    episode: 172/800, returns: -59.8134328358, epsilon: 0.18
    episode: 173/800, returns: -70.1496674058, epsilon: 0.18
    episode: 174/800, returns: -61.4393125671, epsilon: 0.18
    episode: 175/800, returns: -75.7868705036, epsilon: 0.17
    episode: 176/800, returns: -87.784960871, epsilon: 0.17
    episode: 177/800, returns: -89.5649646352, epsilon: 0.17
    episode: 178/800, returns: -89.5649646352, epsilon: 0.17
    episode: 179/800, returns: -62.849258365, epsilon: 0.17
    episode: 180/800, returns: -70.1496674058, epsilon: 0.17
    episode: 181/800, returns: -69.272467903, epsilon: 0.16
    episode: 182/800, returns: -75.7868705036, epsilon: 0.16
    episode: 183/800, returns: -69.272467903, epsilon: 0.16
    episode: 184/800, returns: -54.2286442839, epsilon: 0.16
    episode: 185/800, returns: -69.6875879538, epsilon: 0.16
    episode: 186/800, returns: -75.0521195274, epsilon: 0.16
    episode: 187/800, returns: -61.4393125671, epsilon: 0.15
    episode: 188/800, returns: -59.8134328358, epsilon: 0.15
    episode: 189/800, returns: -79.2565485362, epsilon: 0.15
    episode: 190/800, returns: -53.5775862069, epsilon: 0.15
    episode: 191/800, returns: -81.9959879639, epsilon: 0.15
    episode: 192/800, returns: -59.8134328358, epsilon: 0.15
    episode: 193/800, returns: -75.0231910946, epsilon: 0.15
    episode: 194/800, returns: -37.2011661808, epsilon: 0.14
    episode: 195/800, returns: -70.1496674058, epsilon: 0.14
    episode: 196/800, returns: -75.0231910946, epsilon: 0.14
    episode: 197/800, returns: -74.1105769231, epsilon: 0.14
    episode: 198/800, returns: -75.0521195274, epsilon: 0.14
    episode: 199/800, returns: -62.849258365, epsilon: 0.14
    episode: 200/800, returns: -75.0231910946, epsilon: 0.14
    episode: 201/800, returns: -81.9959879639, epsilon: 0.13
    episode: 202/800, returns: -74.5149077142, epsilon: 0.13
    episode: 203/800, returns: -62.4083769634, epsilon: 0.13
    episode: 204/800, returns: -74.1105769231, epsilon: 0.13
    episode: 205/800, returns: -89.5649646352, epsilon: 0.13
    episode: 206/800, returns: -70.1496674058, epsilon: 0.13
    episode: 207/800, returns: -65.066493675, epsilon: 0.13
    episode: 208/800, returns: -70.5657283411, epsilon: 0.12
    episode: 209/800, returns: -60.8932461874, epsilon: 0.12
    episode: 210/800, returns: -75.7868705036, epsilon: 0.12
    episode: 211/800, returns: -74.5149077142, epsilon: 0.12
    episode: 212/800, returns: -79.2565485362, epsilon: 0.12
    episode: 213/800, returns: -69.1315563199, epsilon: 0.12
    episode: 214/800, returns: -75.0521195274, epsilon: 0.12
    episode: 215/800, returns: -28.9108910891, epsilon: 0.12
    episode: 216/800, returns: -68.4995612752, epsilon: 0.12
    episode: 217/800, returns: -75.0231910946, epsilon: 0.11
    episode: 218/800, returns: -58.0116959064, epsilon: 0.11
    episode: 219/800, returns: -66.8513388735, epsilon: 0.11
    episode: 220/800, returns: -57.3971518987, epsilon: 0.11
    episode: 221/800, returns: -59.2970521542, epsilon: 0.11
    episode: 222/800, returns: -58.5928489043, epsilon: 0.11
    episode: 223/800, returns: -28.9108910891, epsilon: 0.11
    episode: 224/800, returns: -68.0320569902, epsilon: 0.11
    episode: 225/800, returns: -43.2859399684, epsilon: 0.11
    episode: 226/800, returns: -48.3948251078, epsilon: 0.1
    episode: 227/800, returns: -59.8134328358, epsilon: 0.1
    episode: 228/800, returns: -79.2565485362, epsilon: 0.1
    episode: 229/800, returns: -69.272467903, epsilon: 0.1
    episode: 230/800, returns: -79.2565485362, epsilon: 0.1
    episode: 231/800, returns: -69.7471910112, epsilon: 0.099
    episode: 232/800, returns: -60.807860262, epsilon: 0.098
    episode: 233/800, returns: -53.396797923, epsilon: 0.097
    episode: 234/800, returns: -75.7868705036, epsilon: 0.096
    episode: 235/800, returns: -43.2859399684, epsilon: 0.095
    episode: 236/800, returns: -61.9703389831, epsilon: 0.094
    episode: 237/800, returns: -72.0984455959, epsilon: 0.093
    episode: 238/800, returns: -68.4995612752, epsilon: 0.092
    episode: 239/800, returns: -75.0231910946, epsilon: 0.091
    episode: 240/800, returns: -61.0488245931, epsilon: 0.091
    episode: 241/800, returns: -72.0984455959, epsilon: 0.09
    episode: 242/800, returns: -68.9893463864, epsilon: 0.089
    episode: 243/800, returns: -61.9703389831, epsilon: 0.088
    episode: 244/800, returns: -63.39225017, epsilon: 0.087
    episode: 245/800, returns: -54.2286442839, epsilon: 0.086
    episode: 246/800, returns: -66.8513388735, epsilon: 0.085
    episode: 247/800, returns: -54.2286442839, epsilon: 0.084
    episode: 248/800, returns: -30.6503541533, epsilon: 0.084
    episode: 249/800, returns: -69.272467903, epsilon: 0.083
    episode: 250/800, returns: -61.0488245931, epsilon: 0.082
    episode: 251/800, returns: -61.0488245931, epsilon: 0.081
    episode: 252/800, returns: -65.066493675, epsilon: 0.08
    episode: 253/800, returns: -72.6580350343, epsilon: 0.079
    episode: 254/800, returns: -79.2565485362, epsilon: 0.079
    episode: 255/800, returns: -61.0488245931, epsilon: 0.078
    episode: 256/800, returns: -61.4806866953, epsilon: 0.077
    episode: 257/800, returns: -58.3848531685, epsilon: 0.076
    episode: 258/800, returns: -68.4995612752, epsilon: 0.076
    episode: 259/800, returns: -36.7586611861, epsilon: 0.075
    episode: 260/800, returns: -72.0984455959, epsilon: 0.074
    episode: 261/800, returns: -60.807860262, epsilon: 0.073
    episode: 262/800, returns: -36.7586611861, epsilon: 0.073
    episode: 263/800, returns: -58.3848531685, epsilon: 0.072
    episode: 264/800, returns: -17.9741051028, epsilon: 0.071
    episode: 265/800, returns: -61.0488245931, epsilon: 0.07
    episode: 266/800, returns: -58.8145315488, epsilon: 0.07
    episode: 267/800, returns: -57.0574162679, epsilon: 0.069
    episode: 268/800, returns: -69.7131608549, epsilon: 0.068
    episode: 269/800, returns: -72.6580350343, epsilon: 0.068
    episode: 270/800, returns: -69.974909395, epsilon: 0.067
    episode: 271/800, returns: -61.4393125671, epsilon: 0.066
    episode: 272/800, returns: -63.39225017, epsilon: 0.066
    episode: 273/800, returns: -69.7471910112, epsilon: 0.065
    episode: 274/800, returns: -58.5928489043, epsilon: 0.064
    episode: 275/800, returns: -70.5657283411, epsilon: 0.064
    episode: 276/800, returns: -67.3735231748, epsilon: 0.063
    episode: 277/800, returns: -59.8134328358, epsilon: 0.062
    episode: 278/800, returns: -60.8932461874, epsilon: 0.062
    episode: 279/800, returns: -65.066493675, epsilon: 0.061
    episode: 280/800, returns: -69.7131608549, epsilon: 0.061
    episode: 281/800, returns: 0.0, epsilon: 0.06
    episode: 282/800, returns: -26.8342391304, epsilon: 0.059
    episode: 283/800, returns: -59.8134328358, epsilon: 0.059
    episode: 284/800, returns: -46.9980314961, epsilon: 0.058
    episode: 285/800, returns: -69.7471910112, epsilon: 0.058
    episode: 286/800, returns: -72.0984455959, epsilon: 0.057
    episode: 287/800, returns: -61.4393125671, epsilon: 0.056
    episode: 288/800, returns: -79.2565485362, epsilon: 0.056
    episode: 289/800, returns: -69.0961262554, epsilon: 0.055
    episode: 290/800, returns: -36.6470588235, epsilon: 0.055
    episode: 291/800, returns: -6.75324675325, epsilon: 0.054
    episode: 292/800, returns: -24.6853146853, epsilon: 0.054
    episode: 293/800, returns: -30.6056701031, epsilon: 0.053
    episode: 294/800, returns: -58.0116959064, epsilon: 0.053
    episode: 295/800, returns: 0.0, epsilon: 0.052
    episode: 296/800, returns: -81.9959879639, epsilon: 0.052
    episode: 297/800, returns: -43.2859399684, epsilon: 0.051
    episode: 298/800, returns: -68.9893463864, epsilon: 0.051
    episode: 299/800, returns: -72.6580350343, epsilon: 0.05
    episode: 300/800, returns: -53.2552083333, epsilon: 0.05
    episode: 301/800, returns: -75.7868705036, epsilon: 0.049
    episode: 302/800, returns: -53.5775862069, epsilon: 0.049
    episode: 303/800, returns: 0.0, epsilon: 0.048
    episode: 304/800, returns: -69.272467903, epsilon: 0.048
    episode: 305/800, returns: 0.0, epsilon: 0.047
    episode: 306/800, returns: -38.3161512027, epsilon: 0.047
    episode: 307/800, returns: -11.2850082372, epsilon: 0.046
    episode: 308/800, returns: -11.0652353427, epsilon: 0.046
    episode: 309/800, returns: -79.7784453624, epsilon: 0.045
    episode: 310/800, returns: -26.8342391304, epsilon: 0.045
    episode: 311/800, returns: -68.9893463864, epsilon: 0.044
    episode: 312/800, returns: -68.1266646937, epsilon: 0.044
    episode: 313/800, returns: -49.9767765908, epsilon: 0.043
    episode: 314/800, returns: -57.0574162679, epsilon: 0.043
    episode: 315/800, returns: -26.4846416382, epsilon: 0.043
    episode: 316/800, returns: 0.0, epsilon: 0.042
    episode: 317/800, returns: -51.7905102954, epsilon: 0.042
    episode: 318/800, returns: -69.1315563199, epsilon: 0.041
    episode: 319/800, returns: -61.1471861472, epsilon: 0.041
    episode: 320/800, returns: 0.0, epsilon: 0.041
    episode: 321/800, returns: -38.3161512027, epsilon: 0.04
    episode: 322/800, returns: -68.453427065, epsilon: 0.04
    episode: 323/800, returns: 0.0, epsilon: 0.039
    episode: 324/800, returns: -58.3848531685, epsilon: 0.039
    episode: 325/800, returns: -53.5775862069, epsilon: 0.039
    episode: 326/800, returns: -36.6470588235, epsilon: 0.038
    episode: 327/800, returns: 0.0, epsilon: 0.038
    episode: 328/800, returns: -74.6050459797, epsilon: 0.037
    episode: 329/800, returns: 0.0, epsilon: 0.037
    episode: 330/800, returns: 0.0, epsilon: 0.037
    episode: 331/800, returns: 0.0, epsilon: 0.036
    episode: 332/800, returns: -59.2970521542, epsilon: 0.036
    episode: 333/800, returns: -69.974909395, epsilon: 0.036
    episode: 334/800, returns: 0.0, epsilon: 0.035
    episode: 335/800, returns: 0.0, epsilon: 0.035
    episode: 336/800, returns: -68.453427065, epsilon: 0.034
    episode: 337/800, returns: -11.1386138614, epsilon: 0.034
    episode: 338/800, returns: 0.0, epsilon: 0.034
    episode: 339/800, returns: -58.0116959064, epsilon: 0.033
    episode: 340/800, returns: -21.5586307356, epsilon: 0.033
    episode: 341/800, returns: -46.9980314961, epsilon: 0.033
    episode: 342/800, returns: -30.8729139923, epsilon: 0.032
    episode: 343/800, returns: -69.974909395, epsilon: 0.032
    episode: 344/800, returns: -22.1820809249, epsilon: 0.032
    episode: 345/800, returns: 0.0, epsilon: 0.032
    episode: 346/800, returns: -54.2286442839, epsilon: 0.031
    episode: 347/800, returns: -61.1471861472, epsilon: 0.031
    episode: 348/800, returns: 0.0, epsilon: 0.031
    episode: 349/800, returns: -51.7905102954, epsilon: 0.03
    episode: 350/800, returns: 0.0, epsilon: 0.03
    episode: 351/800, returns: -26.4846416382, epsilon: 0.03
    episode: 352/800, returns: -70.1496674058, epsilon: 0.029
    episode: 353/800, returns: -63.39225017, epsilon: 0.029
    episode: 354/800, returns: 0.0, epsilon: 0.029
    episode: 355/800, returns: -36.7215041128, epsilon: 0.029
    episode: 356/800, returns: -17.1538461538, epsilon: 0.028
    episode: 357/800, returns: -72.6580350343, epsilon: 0.028
    episode: 358/800, returns: -30.1103179753, epsilon: 0.028
    episode: 359/800, returns: -61.4806866953, epsilon: 0.027
    episode: 360/800, returns: -58.0116959064, epsilon: 0.027
    episode: 361/800, returns: 0.0, epsilon: 0.027
    episode: 362/800, returns: -26.883910387, epsilon: 0.027
    episode: 363/800, returns: 0.0, epsilon: 0.026
    episode: 364/800, returns: -38.3161512027, epsilon: 0.026
    episode: 365/800, returns: -26.7346938776, epsilon: 0.026
    episode: 366/800, returns: -58.0116959064, epsilon: 0.026
    episode: 367/800, returns: 0.0, epsilon: 0.025
    episode: 368/800, returns: 0.0, epsilon: 0.025
    episode: 369/800, returns: -20.399113082, epsilon: 0.025
    episode: 370/800, returns: -43.2859399684, epsilon: 0.025
    episode: 371/800, returns: -81.9959879639, epsilon: 0.024
    episode: 372/800, returns: -30.6056701031, epsilon: 0.024
    episode: 373/800, returns: -57.0574162679, epsilon: 0.024
    episode: 374/800, returns: 0.0, epsilon: 0.024
    episode: 375/800, returns: -72.0984455959, epsilon: 0.023
    episode: 376/800, returns: -74.6050459797, epsilon: 0.023
    episode: 377/800, returns: -81.9959879639, epsilon: 0.023
    episode: 378/800, returns: -61.4806866953, epsilon: 0.023
    episode: 379/800, returns: -24.6853146853, epsilon: 0.022
    episode: 380/800, returns: -43.8184663537, epsilon: 0.022
    episode: 381/800, returns: -58.3848531685, epsilon: 0.022
    episode: 382/800, returns: -70.7336956522, epsilon: 0.022
    episode: 383/800, returns: -69.0961262554, epsilon: 0.022
    episode: 384/800, returns: -65.066493675, epsilon: 0.021
    episode: 385/800, returns: -11.2850082372, epsilon: 0.021
    episode: 386/800, returns: -68.0320569902, epsilon: 0.021
    episode: 387/800, returns: -67.3735231748, epsilon: 0.021
    episode: 388/800, returns: -61.4393125671, epsilon: 0.02
    episode: 389/800, returns: 0.0, epsilon: 0.02
    episode: 390/800, returns: 0.0, epsilon: 0.02
    episode: 391/800, returns: -61.4393125671, epsilon: 0.02
    episode: 392/800, returns: -36.7586611861, epsilon: 0.02
    episode: 393/800, returns: -22.1820809249, epsilon: 0.019
    episode: 394/800, returns: 0.0, epsilon: 0.019
    episode: 395/800, returns: -68.0320569902, epsilon: 0.019
    episode: 396/800, returns: 0.0, epsilon: 0.019
    episode: 397/800, returns: -69.7131608549, epsilon: 0.019
    episode: 398/800, returns: -8.88324873096, epsilon: 0.019
    episode: 399/800, returns: -30.6056701031, epsilon: 0.018
    episode: 400/800, returns: 0.0, epsilon: 0.018
    episode: 401/800, returns: -38.3161512027, epsilon: 0.018
    episode: 402/800, returns: -69.0071942446, epsilon: 0.018
    episode: 403/800, returns: 0.0, epsilon: 0.018
    episode: 404/800, returns: 0.0, epsilon: 0.017
    episode: 405/800, returns: 0.0, epsilon: 0.017
    episode: 406/800, returns: 0.0, epsilon: 0.017
    episode: 407/800, returns: 0.0, epsilon: 0.017
    episode: 408/800, returns: 0.0, epsilon: 0.017
    episode: 409/800, returns: 0.0, epsilon: 0.017
    episode: 410/800, returns: 0.0, epsilon: 0.016
    episode: 411/800, returns: -5.93886462882, epsilon: 0.016
    episode: 412/800, returns: 0.0, epsilon: 0.016
    episode: 413/800, returns: 0.0, epsilon: 0.016
    episode: 414/800, returns: 0.0, epsilon: 0.016
    episode: 415/800, returns: 0.0, epsilon: 0.016
    episode: 416/800, returns: 0.0, epsilon: 0.015
    episode: 417/800, returns: 0.0, epsilon: 0.015
    episode: 418/800, returns: 0.0, epsilon: 0.015
    episode: 419/800, returns: -70.1496674058, epsilon: 0.015
    episode: 420/800, returns: 0.0, epsilon: 0.015
    episode: 421/800, returns: -68.4995612752, epsilon: 0.015
    episode: 422/800, returns: -79.2565485362, epsilon: 0.015
    episode: 423/800, returns: 0.0, epsilon: 0.014
    episode: 424/800, returns: 0.0, epsilon: 0.014
    episode: 425/800, returns: -69.272467903, epsilon: 0.014
    episode: 426/800, returns: -38.3161512027, epsilon: 0.014
    episode: 427/800, returns: -30.1103179753, epsilon: 0.014
    episode: 428/800, returns: 0.0, epsilon: 0.014
    episode: 429/800, returns: 0.0, epsilon: 0.014
    episode: 430/800, returns: 0.0, epsilon: 0.013
    episode: 431/800, returns: 0.0, epsilon: 0.013
    episode: 432/800, returns: 0.0, epsilon: 0.013
    episode: 433/800, returns: -29.9739921977, epsilon: 0.013
    episode: 434/800, returns: 0.0, epsilon: 0.013
    episode: 435/800, returns: 0.0, epsilon: 0.013
    episode: 436/800, returns: 0.0, epsilon: 0.013
    episode: 437/800, returns: -11.0652353427, epsilon: 0.013
    episode: 438/800, returns: -74.6050459797, epsilon: 0.012
    episode: 439/800, returns: 0.0, epsilon: 0.012
    episode: 440/800, returns: 0.0, epsilon: 0.012
    episode: 441/800, returns: 0.0, epsilon: 0.012
    episode: 442/800, returns: 0.0, epsilon: 0.012
    episode: 443/800, returns: -11.1386138614, epsilon: 0.012
    episode: 444/800, returns: 0.0, epsilon: 0.012
    episode: 445/800, returns: -74.5149077142, epsilon: 0.012
    episode: 446/800, returns: -75.0521195274, epsilon: 0.011
    episode: 447/800, returns: -59.8134328358, epsilon: 0.011
    episode: 448/800, returns: 0.0, epsilon: 0.011
    episode: 449/800, returns: 0.0, epsilon: 0.011
    episode: 450/800, returns: -87.784960871, epsilon: 0.011
    episode: 451/800, returns: -28.9108910891, epsilon: 0.011
    episode: 452/800, returns: 0.0, epsilon: 0.011
    episode: 453/800, returns: 0.0, epsilon: 0.011
    episode: 454/800, returns: 0.0, epsilon: 0.011
    episode: 455/800, returns: -70.1827242525, epsilon: 0.01
    episode: 456/800, returns: 0.0, epsilon: 0.01
    episode: 457/800, returns: 0.0, epsilon: 0.01
    episode: 458/800, returns: -26.4846416382, epsilon: 0.01
    episode: 459/800, returns: 0.0, epsilon: 0.01
    episode: 460/800, returns: -17.1538461538, epsilon: 0.0099
    episode: 461/800, returns: 0.0, epsilon: 0.0098
    episode: 462/800, returns: 0.0, epsilon: 0.0097
    episode: 463/800, returns: 0.0, epsilon: 0.0096
    episode: 464/800, returns: -46.9980314961, epsilon: 0.0095
    episode: 465/800, returns: 0.0, epsilon: 0.0094
    episode: 466/800, returns: -69.6705153478, epsilon: 0.0093
    episode: 467/800, returns: 0.0, epsilon: 0.0092
    episode: 468/800, returns: -61.9703389831, epsilon: 0.0092
    episode: 469/800, returns: 0.0, epsilon: 0.0091
    episode: 470/800, returns: -10.1001669449, epsilon: 0.009
    episode: 471/800, returns: -34.9637681159, epsilon: 0.0089
    episode: 472/800, returns: 0.0, epsilon: 0.0088
    episode: 473/800, returns: 0.0, epsilon: 0.0087
    episode: 474/800, returns: 0.0, epsilon: 0.0086
    episode: 475/800, returns: 0.0, epsilon: 0.0085
    episode: 476/800, returns: -26.4846416382, epsilon: 0.0084
    episode: 477/800, returns: 0.0, epsilon: 0.0084
    episode: 478/800, returns: 0.0, epsilon: 0.0083
    episode: 479/800, returns: 0.0, epsilon: 0.0082
    episode: 480/800, returns: -44.0228690229, epsilon: 0.0081
    episode: 481/800, returns: 0.0, epsilon: 0.008
    episode: 482/800, returns: 0.0, epsilon: 0.008
    episode: 483/800, returns: 0.0, epsilon: 0.0079
    episode: 484/800, returns: 0.0, epsilon: 0.0078
    episode: 485/800, returns: 0.0, epsilon: 0.0077
    episode: 486/800, returns: -11.6488925349, epsilon: 0.0076
    episode: 487/800, returns: 0.0, epsilon: 0.0076
    episode: 488/800, returns: -11.2850082372, epsilon: 0.0075
    episode: 489/800, returns: 0.0, epsilon: 0.0074
    episode: 490/800, returns: 0.0, epsilon: 0.0073
    episode: 491/800, returns: 0.0, epsilon: 0.0073
    episode: 492/800, returns: -6.75324675325, epsilon: 0.0072
    episode: 493/800, returns: -58.5928489043, epsilon: 0.0071
    episode: 494/800, returns: 0.0, epsilon: 0.007
    episode: 495/800, returns: 0.0, epsilon: 0.007
    episode: 496/800, returns: 0.0, epsilon: 0.0069
    episode: 497/800, returns: -14.7268408551, epsilon: 0.0068
    episode: 498/800, returns: 0.0, epsilon: 0.0068
    episode: 499/800, returns: 0.0, epsilon: 0.0067
    episode: 500/800, returns: -59.8134328358, epsilon: 0.0066
    episode: 501/800, returns: 0.0, epsilon: 0.0066
    episode: 502/800, returns: 0.0, epsilon: 0.0065
    episode: 503/800, returns: 0.0, epsilon: 0.0064
    episode: 504/800, returns: 0.0, epsilon: 0.0064
    episode: 505/800, returns: 0.0, epsilon: 0.0063
    episode: 506/800, returns: -75.0231910946, epsilon: 0.0062
    episode: 507/800, returns: 0.0, epsilon: 0.0062
    episode: 508/800, returns: 0.0, epsilon: 0.0061
    episode: 509/800, returns: 0.0, epsilon: 0.0061
    episode: 510/800, returns: 0.0, epsilon: 0.006
    episode: 511/800, returns: 0.0, epsilon: 0.0059
    episode: 512/800, returns: 0.0, epsilon: 0.0059
    episode: 513/800, returns: 0.0, epsilon: 0.0058
    episode: 514/800, returns: -22.6293103448, epsilon: 0.0058
    episode: 515/800, returns: 0.0, epsilon: 0.0057
    episode: 516/800, returns: 0.0, epsilon: 0.0057
    episode: 517/800, returns: 0.0, epsilon: 0.0056
    episode: 518/800, returns: 0.0, epsilon: 0.0055
    episode: 519/800, returns: 0.0, epsilon: 0.0055
    episode: 520/800, returns: 0.0, epsilon: 0.0054
    episode: 521/800, returns: -58.0116959064, epsilon: 0.0054
    episode: 522/800, returns: 0.0, epsilon: 0.0053
    episode: 523/800, returns: 0.0, epsilon: 0.0053
    episode: 524/800, returns: 0.0, epsilon: 0.0052
    episode: 525/800, returns: 0.0, epsilon: 0.0052
    episode: 526/800, returns: 0.0, epsilon: 0.0051
    episode: 527/800, returns: -79.2565485362, epsilon: 0.0051
    episode: 528/800, returns: 0.0, epsilon: 0.005
    episode: 529/800, returns: 0.0, epsilon: 0.005
    episode: 530/800, returns: 0.0, epsilon: 0.0049
    episode: 531/800, returns: 0.0, epsilon: 0.0049
    episode: 532/800, returns: -43.8184663537, epsilon: 0.0048
    episode: 533/800, returns: 0.0, epsilon: 0.0048
    episode: 534/800, returns: -23.7252124646, epsilon: 0.0047
    episode: 535/800, returns: -74.6050459797, epsilon: 0.0047
    episode: 536/800, returns: 0.0, epsilon: 0.0046
    episode: 537/800, returns: 0.0, epsilon: 0.0046
    episode: 538/800, returns: 0.0, epsilon: 0.0045
    episode: 539/800, returns: 0.0, epsilon: 0.0045
    episode: 540/800, returns: -57.3971518987, epsilon: 0.0044
    episode: 541/800, returns: 0.0, epsilon: 0.0044
    episode: 542/800, returns: 0.0, epsilon: 0.0044
    episode: 543/800, returns: 0.0, epsilon: 0.0043
    episode: 544/800, returns: 0.0, epsilon: 0.0043
    episode: 545/800, returns: 0.0, epsilon: 0.0042
    episode: 546/800, returns: 0.0, epsilon: 0.0042
    episode: 547/800, returns: 0.0, epsilon: 0.0041
    episode: 548/800, returns: 0.0, epsilon: 0.0041
    episode: 549/800, returns: 0.0, epsilon: 0.0041
    episode: 550/800, returns: 0.0, epsilon: 0.004
    episode: 551/800, returns: 0.0, epsilon: 0.004
    episode: 552/800, returns: 0.0, epsilon: 0.0039
    episode: 553/800, returns: 0.0, epsilon: 0.0039
    episode: 554/800, returns: 0.0, epsilon: 0.0039
    episode: 555/800, returns: 0.0, epsilon: 0.0038
    episode: 556/800, returns: 0.0, epsilon: 0.0038
    episode: 557/800, returns: -69.0961262554, epsilon: 0.0037
    episode: 558/800, returns: 0.0, epsilon: 0.0037
    episode: 559/800, returns: 0.0, epsilon: 0.0037
    episode: 560/800, returns: 0.0, epsilon: 0.0036
    episode: 561/800, returns: 0.0, epsilon: 0.0036
    episode: 562/800, returns: 0.0, epsilon: 0.0036
    episode: 563/800, returns: 0.0, epsilon: 0.0035
    episode: 564/800, returns: 0.0, epsilon: 0.0035
    episode: 565/800, returns: 0.0, epsilon: 0.0035
    episode: 566/800, returns: 0.0, epsilon: 0.0034
    episode: 567/800, returns: 0.0, epsilon: 0.0034
    episode: 568/800, returns: -8.10580204778, epsilon: 0.0034
    episode: 569/800, returns: 0.0, epsilon: 0.0033
    episode: 570/800, returns: 0.0, epsilon: 0.0033
    episode: 571/800, returns: 0.0, epsilon: 0.0033
    episode: 572/800, returns: 0.0, epsilon: 0.0032
    episode: 573/800, returns: 0.0, epsilon: 0.0032
    episode: 574/800, returns: 0.0, epsilon: 0.0032
    episode: 575/800, returns: 0.0, epsilon: 0.0031
    episode: 576/800, returns: 0.0, epsilon: 0.0031
    episode: 577/800, returns: 0.0, epsilon: 0.0031
    episode: 578/800, returns: 0.0, epsilon: 0.003
    episode: 579/800, returns: 0.0, epsilon: 0.003
    episode: 580/800, returns: 0.0, epsilon: 0.003
    episode: 581/800, returns: 0.0, epsilon: 0.0029
    episode: 582/800, returns: 0.0, epsilon: 0.0029
    episode: 583/800, returns: 0.0, epsilon: 0.0029
    episode: 584/800, returns: 0.0, epsilon: 0.0029
    episode: 585/800, returns: 0.0, epsilon: 0.0028
    episode: 586/800, returns: 0.0, epsilon: 0.0028
    episode: 587/800, returns: 0.0, epsilon: 0.0028
    episode: 588/800, returns: 0.0, epsilon: 0.0027
    episode: 589/800, returns: 0.0, epsilon: 0.0027
    episode: 590/800, returns: 0.0, epsilon: 0.0027
    episode: 591/800, returns: -59.977703456, epsilon: 0.0027
    episode: 592/800, returns: 0.0, epsilon: 0.0026
    episode: 593/800, returns: 0.0, epsilon: 0.0026
    episode: 594/800, returns: -4.60584588131, epsilon: 0.0026
    episode: 595/800, returns: 0.0, epsilon: 0.0026
    episode: 596/800, returns: -69.272467903, epsilon: 0.0025
    episode: 597/800, returns: 0.0, epsilon: 0.0025
    episode: 598/800, returns: 0.0, epsilon: 0.0025
    episode: 599/800, returns: 0.0, epsilon: 0.0025
    episode: 600/800, returns: 0.0, epsilon: 0.0024
    episode: 601/800, returns: 0.0, epsilon: 0.0024
    episode: 602/800, returns: 0.0, epsilon: 0.0024
    episode: 603/800, returns: 0.0, epsilon: 0.0024
    episode: 604/800, returns: 0.0, epsilon: 0.0023
    episode: 605/800, returns: 0.0, epsilon: 0.0023
    episode: 606/800, returns: 0.0, epsilon: 0.0023
    episode: 607/800, returns: 0.0, epsilon: 0.0023
    episode: 608/800, returns: 0.0, epsilon: 0.0022
    episode: 609/800, returns: 0.0, epsilon: 0.0022
    episode: 610/800, returns: 0.0, epsilon: 0.0022
    episode: 611/800, returns: 0.0, epsilon: 0.0022
    episode: 612/800, returns: 0.0, epsilon: 0.0022
    episode: 613/800, returns: 0.0, epsilon: 0.0021
    episode: 614/800, returns: 0.0, epsilon: 0.0021
    episode: 615/800, returns: 0.0, epsilon: 0.0021
    episode: 616/800, returns: 0.0, epsilon: 0.0021
    episode: 617/800, returns: 0.0, epsilon: 0.002
    episode: 618/800, returns: 0.0, epsilon: 0.002
    episode: 619/800, returns: 0.0, epsilon: 0.002
    episode: 620/800, returns: 0.0, epsilon: 0.002
    episode: 621/800, returns: 0.0, epsilon: 0.002
    episode: 622/800, returns: 0.0, epsilon: 0.0019
    episode: 623/800, returns: 0.0, epsilon: 0.0019
    episode: 624/800, returns: 0.0, epsilon: 0.0019
    episode: 625/800, returns: 0.0, epsilon: 0.0019
    episode: 626/800, returns: 0.0, epsilon: 0.0019
    episode: 627/800, returns: 0.0, epsilon: 0.0019
    episode: 628/800, returns: 0.0, epsilon: 0.0018
    episode: 629/800, returns: 0.0, epsilon: 0.0018
    episode: 630/800, returns: 0.0, epsilon: 0.0018
    episode: 631/800, returns: 0.0, epsilon: 0.0018
    episode: 632/800, returns: 0.0, epsilon: 0.0018
    episode: 633/800, returns: 0.0, epsilon: 0.0017
    episode: 634/800, returns: 0.0, epsilon: 0.0017
    episode: 635/800, returns: 0.0, epsilon: 0.0017
    episode: 636/800, returns: 0.0, epsilon: 0.0017
    episode: 637/800, returns: 0.0, epsilon: 0.0017
    episode: 638/800, returns: 0.0, epsilon: 0.0017
    episode: 639/800, returns: 0.0, epsilon: 0.0016
    episode: 640/800, returns: 0.0, epsilon: 0.0016
    episode: 641/800, returns: 0.0, epsilon: 0.0016
    episode: 642/800, returns: 0.0, epsilon: 0.0016
    episode: 643/800, returns: 0.0, epsilon: 0.0016
    episode: 644/800, returns: 0.0, epsilon: 0.0016
    episode: 645/800, returns: 0.0, epsilon: 0.0015
    episode: 646/800, returns: 0.0, epsilon: 0.0015
    episode: 647/800, returns: 0.0, epsilon: 0.0015
    episode: 648/800, returns: 0.0, epsilon: 0.0015
    episode: 649/800, returns: 0.0, epsilon: 0.0015
    episode: 650/800, returns: 0.0, epsilon: 0.0015
    episode: 651/800, returns: 0.0, epsilon: 0.0015
    episode: 652/800, returns: 0.0, epsilon: 0.0014
    episode: 653/800, returns: 0.0, epsilon: 0.0014
    episode: 654/800, returns: -61.4393125671, epsilon: 0.0014
    episode: 655/800, returns: 0.0, epsilon: 0.0014
    episode: 656/800, returns: 0.0, epsilon: 0.0014
    episode: 657/800, returns: 0.0, epsilon: 0.0014
    episode: 658/800, returns: 0.0, epsilon: 0.0014
    episode: 659/800, returns: 0.0, epsilon: 0.0013
    episode: 660/800, returns: 0.0, epsilon: 0.0013
    episode: 661/800, returns: 0.0, epsilon: 0.0013
    episode: 662/800, returns: 0.0, epsilon: 0.0013
    episode: 663/800, returns: 0.0, epsilon: 0.0013
    episode: 664/800, returns: 0.0, epsilon: 0.0013
    episode: 665/800, returns: 0.0, epsilon: 0.0013
    episode: 666/800, returns: 0.0, epsilon: 0.0013
    episode: 667/800, returns: -10.1001669449, epsilon: 0.0012
    episode: 668/800, returns: 0.0, epsilon: 0.0012
    episode: 669/800, returns: 0.0, epsilon: 0.0012
    episode: 670/800, returns: 0.0, epsilon: 0.0012
    episode: 671/800, returns: 0.0, epsilon: 0.0012
    episode: 672/800, returns: 0.0, epsilon: 0.0012
    episode: 673/800, returns: 0.0, epsilon: 0.0012
    episode: 674/800, returns: 0.0, epsilon: 0.0012
    episode: 675/800, returns: 0.0, epsilon: 0.0011
    episode: 676/800, returns: 0.0, epsilon: 0.0011
    episode: 677/800, returns: 0.0, epsilon: 0.0011
    episode: 678/800, returns: 0.0, epsilon: 0.0011
    episode: 679/800, returns: 0.0, epsilon: 0.0011
    episode: 680/800, returns: 0.0, epsilon: 0.0011
    episode: 681/800, returns: 0.0, epsilon: 0.0011
    episode: 682/800, returns: 0.0, epsilon: 0.0011
    episode: 683/800, returns: 0.0, epsilon: 0.0011
    episode: 684/800, returns: 0.0, epsilon: 0.001
    episode: 685/800, returns: 0.0, epsilon: 0.001
    episode: 686/800, returns: 0.0, epsilon: 0.001
    episode: 687/800, returns: 0.0, epsilon: 0.001
    episode: 688/800, returns: 0.0, epsilon: 0.001
    episode: 689/800, returns: 0.0, epsilon: 0.00099
    episode: 690/800, returns: -62.6560332871, epsilon: 0.00099
    episode: 691/800, returns: 0.0, epsilon: 0.00099
    episode: 692/800, returns: 0.0, epsilon: 0.00099
    episode: 693/800, returns: 0.0, epsilon: 0.00099
    episode: 694/800, returns: 0.0, epsilon: 0.00099
    episode: 695/800, returns: 0.0, epsilon: 0.00099
    episode: 696/800, returns: 0.0, epsilon: 0.00099
    episode: 697/800, returns: 0.0, epsilon: 0.00099
    episode: 698/800, returns: 0.0, epsilon: 0.00099
    episode: 699/800, returns: 0.0, epsilon: 0.00099
    episode: 700/800, returns: 0.0, epsilon: 0.00099
    episode: 701/800, returns: 0.0, epsilon: 0.00099
    episode: 702/800, returns: 0.0, epsilon: 0.00099
    episode: 703/800, returns: 0.0, epsilon: 0.00099
    episode: 704/800, returns: 0.0, epsilon: 0.00099
    episode: 705/800, returns: -59.8134328358, epsilon: 0.00099
    episode: 706/800, returns: 0.0, epsilon: 0.00099
    episode: 707/800, returns: 0.0, epsilon: 0.00099
    episode: 708/800, returns: 0.0, epsilon: 0.00099
    episode: 709/800, returns: 0.0, epsilon: 0.00099
    episode: 710/800, returns: 0.0, epsilon: 0.00099
    episode: 711/800, returns: 0.0, epsilon: 0.00099
    episode: 712/800, returns: 0.0, epsilon: 0.00099
    episode: 713/800, returns: 0.0, epsilon: 0.00099
    episode: 714/800, returns: 0.0, epsilon: 0.00099
    episode: 715/800, returns: 0.0, epsilon: 0.00099
    episode: 716/800, returns: -51.7905102954, epsilon: 0.00099
    episode: 717/800, returns: 0.0, epsilon: 0.00099
    episode: 718/800, returns: 0.0, epsilon: 0.00099
    episode: 719/800, returns: 0.0, epsilon: 0.00099
    episode: 720/800, returns: 0.0, epsilon: 0.00099
    episode: 721/800, returns: 0.0, epsilon: 0.00099
    episode: 722/800, returns: 0.0, epsilon: 0.00099
    episode: 723/800, returns: 0.0, epsilon: 0.00099
    episode: 724/800, returns: 0.0, epsilon: 0.00099
    episode: 725/800, returns: 0.0, epsilon: 0.00099
    episode: 726/800, returns: 0.0, epsilon: 0.00099
    episode: 727/800, returns: 0.0, epsilon: 0.00099
    episode: 728/800, returns: 0.0, epsilon: 0.00099
    episode: 729/800, returns: 0.0, epsilon: 0.00099
    episode: 730/800, returns: 0.0, epsilon: 0.00099
    episode: 731/800, returns: 0.0, epsilon: 0.00099
    episode: 732/800, returns: 0.0, epsilon: 0.00099
    episode: 733/800, returns: 0.0, epsilon: 0.00099
    episode: 734/800, returns: 0.0, epsilon: 0.00099
    episode: 735/800, returns: 0.0, epsilon: 0.00099
    episode: 736/800, returns: 0.0, epsilon: 0.00099
    episode: 737/800, returns: 0.0, epsilon: 0.00099
    episode: 738/800, returns: 0.0, epsilon: 0.00099
    episode: 739/800, returns: 0.0, epsilon: 0.00099
    episode: 740/800, returns: 0.0, epsilon: 0.00099
    episode: 741/800, returns: 0.0, epsilon: 0.00099
    episode: 742/800, returns: 0.0, epsilon: 0.00099
    episode: 743/800, returns: 0.0, epsilon: 0.00099
    episode: 744/800, returns: 0.0, epsilon: 0.00099
    episode: 745/800, returns: 0.0, epsilon: 0.00099
    episode: 746/800, returns: 0.0, epsilon: 0.00099
    episode: 747/800, returns: 0.0, epsilon: 0.00099
    episode: 748/800, returns: 0.0, epsilon: 0.00099
    episode: 749/800, returns: 0.0, epsilon: 0.00099
    episode: 750/800, returns: 0.0, epsilon: 0.00099
    episode: 751/800, returns: 0.0, epsilon: 0.00099
    episode: 752/800, returns: 0.0, epsilon: 0.00099
    episode: 753/800, returns: 0.0, epsilon: 0.00099
    episode: 754/800, returns: 0.0, epsilon: 0.00099
    episode: 755/800, returns: -70.7336956522, epsilon: 0.00099
    episode: 756/800, returns: 0.0, epsilon: 0.00099
    episode: 757/800, returns: 0.0, epsilon: 0.00099
    episode: 758/800, returns: 0.0, epsilon: 0.00099
    episode: 759/800, returns: 0.0, epsilon: 0.00099
    episode: 760/800, returns: -63.39225017, epsilon: 0.00099
    episode: 761/800, returns: -69.7131608549, epsilon: 0.00099
    episode: 762/800, returns: 0.0, epsilon: 0.00099
    episode: 763/800, returns: 0.0, epsilon: 0.00099
    episode: 764/800, returns: 0.0, epsilon: 0.00099
    episode: 765/800, returns: 0.0, epsilon: 0.00099
    episode: 766/800, returns: 0.0, epsilon: 0.00099
    episode: 767/800, returns: 0.0, epsilon: 0.00099
    episode: 768/800, returns: 0.0, epsilon: 0.00099
    episode: 769/800, returns: -6.75324675325, epsilon: 0.00099
    episode: 770/800, returns: 0.0, epsilon: 0.00099
    episode: 771/800, returns: 0.0, epsilon: 0.00099
    episode: 772/800, returns: 0.0, epsilon: 0.00099
    episode: 773/800, returns: 0.0, epsilon: 0.00099
    episode: 774/800, returns: 0.0, epsilon: 0.00099
    episode: 775/800, returns: 0.0, epsilon: 0.00099
    episode: 776/800, returns: 0.0, epsilon: 0.00099
    episode: 777/800, returns: -11.6488925349, epsilon: 0.00099
    episode: 778/800, returns: 0.0, epsilon: 0.00099
    episode: 779/800, returns: 0.0, epsilon: 0.00099
    episode: 780/800, returns: 0.0, epsilon: 0.00099
    episode: 781/800, returns: 0.0, epsilon: 0.00099
    episode: 782/800, returns: 0.0, epsilon: 0.00099
    episode: 783/800, returns: 0.0, epsilon: 0.00099
    episode: 784/800, returns: 0.0, epsilon: 0.00099
    episode: 785/800, returns: 0.0, epsilon: 0.00099
    episode: 786/800, returns: 0.0, epsilon: 0.00099
    episode: 787/800, returns: 0.0, epsilon: 0.00099
    episode: 788/800, returns: 0.0, epsilon: 0.00099
    episode: 789/800, returns: 0.0, epsilon: 0.00099
    episode: 790/800, returns: 0.0, epsilon: 0.00099
    episode: 791/800, returns: -5.93886462882, epsilon: 0.00099
    episode: 792/800, returns: 0.0, epsilon: 0.00099
    episode: 793/800, returns: 0.0, epsilon: 0.00099
    episode: 794/800, returns: 0.0, epsilon: 0.00099
    episode: 795/800, returns: 0.0, epsilon: 0.00099
    episode: 796/800, returns: 0.0, epsilon: 0.00099
    episode: 797/800, returns: 0.0, epsilon: 0.00099
    episode: 798/800, returns: 0.0, epsilon: 0.00099
    episode: 799/800, returns: 0.0, epsilon: 0.00099
    episode: 800/800, returns: -46.9980314961, epsilon: 0.00099



```python
nmr_agent.test(epsilon=0)
```

    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    0.0


## Qtum

### Bench Marks


```python
from v2 import run_benchmarks
```


```python
run_benchmarks.run_bollingerband_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                                      coin_name = "qtum")
```




    52.993472090823069




```python
run_benchmarks.run_random_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                               coin_name = "qtum")
```




    186.54765248295743




```python
run_benchmarks.run_alwaysbuy_agent(num_coins_per_order = num_coins_per_order, recent_k=recent_k,
                                  coin_name = "qtum")
```




    -9.154929577464781



### DDQN Agent


```python
from v2.ddqn_agent import DDQNAgent
```


```python
qtum_agent = DDQNAgent(recent_k = 150, num_coins_per_order = num_coins_per_order, coin_name = "qtum",
                      external_states = ["upper_band", "lower_band", "price_over_sma"],
                     internal_states = ["is_holding_coin", "return_since_entry"])
```


```python
#qtum_agent.plot_external_states()
```


```python
qtum_agent.train(num_episodes=800)
```

    episode: 1/800, reward: -31.5733668852, epsilon: 1.0
    episode: 2/800, reward: 110.01974425, epsilon: 0.99
    episode: 3/800, reward: -52.1904746553, epsilon: 0.98
    episode: 4/800, reward: -73.794513504, epsilon: 0.97
    episode: 5/800, reward: -47.5681682785, epsilon: 0.96
    episode: 6/800, reward: -20.353216964, epsilon: 0.95
    episode: 7/800, reward: -48.4484927053, epsilon: 0.94
    episode: 8/800, reward: -61.6082999243, epsilon: 0.93
    episode: 9/800, reward: 19.0555750735, epsilon: 0.92
    episode: 10/800, reward: 44.3755869747, epsilon: 0.91
    episode: 11/800, reward: 25.2159802999, epsilon: 0.9
    episode: 12/800, reward: 77.3696605056, epsilon: 0.9
    episode: 13/800, reward: -1.67109959898, epsilon: 0.89
    episode: 14/800, reward: -13.6345853319, epsilon: 0.88
    episode: 15/800, reward: -20.2692597785, epsilon: 0.87
    episode: 16/800, reward: -53.3073399322, epsilon: 0.86
    episode: 17/800, reward: -54.0132854465, epsilon: 0.85
    episode: 18/800, reward: 42.9995267822, epsilon: 0.84
    episode: 19/800, reward: 17.1613883544, epsilon: 0.83
    episode: 20/800, reward: 39.6380468891, epsilon: 0.83
    episode: 21/800, reward: -41.8263249787, epsilon: 0.82
    episode: 22/800, reward: -17.129443131, epsilon: 0.81
    episode: 23/800, reward: -53.5755064298, epsilon: 0.8
    episode: 24/800, reward: -31.4253818074, epsilon: 0.79
    episode: 25/800, reward: 73.2605337846, epsilon: 0.79
    episode: 26/800, reward: -10.5172967771, epsilon: 0.78
    episode: 27/800, reward: 46.2583161127, epsilon: 0.77
    episode: 28/800, reward: -67.0853777847, epsilon: 0.76
    episode: 29/800, reward: -25.6615316315, epsilon: 0.75
    episode: 30/800, reward: 170.425259791, epsilon: 0.75
    episode: 31/800, reward: 23.5715291256, epsilon: 0.74
    episode: 32/800, reward: -35.4026254738, epsilon: 0.73
    episode: 33/800, reward: -54.1628853452, epsilon: 0.72
    episode: 34/800, reward: -51.3548468104, epsilon: 0.72
    episode: 35/800, reward: -20.4169248861, epsilon: 0.71
    episode: 36/800, reward: -7.7225798987, epsilon: 0.7
    episode: 37/800, reward: -2.92016017006, epsilon: 0.7
    episode: 38/800, reward: -12.6459724037, epsilon: 0.69
    episode: 39/800, reward: -44.8552394726, epsilon: 0.68
    episode: 40/800, reward: 93.7047129946, epsilon: 0.68
    episode: 41/800, reward: -11.7402447171, epsilon: 0.67
    episode: 42/800, reward: -61.5918597695, epsilon: 0.66
    episode: 43/800, reward: -65.1795175074, epsilon: 0.66
    episode: 44/800, reward: -32.1225674399, epsilon: 0.65
    episode: 45/800, reward: -20.0357452596, epsilon: 0.64
    episode: 46/800, reward: 3.68869929349, epsilon: 0.64
    episode: 47/800, reward: -46.6817122421, epsilon: 0.63
    episode: 48/800, reward: 4.27343013151, epsilon: 0.62
    episode: 49/800, reward: -32.3504652344, epsilon: 0.62
    episode: 50/800, reward: 191.561129411, epsilon: 0.61
    episode: 51/800, reward: -6.27951025359, epsilon: 0.61
    episode: 52/800, reward: -7.16106059801, epsilon: 0.6
    episode: 53/800, reward: -60.962370823, epsilon: 0.59
    episode: 54/800, reward: -46.0902443481, epsilon: 0.59
    episode: 55/800, reward: -36.1206355475, epsilon: 0.58
    episode: 56/800, reward: 195.188814559, epsilon: 0.58
    episode: 57/800, reward: 21.467551815, epsilon: 0.57
    episode: 58/800, reward: -23.775236539, epsilon: 0.56
    episode: 59/800, reward: -27.5755457944, epsilon: 0.56
    episode: 60/800, reward: -10.9341696111, epsilon: 0.55
    episode: 61/800, reward: -63.6024599554, epsilon: 0.55
    episode: 62/800, reward: -50.4378233112, epsilon: 0.54
    episode: 63/800, reward: 159.660156796, epsilon: 0.54
    episode: 64/800, reward: 46.9925211959, epsilon: 0.53
    episode: 65/800, reward: 71.0243760282, epsilon: 0.53
    episode: 66/800, reward: -57.327997893, epsilon: 0.52
    episode: 67/800, reward: 129.552330714, epsilon: 0.52
    episode: 68/800, reward: 20.6193196484, epsilon: 0.51
    episode: 69/800, reward: -38.898783995, epsilon: 0.5
    episode: 70/800, reward: 10.3103614774, epsilon: 0.5
    episode: 71/800, reward: -58.7929601458, epsilon: 0.49
    episode: 72/800, reward: 9.13290744931, epsilon: 0.49
    episode: 73/800, reward: 98.1243141512, epsilon: 0.48
    episode: 74/800, reward: -14.4821033891, epsilon: 0.48
    episode: 75/800, reward: -47.97039202, epsilon: 0.48
    episode: 76/800, reward: -3.12840804246, epsilon: 0.47
    episode: 77/800, reward: -28.4580243314, epsilon: 0.47
    episode: 78/800, reward: 7.2236169875, epsilon: 0.46
    episode: 79/800, reward: -56.2104264534, epsilon: 0.46
    episode: 80/800, reward: -65.2886992017, epsilon: 0.45
    episode: 81/800, reward: 21.2016031984, epsilon: 0.45
    episode: 82/800, reward: 76.5836901372, epsilon: 0.44
    episode: 83/800, reward: -48.324552186, epsilon: 0.44
    episode: 84/800, reward: 6.09063363657, epsilon: 0.43
    episode: 85/800, reward: -72.9323478704, epsilon: 0.43
    episode: 86/800, reward: -10.2512487548, epsilon: 0.43
    episode: 87/800, reward: -69.1070895303, epsilon: 0.42
    episode: 88/800, reward: -1.81677009765, epsilon: 0.42
    episode: 89/800, reward: 71.2470453934, epsilon: 0.41
    episode: 90/800, reward: -23.732215887, epsilon: 0.41
    episode: 91/800, reward: 61.2110723679, epsilon: 0.4
    episode: 92/800, reward: 154.279985579, epsilon: 0.4
    episode: 93/800, reward: -28.9630473285, epsilon: 0.4
    episode: 94/800, reward: 79.2280073658, epsilon: 0.39
    episode: 95/800, reward: 84.1936047961, epsilon: 0.39
    episode: 96/800, reward: 8.30951173939, epsilon: 0.38
    episode: 97/800, reward: 10.3675176083, epsilon: 0.38
    episode: 98/800, reward: -76.0208750983, epsilon: 0.38
    episode: 99/800, reward: 20.643891067, epsilon: 0.37
    episode: 100/800, reward: 31.6745551585, epsilon: 0.37
    episode: 101/800, reward: 157.352532342, epsilon: 0.37
    episode: 102/800, reward: 49.1657058244, epsilon: 0.36
    episode: 103/800, reward: 55.7473781167, epsilon: 0.36
    episode: 104/800, reward: 101.93073092, epsilon: 0.36
    episode: 105/800, reward: 82.6667370557, epsilon: 0.35
    episode: 106/800, reward: -1.25887060432, epsilon: 0.35
    episode: 107/800, reward: -51.1423025552, epsilon: 0.34
    episode: 108/800, reward: -56.8903340637, epsilon: 0.34
    episode: 109/800, reward: -28.5256739864, epsilon: 0.34
    episode: 110/800, reward: -65.9655973166, epsilon: 0.33
    episode: 111/800, reward: 115.210561233, epsilon: 0.33
    episode: 112/800, reward: -41.7623407018, epsilon: 0.33
    episode: 113/800, reward: -6.07904563446, epsilon: 0.32
    episode: 114/800, reward: -33.9296572731, epsilon: 0.32
    episode: 115/800, reward: 155.559921716, epsilon: 0.32
    episode: 116/800, reward: 6.66646046778, epsilon: 0.31
    episode: 117/800, reward: 50.6295580164, epsilon: 0.31
    episode: 118/800, reward: 66.3459708557, epsilon: 0.31
    episode: 119/800, reward: 63.0827604462, epsilon: 0.31
    episode: 120/800, reward: 146.513109339, epsilon: 0.3
    episode: 121/800, reward: 7.23265701373, epsilon: 0.3
    episode: 122/800, reward: 52.0417563274, epsilon: 0.3
    episode: 123/800, reward: -26.4782559994, epsilon: 0.29
    episode: 124/800, reward: -9.53841805186, epsilon: 0.29
    episode: 125/800, reward: -9.88561961771, epsilon: 0.29
    episode: 126/800, reward: 70.6537213242, epsilon: 0.28
    episode: 127/800, reward: -58.805291962, epsilon: 0.28
    episode: 128/800, reward: 19.791237101, epsilon: 0.28
    episode: 129/800, reward: -47.2823351277, epsilon: 0.28
    episode: 130/800, reward: 9.04025632705, epsilon: 0.27
    episode: 131/800, reward: 72.0925556921, epsilon: 0.27
    episode: 132/800, reward: -2.76453934511, epsilon: 0.27
    episode: 133/800, reward: -61.6614234968, epsilon: 0.27
    episode: 134/800, reward: 169.419724936, epsilon: 0.26
    episode: 135/800, reward: -47.4437135674, epsilon: 0.26
    episode: 136/800, reward: 38.3552277006, epsilon: 0.26
    episode: 137/800, reward: -58.0251026449, epsilon: 0.25
    episode: 138/800, reward: -16.9015373855, epsilon: 0.25
    episode: 139/800, reward: 19.672247802, epsilon: 0.25
    episode: 140/800, reward: -28.0953512425, epsilon: 0.25
    episode: 141/800, reward: 100.573993576, epsilon: 0.24
    episode: 142/800, reward: 91.257841279, epsilon: 0.24
    episode: 143/800, reward: -46.6487331023, epsilon: 0.24
    episode: 144/800, reward: -1.52556005745, epsilon: 0.24
    episode: 145/800, reward: 16.2640030474, epsilon: 0.24
    episode: 146/800, reward: -46.6700071896, epsilon: 0.23
    episode: 147/800, reward: -58.9923863789, epsilon: 0.23
    episode: 148/800, reward: 41.5801603933, epsilon: 0.23
    episode: 149/800, reward: -2.15435804759, epsilon: 0.23
    episode: 150/800, reward: 74.2358819597, epsilon: 0.22
    episode: 151/800, reward: 47.4481850083, epsilon: 0.22
    episode: 152/800, reward: -10.6311132242, epsilon: 0.22
    episode: 153/800, reward: -60.4824043445, epsilon: 0.22
    episode: 154/800, reward: -41.6490142406, epsilon: 0.21
    episode: 155/800, reward: -37.1527372829, epsilon: 0.21
    episode: 156/800, reward: 8.45521034103, epsilon: 0.21
    episode: 157/800, reward: -19.6570421772, epsilon: 0.21
    episode: 158/800, reward: 124.201393509, epsilon: 0.21
    episode: 159/800, reward: -19.3907750126, epsilon: 0.2
    episode: 160/800, reward: 28.2736300968, epsilon: 0.2
    episode: 161/800, reward: 27.2720664653, epsilon: 0.2
    episode: 162/800, reward: 12.8707251485, epsilon: 0.2
    episode: 163/800, reward: 5.08633402929, epsilon: 0.2
    episode: 164/800, reward: -18.0056412687, epsilon: 0.19
    episode: 165/800, reward: -14.7561902486, epsilon: 0.19
    episode: 166/800, reward: 48.9594134073, epsilon: 0.19
    episode: 167/800, reward: 88.7767530973, epsilon: 0.19
    episode: 168/800, reward: -21.1491850024, epsilon: 0.19
    episode: 169/800, reward: 53.8745970798, epsilon: 0.18
    episode: 170/800, reward: -29.9546173579, epsilon: 0.18
    episode: 171/800, reward: 119.272033772, epsilon: 0.18
    episode: 172/800, reward: 26.3248322031, epsilon: 0.18
    episode: 173/800, reward: -0.626657284368, epsilon: 0.18
    episode: 174/800, reward: -56.0789587133, epsilon: 0.18
    episode: 175/800, reward: -22.8900634259, epsilon: 0.17
    episode: 176/800, reward: -23.0087974436, epsilon: 0.17
    episode: 177/800, reward: -37.7273417834, epsilon: 0.17
    episode: 178/800, reward: -0.82338821629, epsilon: 0.17
    episode: 179/800, reward: 101.82932263, epsilon: 0.17
    episode: 180/800, reward: 21.1305219121, epsilon: 0.17
    episode: 181/800, reward: -4.49986143961, epsilon: 0.16
    episode: 182/800, reward: 76.3444521527, epsilon: 0.16
    episode: 183/800, reward: 65.2808361163, epsilon: 0.16
    episode: 184/800, reward: -37.7954668814, epsilon: 0.16
    episode: 185/800, reward: 2.98247972285, epsilon: 0.16
    episode: 186/800, reward: 7.42015245029, epsilon: 0.16
    episode: 187/800, reward: 67.0282759964, epsilon: 0.15
    episode: 188/800, reward: 93.1349256273, epsilon: 0.15
    episode: 189/800, reward: -28.7523833741, epsilon: 0.15
    episode: 190/800, reward: -12.7529045584, epsilon: 0.15
    episode: 191/800, reward: -2.07885727876, epsilon: 0.15
    episode: 192/800, reward: -67.5186230227, epsilon: 0.15
    episode: 193/800, reward: -5.74377862019, epsilon: 0.15
    episode: 194/800, reward: 52.9896955183, epsilon: 0.14
    episode: 195/800, reward: -54.762994797, epsilon: 0.14
    episode: 196/800, reward: -4.22001772525, epsilon: 0.14
    episode: 197/800, reward: 77.7429525679, epsilon: 0.14
    episode: 198/800, reward: -57.3117021325, epsilon: 0.14
    episode: 199/800, reward: -47.936989838, epsilon: 0.14
    episode: 200/800, reward: -35.7890019709, epsilon: 0.14
    episode: 201/800, reward: 17.7627762133, epsilon: 0.13
    episode: 202/800, reward: -57.1209938274, epsilon: 0.13
    episode: 203/800, reward: 4.05082305777, epsilon: 0.13
    episode: 204/800, reward: 146.97542402, epsilon: 0.13
    episode: 205/800, reward: -55.5653419679, epsilon: 0.13
    episode: 206/800, reward: 60.9005823178, epsilon: 0.13
    episode: 207/800, reward: -3.22276087868, epsilon: 0.13
    episode: 208/800, reward: 86.2099852202, epsilon: 0.12
    episode: 209/800, reward: -35.5373827956, epsilon: 0.12
    episode: 210/800, reward: -6.58949625708, epsilon: 0.12
    episode: 211/800, reward: -2.71548622235, epsilon: 0.12
    episode: 212/800, reward: -71.972290761, epsilon: 0.12
    episode: 213/800, reward: -48.2340375436, epsilon: 0.12
    episode: 214/800, reward: 44.8607243344, epsilon: 0.12
    episode: 215/800, reward: -64.2992380206, epsilon: 0.12
    episode: 216/800, reward: 96.1781527531, epsilon: 0.12
    episode: 217/800, reward: -35.122372927, epsilon: 0.11
    episode: 218/800, reward: -4.67621717095, epsilon: 0.11
    episode: 219/800, reward: 26.4781954887, epsilon: 0.11
    episode: 220/800, reward: 20.2310549602, epsilon: 0.11
    episode: 221/800, reward: -23.6734043795, epsilon: 0.11
    episode: 222/800, reward: -72.9827646437, epsilon: 0.11
    episode: 223/800, reward: -26.5571390728, epsilon: 0.11
    episode: 224/800, reward: 39.3848454636, epsilon: 0.11
    episode: 225/800, reward: -15.4835837124, epsilon: 0.11
    episode: 226/800, reward: 148.80915493, epsilon: 0.1
    episode: 227/800, reward: -26.4819494816, epsilon: 0.1
    episode: 228/800, reward: 86.9606197207, epsilon: 0.1
    episode: 229/800, reward: 5.15913154534, epsilon: 0.1
    episode: 230/800, reward: -3.061181678, epsilon: 0.1
    episode: 231/800, reward: 106.394051223, epsilon: 0.099
    episode: 232/800, reward: -26.8516001449, epsilon: 0.098
    episode: 233/800, reward: 17.8613773237, epsilon: 0.097
    episode: 234/800, reward: 18.608871643, epsilon: 0.096
    episode: 235/800, reward: 31.2537385434, epsilon: 0.095
    episode: 236/800, reward: -55.7679135, epsilon: 0.094
    episode: 237/800, reward: 7.06846528521, epsilon: 0.093
    episode: 238/800, reward: 201.012207189, epsilon: 0.092
    episode: 239/800, reward: -38.2766627865, epsilon: 0.091
    episode: 240/800, reward: -5.20232597314, epsilon: 0.091
    episode: 241/800, reward: -29.3996071618, epsilon: 0.09
    episode: 242/800, reward: -29.8553851658, epsilon: 0.089
    episode: 243/800, reward: 7.89463128707, epsilon: 0.088
    episode: 244/800, reward: -4.07108522875, epsilon: 0.087
    episode: 245/800, reward: -74.4279617626, epsilon: 0.086
    episode: 246/800, reward: 64.4664040145, epsilon: 0.085
    episode: 247/800, reward: -6.66619981703, epsilon: 0.084
    episode: 248/800, reward: 96.8126338768, epsilon: 0.084
    episode: 249/800, reward: -48.4162332491, epsilon: 0.083
    episode: 250/800, reward: -45.69618989, epsilon: 0.082
    episode: 251/800, reward: 14.422310757, epsilon: 0.081
    episode: 252/800, reward: 7.87003891051, epsilon: 0.08
    episode: 253/800, reward: 33.1476190476, epsilon: 0.079
    episode: 254/800, reward: -26.4076432382, epsilon: 0.079
    episode: 255/800, reward: 85.1335882621, epsilon: 0.078
    episode: 256/800, reward: 18.7423436206, epsilon: 0.077
    episode: 257/800, reward: -54.6818096575, epsilon: 0.076
    episode: 258/800, reward: -15.854775758, epsilon: 0.076
    episode: 259/800, reward: 5.01578947368, epsilon: 0.075
    episode: 260/800, reward: -30.8735274505, epsilon: 0.074
    episode: 261/800, reward: -26.9395949395, epsilon: 0.073
    episode: 262/800, reward: 10.1384374336, epsilon: 0.073
    episode: 263/800, reward: -27.2196600417, epsilon: 0.072
    episode: 264/800, reward: 0.0, epsilon: 0.071
    episode: 265/800, reward: 30.3, epsilon: 0.07
    episode: 266/800, reward: 20.4579054449, epsilon: 0.07
    episode: 267/800, reward: -42.0611623637, epsilon: 0.069
    episode: 268/800, reward: 2.29847036329, epsilon: 0.068
    episode: 269/800, reward: -7.47118176009, epsilon: 0.068
    episode: 270/800, reward: 4.82280807269, epsilon: 0.067
    episode: 271/800, reward: -26.1043012471, epsilon: 0.066
    episode: 272/800, reward: -25.0874479026, epsilon: 0.066
    episode: 273/800, reward: -61.2686294789, epsilon: 0.065
    episode: 274/800, reward: 23.6626310772, epsilon: 0.064
    episode: 275/800, reward: 11.8616179002, epsilon: 0.064
    episode: 276/800, reward: 41.5142857143, epsilon: 0.063
    episode: 277/800, reward: 0.0, epsilon: 0.062
    episode: 278/800, reward: -19.292211794, epsilon: 0.062
    episode: 279/800, reward: -0.248550202095, epsilon: 0.061
    episode: 280/800, reward: -39.9119665226, epsilon: 0.061
    episode: 281/800, reward: 36.4, epsilon: 0.06
    episode: 282/800, reward: 30.817311609, epsilon: 0.059
    episode: 283/800, reward: -25.2625588854, epsilon: 0.059
    episode: 284/800, reward: -67.6285759885, epsilon: 0.058
    episode: 285/800, reward: 1.2931820669, epsilon: 0.058
    episode: 286/800, reward: 94.6182926829, epsilon: 0.057
    episode: 287/800, reward: -4.55718604823, epsilon: 0.056
    episode: 288/800, reward: 34.2489013606, epsilon: 0.056
    episode: 289/800, reward: -9.84249870081, epsilon: 0.055
    episode: 290/800, reward: -14.4504211197, epsilon: 0.055
    episode: 291/800, reward: -23.3240976704, epsilon: 0.054
    episode: 292/800, reward: 6.46694283347, epsilon: 0.054
    episode: 293/800, reward: 6.0, epsilon: 0.053
    episode: 294/800, reward: 112.809327098, epsilon: 0.053
    episode: 295/800, reward: 0.0, epsilon: 0.052
    episode: 296/800, reward: -31.8252203116, epsilon: 0.052
    episode: 297/800, reward: 12.5956672444, epsilon: 0.051
    episode: 298/800, reward: -59.5196005452, epsilon: 0.051
    episode: 299/800, reward: -38.9974211949, epsilon: 0.05
    episode: 300/800, reward: -50.5704045834, epsilon: 0.05
    episode: 301/800, reward: -28.6230266205, epsilon: 0.049
    episode: 302/800, reward: -16.0268842564, epsilon: 0.049
    episode: 303/800, reward: -23.285658612, epsilon: 0.048
    episode: 304/800, reward: -20.5865408274, epsilon: 0.048
    episode: 305/800, reward: 10.4091549296, epsilon: 0.047
    episode: 306/800, reward: 7.34977511244, epsilon: 0.047
    episode: 307/800, reward: 56.437886341, epsilon: 0.046
    episode: 308/800, reward: -12.582781457, epsilon: 0.046
    episode: 309/800, reward: 14.5782190377, epsilon: 0.045
    episode: 310/800, reward: -10.4124453834, epsilon: 0.045
    episode: 311/800, reward: -16.0304836416, epsilon: 0.044
    episode: 312/800, reward: 18.3, epsilon: 0.044
    episode: 313/800, reward: 77.8, epsilon: 0.043
    episode: 314/800, reward: -8.14743240459, epsilon: 0.043
    episode: 315/800, reward: 31.9995877988, epsilon: 0.043
    episode: 316/800, reward: 4.23266489239, epsilon: 0.042
    episode: 317/800, reward: 49.1, epsilon: 0.042
    episode: 318/800, reward: 0.0, epsilon: 0.041
    episode: 319/800, reward: -27.3278336417, epsilon: 0.041
    episode: 320/800, reward: -41.8236623964, epsilon: 0.041
    episode: 321/800, reward: -43.9702093443, epsilon: 0.04
    episode: 322/800, reward: 4.52760573597, epsilon: 0.04
    episode: 323/800, reward: 0.0, epsilon: 0.039
    episode: 324/800, reward: 42.8430292599, epsilon: 0.039
    episode: 325/800, reward: -4.83870967742, epsilon: 0.039
    episode: 326/800, reward: 0.0, epsilon: 0.038
    episode: 327/800, reward: -10.5978666667, epsilon: 0.038
    episode: 328/800, reward: 0.0, epsilon: 0.037
    episode: 329/800, reward: -56.9290190465, epsilon: 0.037
    episode: 330/800, reward: 42.6777343366, epsilon: 0.037
    episode: 331/800, reward: 60.7, epsilon: 0.036
    episode: 332/800, reward: 10.3461271676, epsilon: 0.036
    episode: 333/800, reward: -39.7603785351, epsilon: 0.036
    episode: 334/800, reward: -8.81562249302, epsilon: 0.035
    episode: 335/800, reward: 45.8008830886, epsilon: 0.035
    episode: 336/800, reward: -45.1496035949, epsilon: 0.034
    episode: 337/800, reward: -21.0861503352, epsilon: 0.034
    episode: 338/800, reward: -36.1009286718, epsilon: 0.034
    episode: 339/800, reward: 0.0, epsilon: 0.033
    episode: 340/800, reward: 77.8335459184, epsilon: 0.033
    episode: 341/800, reward: -24.7475158375, epsilon: 0.033
    episode: 342/800, reward: 4.91679879729, epsilon: 0.032
    episode: 343/800, reward: 0.0, epsilon: 0.032
    episode: 344/800, reward: 7.26539222418, epsilon: 0.032
    episode: 345/800, reward: 7.14066905857, epsilon: 0.032
    episode: 346/800, reward: 0.0, epsilon: 0.031
    episode: 347/800, reward: 0.0, epsilon: 0.031
    episode: 348/800, reward: 0.0, epsilon: 0.031
    episode: 349/800, reward: 0.0, epsilon: 0.03
    episode: 350/800, reward: 18.8416127627, epsilon: 0.03
    episode: 351/800, reward: 0.0, epsilon: 0.03
    episode: 352/800, reward: -13.3256817297, epsilon: 0.029
    episode: 353/800, reward: 46.2699442566, epsilon: 0.029
    episode: 354/800, reward: 59.3, epsilon: 0.029
    episode: 355/800, reward: 0.0, epsilon: 0.029
    episode: 356/800, reward: 18.3, epsilon: 0.028
    episode: 357/800, reward: 0.0, epsilon: 0.028
    episode: 358/800, reward: -2.6296784831, epsilon: 0.028
    episode: 359/800, reward: -3.1894934334, epsilon: 0.027
    episode: 360/800, reward: -9.46931982634, epsilon: 0.027
    episode: 361/800, reward: 0.0, epsilon: 0.027
    episode: 362/800, reward: -23.2713754647, epsilon: 0.027
    episode: 363/800, reward: -9.47368421053, epsilon: 0.026
    episode: 364/800, reward: 23.6101694915, epsilon: 0.026
    episode: 365/800, reward: -4.6017699115, epsilon: 0.026
    episode: 366/800, reward: 0.0, epsilon: 0.026
    episode: 367/800, reward: 58.0844919786, epsilon: 0.025
    episode: 368/800, reward: -13.567839196, epsilon: 0.025
    episode: 369/800, reward: 27.5, epsilon: 0.025
    episode: 370/800, reward: 0.0, epsilon: 0.025
    episode: 371/800, reward: 0.0, epsilon: 0.024
    episode: 372/800, reward: 0.0, epsilon: 0.024
    episode: 373/800, reward: 0.0, epsilon: 0.024
    episode: 374/800, reward: -9.20692798541, epsilon: 0.024
    episode: 375/800, reward: 0.0, epsilon: 0.023
    episode: 376/800, reward: 24.5418326693, epsilon: 0.023
    episode: 377/800, reward: 0.0, epsilon: 0.023
    episode: 378/800, reward: -44.8420283234, epsilon: 0.023
    episode: 379/800, reward: 31.9912927045, epsilon: 0.022
    episode: 380/800, reward: -11.5681233933, epsilon: 0.022
    episode: 381/800, reward: 0.0, epsilon: 0.022
    episode: 382/800, reward: 9.29824561404, epsilon: 0.022
    episode: 383/800, reward: 0.0, epsilon: 0.022
    episode: 384/800, reward: -26.41353996, epsilon: 0.021
    episode: 385/800, reward: 18.4606181456, epsilon: 0.021
    episode: 386/800, reward: 0.0, epsilon: 0.021
    episode: 387/800, reward: 0.0, epsilon: 0.021
    episode: 388/800, reward: -5.49898167006, epsilon: 0.02
    episode: 389/800, reward: 22.9, epsilon: 0.02
    episode: 390/800, reward: 0.0, epsilon: 0.02
    episode: 391/800, reward: -46.415952065, epsilon: 0.02
    episode: 392/800, reward: 0.0, epsilon: 0.02
    episode: 393/800, reward: -59.7853694494, epsilon: 0.019
    episode: 394/800, reward: -57.8292490119, epsilon: 0.019
    episode: 395/800, reward: 3.7, epsilon: 0.019
    episode: 396/800, reward: -32.1965317919, epsilon: 0.019
    episode: 397/800, reward: 0.0, epsilon: 0.019
    episode: 398/800, reward: -33.7828246984, epsilon: 0.019
    episode: 399/800, reward: 9.4, epsilon: 0.018
    episode: 400/800, reward: 32.1243421053, epsilon: 0.018
    episode: 401/800, reward: -29.6560766349, epsilon: 0.018
    episode: 402/800, reward: 0.0, epsilon: 0.018
    episode: 403/800, reward: 0.0, epsilon: 0.018
    episode: 404/800, reward: -60.2497651235, epsilon: 0.017
    episode: 405/800, reward: -4.60526315789, epsilon: 0.017
    episode: 406/800, reward: 0.0, epsilon: 0.017
    episode: 407/800, reward: -28.5191956124, epsilon: 0.017
    episode: 408/800, reward: 23.1696008188, epsilon: 0.017
    episode: 409/800, reward: 0.0, epsilon: 0.017
    episode: 410/800, reward: 0.0, epsilon: 0.016
    episode: 411/800, reward: 0.0, epsilon: 0.016
    episode: 412/800, reward: -11.0344827586, epsilon: 0.016
    episode: 413/800, reward: -38.1704358273, epsilon: 0.016
    episode: 414/800, reward: 24.5771428571, epsilon: 0.016
    episode: 415/800, reward: 0.0, epsilon: 0.016
    episode: 416/800, reward: 0.0, epsilon: 0.015
    episode: 417/800, reward: -9.15492957746, epsilon: 0.015
    episode: 418/800, reward: 23.1788487321, epsilon: 0.015
    episode: 419/800, reward: -3.86132584984, epsilon: 0.015
    episode: 420/800, reward: -13.4022315712, epsilon: 0.015
    episode: 421/800, reward: -8.09011981947, epsilon: 0.015
    episode: 422/800, reward: -44.2280081766, epsilon: 0.015
    episode: 423/800, reward: -24.4331314439, epsilon: 0.014
    episode: 424/800, reward: -30.6059313178, epsilon: 0.014
    episode: 425/800, reward: -1.73389306243, epsilon: 0.014
    episode: 426/800, reward: 0.0, epsilon: 0.014
    episode: 427/800, reward: 0.0, epsilon: 0.014
    episode: 428/800, reward: 0.0, epsilon: 0.014
    episode: 429/800, reward: -9.94764397906, epsilon: 0.014
    episode: 430/800, reward: -26.1096605744, epsilon: 0.013
    episode: 431/800, reward: 0.0, epsilon: 0.013
    episode: 432/800, reward: 0.0, epsilon: 0.013
    episode: 433/800, reward: -31.925456157, epsilon: 0.013
    episode: 434/800, reward: 5.5, epsilon: 0.013
    episode: 435/800, reward: -26.7810684971, epsilon: 0.013
    episode: 436/800, reward: 22.2, epsilon: 0.013
    episode: 437/800, reward: 0.0, epsilon: 0.013
    episode: 438/800, reward: -26.1096605744, epsilon: 0.012
    episode: 439/800, reward: -33.1202349342, epsilon: 0.012
    episode: 440/800, reward: 7.4485915493, epsilon: 0.012
    episode: 441/800, reward: 11.4525920319, epsilon: 0.012
    episode: 442/800, reward: 70.6, epsilon: 0.012
    episode: 443/800, reward: 26.0, epsilon: 0.012
    episode: 444/800, reward: 0.0, epsilon: 0.012
    episode: 445/800, reward: 0.0, epsilon: 0.012
    episode: 446/800, reward: 56.5, epsilon: 0.011
    episode: 447/800, reward: 11.4525920319, epsilon: 0.011
    episode: 448/800, reward: 11.4525920319, epsilon: 0.011
    episode: 449/800, reward: 0.0, epsilon: 0.011
    episode: 450/800, reward: 0.0, epsilon: 0.011
    episode: 451/800, reward: 8.9016080402, epsilon: 0.011
    episode: 452/800, reward: 0.0, epsilon: 0.011
    episode: 453/800, reward: -23.3062330623, epsilon: 0.011
    episode: 454/800, reward: 0.0, epsilon: 0.011
    episode: 455/800, reward: 0.0, epsilon: 0.01
    episode: 456/800, reward: 0.0, epsilon: 0.01
    episode: 457/800, reward: 0.0, epsilon: 0.01
    episode: 458/800, reward: 0.0, epsilon: 0.01
    episode: 459/800, reward: 100.1, epsilon: 0.01
    episode: 460/800, reward: 0.0, epsilon: 0.0099
    episode: 461/800, reward: -37.5226039783, epsilon: 0.0098
    episode: 462/800, reward: 3.7, epsilon: 0.0097
    episode: 463/800, reward: -14.9216817807, epsilon: 0.0096
    episode: 464/800, reward: 0.0, epsilon: 0.0095
    episode: 465/800, reward: 0.0, epsilon: 0.0094
    episode: 466/800, reward: 0.0, epsilon: 0.0093
    episode: 467/800, reward: 0.0, epsilon: 0.0092
    episode: 468/800, reward: -28.0508474576, epsilon: 0.0092
    episode: 469/800, reward: 0.0, epsilon: 0.0091
    episode: 470/800, reward: 64.1, epsilon: 0.009
    episode: 471/800, reward: 0.0, epsilon: 0.0089
    episode: 472/800, reward: 0.0, epsilon: 0.0088
    episode: 473/800, reward: 0.0, epsilon: 0.0087
    episode: 474/800, reward: -11.8559311906, epsilon: 0.0086
    episode: 475/800, reward: -22.0268680531, epsilon: 0.0085
    episode: 476/800, reward: 0.0, epsilon: 0.0084
    episode: 477/800, reward: -4.26716141002, epsilon: 0.0084
    episode: 478/800, reward: -10.4651162791, epsilon: 0.0083
    episode: 479/800, reward: 0.0, epsilon: 0.0082
    episode: 480/800, reward: 26.2, epsilon: 0.0081
    episode: 481/800, reward: -44.4642857143, epsilon: 0.008
    episode: 482/800, reward: 0.0, epsilon: 0.008
    episode: 483/800, reward: 0.0, epsilon: 0.0079
    episode: 484/800, reward: 0.0, epsilon: 0.0078
    episode: 485/800, reward: -19.0861279364, epsilon: 0.0077
    episode: 486/800, reward: 0.0, epsilon: 0.0076
    episode: 487/800, reward: 2.25303292894, epsilon: 0.0076
    episode: 488/800, reward: 0.0, epsilon: 0.0075
    episode: 489/800, reward: 0.0, epsilon: 0.0074
    episode: 490/800, reward: 0.0, epsilon: 0.0073
    episode: 491/800, reward: 0.0, epsilon: 0.0073
    episode: 492/800, reward: 0.0, epsilon: 0.0072
    episode: 493/800, reward: 22.9, epsilon: 0.0071
    episode: 494/800, reward: 0.0, epsilon: 0.007
    episode: 495/800, reward: 0.0, epsilon: 0.007
    episode: 496/800, reward: 0.0, epsilon: 0.0069
    episode: 497/800, reward: -33.6215177714, epsilon: 0.0068
    episode: 498/800, reward: 46.5, epsilon: 0.0068
    episode: 499/800, reward: 0.0, epsilon: 0.0067
    episode: 500/800, reward: 0.0, epsilon: 0.0066
    episode: 501/800, reward: 0.0, epsilon: 0.0066
    episode: 502/800, reward: 0.0, epsilon: 0.0065
    episode: 503/800, reward: 0.0, epsilon: 0.0064
    episode: 504/800, reward: 0.0, epsilon: 0.0064
    episode: 505/800, reward: 0.0, epsilon: 0.0063
    episode: 506/800, reward: 0.0, epsilon: 0.0062
    episode: 507/800, reward: 0.0, epsilon: 0.0062
    episode: 508/800, reward: -45.8111111111, epsilon: 0.0061
    episode: 509/800, reward: 0.0, epsilon: 0.0061
    episode: 510/800, reward: -65.8350579448, epsilon: 0.006
    episode: 511/800, reward: -68.6080471585, epsilon: 0.0059
    episode: 512/800, reward: 0.0, epsilon: 0.0059
    episode: 513/800, reward: 0.0, epsilon: 0.0058
    episode: 514/800, reward: -8.34813499112, epsilon: 0.0058
    episode: 515/800, reward: 11.4525920319, epsilon: 0.0057
    episode: 516/800, reward: 50.6631494776, epsilon: 0.0057
    episode: 517/800, reward: 11.4525920319, epsilon: 0.0056
    episode: 518/800, reward: -56.8327241209, epsilon: 0.0055
    episode: 519/800, reward: -56.8327241209, epsilon: 0.0055
    episode: 520/800, reward: 6.35259203194, epsilon: 0.0054
    episode: 521/800, reward: 0.0, epsilon: 0.0054
    episode: 522/800, reward: 0.0, epsilon: 0.0053
    episode: 523/800, reward: 0.0, epsilon: 0.0053
    episode: 524/800, reward: -17.4077578051, epsilon: 0.0052
    episode: 525/800, reward: 0.0, epsilon: 0.0052
    episode: 526/800, reward: 0.0, epsilon: 0.0051
    episode: 527/800, reward: 41.8, epsilon: 0.0051
    episode: 528/800, reward: 0.0, epsilon: 0.005
    episode: 529/800, reward: -53.8101604278, epsilon: 0.005
    episode: 530/800, reward: 0.0, epsilon: 0.0049
    episode: 531/800, reward: -14.2857142857, epsilon: 0.0049
    episode: 532/800, reward: 40.1, epsilon: 0.0048
    episode: 533/800, reward: 0.0, epsilon: 0.0048
    episode: 534/800, reward: -11.1876075731, epsilon: 0.0047
    episode: 535/800, reward: 1.87561697927, epsilon: 0.0047
    episode: 536/800, reward: 0.0, epsilon: 0.0046
    episode: 537/800, reward: 0.0, epsilon: 0.0046
    episode: 538/800, reward: 0.0, epsilon: 0.0045
    episode: 539/800, reward: -38.5231316726, epsilon: 0.0045
    episode: 540/800, reward: 0.0, epsilon: 0.0044
    episode: 541/800, reward: 0.0, epsilon: 0.0044
    episode: 542/800, reward: 0.0, epsilon: 0.0044
    episode: 543/800, reward: 0.0, epsilon: 0.0043
    episode: 544/800, reward: 0.0, epsilon: 0.0043
    episode: 545/800, reward: 0.0, epsilon: 0.0042
    episode: 546/800, reward: 63.9, epsilon: 0.0042
    episode: 547/800, reward: 0.0, epsilon: 0.0041
    episode: 548/800, reward: 0.0, epsilon: 0.0041
    episode: 549/800, reward: 0.0, epsilon: 0.0041
    episode: 550/800, reward: 0.0, epsilon: 0.004
    episode: 551/800, reward: 29.8, epsilon: 0.004
    episode: 552/800, reward: 0.0, epsilon: 0.0039
    episode: 553/800, reward: 0.0, epsilon: 0.0039
    episode: 554/800, reward: 0.0, epsilon: 0.0039
    episode: 555/800, reward: 0.0, epsilon: 0.0038
    episode: 556/800, reward: 0.0, epsilon: 0.0038
    episode: 557/800, reward: -28.1337047354, epsilon: 0.0037
    episode: 558/800, reward: 0.0, epsilon: 0.0037
    episode: 559/800, reward: 0.0, epsilon: 0.0037
    episode: 560/800, reward: 0.0, epsilon: 0.0036
    episode: 561/800, reward: 0.0, epsilon: 0.0036
    episode: 562/800, reward: 0.0, epsilon: 0.0036
    episode: 563/800, reward: 0.0, epsilon: 0.0035
    episode: 564/800, reward: 0.0, epsilon: 0.0035
    episode: 565/800, reward: 0.0, epsilon: 0.0035
    episode: 566/800, reward: -12.0204603581, epsilon: 0.0034
    episode: 567/800, reward: -46.6041406666, epsilon: 0.0034
    episode: 568/800, reward: 8.1764288084, epsilon: 0.0034
    episode: 569/800, reward: -46.1146594574, epsilon: 0.0033
    episode: 570/800, reward: 11.4525920319, epsilon: 0.0033
    episode: 571/800, reward: 11.4525920319, epsilon: 0.0033
    episode: 572/800, reward: 11.0485915493, epsilon: 0.0032
    episode: 573/800, reward: -53.3848629377, epsilon: 0.0032
    episode: 574/800, reward: 0.0, epsilon: 0.0032
    episode: 575/800, reward: -16.5229885057, epsilon: 0.0031
    episode: 576/800, reward: 0.0, epsilon: 0.0031
    episode: 577/800, reward: 0.0, epsilon: 0.0031
    episode: 578/800, reward: 0.0, epsilon: 0.003
    episode: 579/800, reward: 0.0, epsilon: 0.003
    episode: 580/800, reward: 0.0, epsilon: 0.003
    episode: 581/800, reward: 0.0, epsilon: 0.0029
    episode: 582/800, reward: 0.0, epsilon: 0.0029
    episode: 583/800, reward: 0.0, epsilon: 0.0029
    episode: 584/800, reward: 63.9, epsilon: 0.0029
    episode: 585/800, reward: 0.0, epsilon: 0.0028
    episode: 586/800, reward: 0.0, epsilon: 0.0028
    episode: 587/800, reward: 0.0, epsilon: 0.0028
    episode: 588/800, reward: 0.0, epsilon: 0.0027
    episode: 589/800, reward: 30.1, epsilon: 0.0027
    episode: 590/800, reward: 0.0, epsilon: 0.0027
    episode: 591/800, reward: 34.1, epsilon: 0.0027
    episode: 592/800, reward: 0.0, epsilon: 0.0026
    episode: 593/800, reward: 0.0, epsilon: 0.0026
    episode: 594/800, reward: 0.0, epsilon: 0.0026
    episode: 595/800, reward: 0.0, epsilon: 0.0026
    episode: 596/800, reward: 0.0, epsilon: 0.0025
    episode: 597/800, reward: 10.8, epsilon: 0.0025
    episode: 598/800, reward: -9.66346803622, epsilon: 0.0025
    episode: 599/800, reward: 0.0, epsilon: 0.0025
    episode: 600/800, reward: 72.3, epsilon: 0.0024
    episode: 601/800, reward: -16.7741935484, epsilon: 0.0024
    episode: 602/800, reward: 0.0, epsilon: 0.0024
    episode: 603/800, reward: 0.0, epsilon: 0.0024
    episode: 604/800, reward: -65.5809859155, epsilon: 0.0023
    episode: 605/800, reward: -45.8267716535, epsilon: 0.0023
    episode: 606/800, reward: 56.5, epsilon: 0.0023
    episode: 607/800, reward: 0.0, epsilon: 0.0023
    episode: 608/800, reward: 0.0, epsilon: 0.0022
    episode: 609/800, reward: 0.0, epsilon: 0.0022
    episode: 610/800, reward: -16.5048543689, epsilon: 0.0022
    episode: 611/800, reward: 0.0, epsilon: 0.0022
    episode: 612/800, reward: 6.5745586642, epsilon: 0.0022
    episode: 613/800, reward: -41.8034388567, epsilon: 0.0021
    episode: 614/800, reward: -14.0484849517, epsilon: 0.0021
    episode: 615/800, reward: -14.0484849517, epsilon: 0.0021
    episode: 616/800, reward: -48.4158194627, epsilon: 0.0021
    episode: 617/800, reward: -45.2919317154, epsilon: 0.002
    episode: 618/800, reward: 11.4525920319, epsilon: 0.002
    episode: 619/800, reward: 11.6911678954, epsilon: 0.002
    episode: 620/800, reward: -33.3850254968, epsilon: 0.002
    episode: 621/800, reward: 0.0, epsilon: 0.002
    episode: 622/800, reward: 0.0, epsilon: 0.0019
    episode: 623/800, reward: 0.0, epsilon: 0.0019
    episode: 624/800, reward: 0.0, epsilon: 0.0019
    episode: 625/800, reward: 0.0, epsilon: 0.0019
    episode: 626/800, reward: 0.0, epsilon: 0.0019
    episode: 627/800, reward: 0.0, epsilon: 0.0019
    episode: 628/800, reward: 0.0, epsilon: 0.0018
    episode: 629/800, reward: 0.0, epsilon: 0.0018
    episode: 630/800, reward: 0.0, epsilon: 0.0018
    episode: 631/800, reward: 1.1701170117, epsilon: 0.0018
    episode: 632/800, reward: 0.0, epsilon: 0.0018
    episode: 633/800, reward: 0.0, epsilon: 0.0017
    episode: 634/800, reward: 0.0, epsilon: 0.0017
    episode: 635/800, reward: 0.0, epsilon: 0.0017
    episode: 636/800, reward: 0.0, epsilon: 0.0017
    episode: 637/800, reward: 0.0, epsilon: 0.0017
    episode: 638/800, reward: 27.7, epsilon: 0.0017
    episode: 639/800, reward: 0.0, epsilon: 0.0016
    episode: 640/800, reward: 0.0, epsilon: 0.0016
    episode: 641/800, reward: 0.0, epsilon: 0.0016
    episode: 642/800, reward: 0.0, epsilon: 0.0016
    episode: 643/800, reward: 0.0, epsilon: 0.0016
    episode: 644/800, reward: 0.0, epsilon: 0.0016
    episode: 645/800, reward: 0.0, epsilon: 0.0015
    episode: 646/800, reward: 0.0, epsilon: 0.0015
    episode: 647/800, reward: 19.7196486526, epsilon: 0.0015
    episode: 648/800, reward: 0.0, epsilon: 0.0015
    episode: 649/800, reward: 0.0, epsilon: 0.0015
    episode: 650/800, reward: 0.0, epsilon: 0.0015
    episode: 651/800, reward: 0.0, epsilon: 0.0015
    episode: 652/800, reward: 45.5, epsilon: 0.0014
    episode: 653/800, reward: 0.0, epsilon: 0.0014
    episode: 654/800, reward: 0.0, epsilon: 0.0014
    episode: 655/800, reward: 0.0, epsilon: 0.0014
    episode: 656/800, reward: 0.0, epsilon: 0.0014
    episode: 657/800, reward: 0.0, epsilon: 0.0014
    episode: 658/800, reward: 0.0, epsilon: 0.0014
    episode: 659/800, reward: 0.0, epsilon: 0.0013
    episode: 660/800, reward: 0.0, epsilon: 0.0013
    episode: 661/800, reward: 0.0, epsilon: 0.0013
    episode: 662/800, reward: 46.7, epsilon: 0.0013
    episode: 663/800, reward: 0.0, epsilon: 0.0013
    episode: 664/800, reward: 0.0, epsilon: 0.0013
    episode: 665/800, reward: 0.0, epsilon: 0.0013
    episode: 666/800, reward: 0.0, epsilon: 0.0013
    episode: 667/800, reward: 0.0, epsilon: 0.0012
    episode: 668/800, reward: 51.4, epsilon: 0.0012
    episode: 669/800, reward: 0.0, epsilon: 0.0012
    episode: 670/800, reward: 0.0, epsilon: 0.0012
    episode: 671/800, reward: 0.0, epsilon: 0.0012
    episode: 672/800, reward: 0.0, epsilon: 0.0012
    episode: 673/800, reward: 0.0, epsilon: 0.0012
    episode: 674/800, reward: 0.0, epsilon: 0.0012
    episode: 675/800, reward: 0.0, epsilon: 0.0011
    episode: 676/800, reward: 0.0, epsilon: 0.0011
    episode: 677/800, reward: 0.0, epsilon: 0.0011
    episode: 678/800, reward: 0.0, epsilon: 0.0011
    episode: 679/800, reward: 0.0, epsilon: 0.0011
    episode: 680/800, reward: -8.44553243574, epsilon: 0.0011
    episode: 681/800, reward: 0.0, epsilon: 0.0011
    episode: 682/800, reward: 0.0, epsilon: 0.0011
    episode: 683/800, reward: 0.0, epsilon: 0.0011
    episode: 684/800, reward: 0.0, epsilon: 0.001
    episode: 685/800, reward: 0.0, epsilon: 0.001
    episode: 686/800, reward: 0.0, epsilon: 0.001
    episode: 687/800, reward: 0.0, epsilon: 0.001
    episode: 688/800, reward: 22.2, epsilon: 0.001
    episode: 689/800, reward: 0.0, epsilon: 0.00099
    episode: 690/800, reward: 0.0, epsilon: 0.00099
    episode: 691/800, reward: 0.0, epsilon: 0.00099
    episode: 692/800, reward: 0.0, epsilon: 0.00099
    episode: 693/800, reward: 0.0, epsilon: 0.00099
    episode: 694/800, reward: 0.0, epsilon: 0.00099
    episode: 695/800, reward: 0.0, epsilon: 0.00099
    episode: 696/800, reward: 0.0, epsilon: 0.00099
    episode: 697/800, reward: 0.0, epsilon: 0.00099
    episode: 698/800, reward: 0.0, epsilon: 0.00099
    episode: 699/800, reward: 0.0, epsilon: 0.00099
    episode: 700/800, reward: 0.0, epsilon: 0.00099
    episode: 701/800, reward: 0.0, epsilon: 0.00099
    episode: 702/800, reward: 0.0, epsilon: 0.00099
    episode: 703/800, reward: 0.0, epsilon: 0.00099
    episode: 704/800, reward: 0.0, epsilon: 0.00099
    episode: 705/800, reward: 0.0, epsilon: 0.00099
    episode: 706/800, reward: 0.0, epsilon: 0.00099
    episode: 707/800, reward: 0.0, epsilon: 0.00099
    episode: 708/800, reward: 0.0, epsilon: 0.00099
    episode: 709/800, reward: 0.0, epsilon: 0.00099
    episode: 710/800, reward: 0.0, epsilon: 0.00099
    episode: 711/800, reward: 0.0, epsilon: 0.00099
    episode: 712/800, reward: 0.0, epsilon: 0.00099
    episode: 713/800, reward: 0.0, epsilon: 0.00099
    episode: 714/800, reward: 0.0, epsilon: 0.00099
    episode: 715/800, reward: 0.0, epsilon: 0.00099
    episode: 716/800, reward: 0.0, epsilon: 0.00099
    episode: 717/800, reward: 0.0, epsilon: 0.00099
    episode: 718/800, reward: 0.0, epsilon: 0.00099
    episode: 719/800, reward: 0.0, epsilon: 0.00099
    episode: 720/800, reward: 0.0, epsilon: 0.00099
    episode: 721/800, reward: 0.0, epsilon: 0.00099
    episode: 722/800, reward: 0.0, epsilon: 0.00099
    episode: 723/800, reward: 0.0, epsilon: 0.00099
    episode: 724/800, reward: 0.0, epsilon: 0.00099
    episode: 725/800, reward: 0.0, epsilon: 0.00099
    episode: 726/800, reward: 0.0, epsilon: 0.00099
    episode: 727/800, reward: 0.0, epsilon: 0.00099
    episode: 728/800, reward: -46.6, epsilon: 0.00099
    episode: 729/800, reward: 0.0, epsilon: 0.00099
    episode: 730/800, reward: 0.0, epsilon: 0.00099
    episode: 731/800, reward: 0.0, epsilon: 0.00099
    episode: 732/800, reward: 0.0, epsilon: 0.00099
    episode: 733/800, reward: 0.0, epsilon: 0.00099
    episode: 734/800, reward: 0.0, epsilon: 0.00099
    episode: 735/800, reward: 0.0, epsilon: 0.00099
    episode: 736/800, reward: 0.0, epsilon: 0.00099
    episode: 737/800, reward: 0.0, epsilon: 0.00099
    episode: 738/800, reward: 0.0, epsilon: 0.00099
    episode: 739/800, reward: 0.0, epsilon: 0.00099
    episode: 740/800, reward: 22.2, epsilon: 0.00099
    episode: 741/800, reward: 0.0, epsilon: 0.00099
    episode: 742/800, reward: 0.0, epsilon: 0.00099
    episode: 743/800, reward: 0.0, epsilon: 0.00099
    episode: 744/800, reward: 0.0, epsilon: 0.00099
    episode: 745/800, reward: 0.0, epsilon: 0.00099
    episode: 746/800, reward: 0.0, epsilon: 0.00099
    episode: 747/800, reward: 0.0, epsilon: 0.00099
    episode: 748/800, reward: 0.0, epsilon: 0.00099
    episode: 749/800, reward: 0.0, epsilon: 0.00099
    episode: 750/800, reward: 0.0, epsilon: 0.00099
    episode: 751/800, reward: 0.0, epsilon: 0.00099
    episode: 752/800, reward: 0.0, epsilon: 0.00099
    episode: 753/800, reward: 0.0, epsilon: 0.00099
    episode: 754/800, reward: 0.0, epsilon: 0.00099
    episode: 755/800, reward: 0.0, epsilon: 0.00099
    episode: 756/800, reward: 0.0, epsilon: 0.00099
    episode: 757/800, reward: 0.0, epsilon: 0.00099
    episode: 758/800, reward: 0.0, epsilon: 0.00099
    episode: 759/800, reward: 0.0, epsilon: 0.00099
    episode: 760/800, reward: 0.0, epsilon: 0.00099
    episode: 761/800, reward: 0.0, epsilon: 0.00099
    episode: 762/800, reward: 0.0, epsilon: 0.00099
    episode: 763/800, reward: 0.0, epsilon: 0.00099
    episode: 764/800, reward: 0.0, epsilon: 0.00099
    episode: 765/800, reward: 0.0, epsilon: 0.00099
    episode: 766/800, reward: 0.0, epsilon: 0.00099
    episode: 767/800, reward: 0.0, epsilon: 0.00099
    episode: 768/800, reward: 0.0, epsilon: 0.00099
    episode: 769/800, reward: 0.0, epsilon: 0.00099
    episode: 770/800, reward: 0.0, epsilon: 0.00099
    episode: 771/800, reward: 0.0, epsilon: 0.00099
    episode: 772/800, reward: 0.0, epsilon: 0.00099
    episode: 773/800, reward: 0.0, epsilon: 0.00099
    episode: 774/800, reward: 0.0, epsilon: 0.00099
    episode: 775/800, reward: 0.0, epsilon: 0.00099
    episode: 776/800, reward: 0.0, epsilon: 0.00099
    episode: 777/800, reward: 0.0, epsilon: 0.00099
    episode: 778/800, reward: 0.0, epsilon: 0.00099
    episode: 779/800, reward: 0.0, epsilon: 0.00099
    episode: 780/800, reward: 0.0, epsilon: 0.00099
    episode: 781/800, reward: 0.0, epsilon: 0.00099
    episode: 782/800, reward: 0.0, epsilon: 0.00099
    episode: 783/800, reward: 0.0, epsilon: 0.00099
    episode: 784/800, reward: 0.0, epsilon: 0.00099
    episode: 785/800, reward: 0.0, epsilon: 0.00099
    episode: 786/800, reward: 0.0, epsilon: 0.00099
    episode: 787/800, reward: 0.0, epsilon: 0.00099
    episode: 788/800, reward: 0.0, epsilon: 0.00099
    episode: 789/800, reward: 0.0, epsilon: 0.00099
    episode: 790/800, reward: 0.0, epsilon: 0.00099
    episode: 791/800, reward: 0.0, epsilon: 0.00099
    episode: 792/800, reward: 0.0, epsilon: 0.00099
    episode: 793/800, reward: 0.0, epsilon: 0.00099
    episode: 794/800, reward: 0.0, epsilon: 0.00099
    episode: 795/800, reward: 0.0, epsilon: 0.00099
    episode: 796/800, reward: -1.9, epsilon: 0.00099
    episode: 797/800, reward: 0.0, epsilon: 0.00099
    episode: 798/800, reward: 0.0, epsilon: 0.00099
    episode: 799/800, reward: 0.0, epsilon: 0.00099
    episode: 800/800, reward: 0.0, epsilon: 0.00099



```python
qtum_agent.test(epsilon=0)
```

    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    Action.HOLD
    0.0



```python

```
