$$P_{d}(k)=A_{1}T_{a}(k)+B_{1}T_{a}(k-1)+C_{1}T_{m}(k-1)+D_{1}(k)$$

$P_d(k) = A_1 T_a(k) + B_1 T_a(k-1) + C_1 T_m(k-1) + D_1(k)$ (A5)

其中：

$\Delta = \text{COP} \left( 1 - e^{\eta T} \left( \frac{1}{c} + \frac{r_2}{r_1 - r_2} \right) + \frac{C_a}{r_1 - r_2} + \frac{1}{c} \right) + e^{\eta T} \left( \frac{C_a}{r_1 - r_2} \right)$ (A6)

$A_1 = -\frac{1}{\Delta}$ (A7)

$B_1 = -\frac{1}{\Delta} \left\{ e^{\eta T} \left[ \frac{r_2}{r_1 - r_2} - \frac{H_m + UA}{C_a(r_1 - r_2)} - 1 \right] - \frac{e^{\eta T}}{r_1 - r_2} \left( -r_2 - \frac{H_m + UA}{C_a} \right) \right\}$ (A8)

$C_1 = -\frac{1}{\Delta} \left[ e^{\eta T} \frac{H_m}{C_a(r_1 - r_2)} - \frac{e^{\eta T} H_m}{(r_1 - r_2)C_a} \right]$ (A9)

$D_1 = -\frac{1}{\Delta} \left\{ -\frac{(2Q_m + T_0 UA)}{c} + e^{\eta T} \left[ \frac{Q_m}{C_a(r_1 - r_2)} + \frac{r_2(2Q_m + T_0 UA)}{c(r_1 - r_2)} + \frac{T_0 UA}{C_a(r_1 - r_2)} + \frac{(2Q_m + T_0 UA)}{c} \right] - \frac{e^{\eta T}}{r_1 - r_2} \left[ \frac{Q_m}{C_a} + \frac{r_2(2Q_m + T_0 UA)}{c} + \frac{T_0 UA}{C_a} \right] \right\}$ (A10)

调度中心在汇集各虚拟机组的报价以及调峰容量后，求解如下最优调度问题：

$$
\begin{cases}
\min_{\Delta P_{a,i}} \sum_{i=1}^{M} F_i(\Delta P_{a,i}) \\
\text{s.t. } \sum_{i=1}^{M} \Delta P_{a,i} = P_{\text{target}} \\
0 \leq \Delta P_{a,i} \leq \Delta P_{a,i}^{\text{c}}
\end{cases}
$$  (33)

式中：$P_{\text{target}}$ 为总调峰功率；$M$ 为集群数量。
$$
