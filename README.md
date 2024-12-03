# dtw
Implementation of dynamic time warping (DTW) algorithm and approximations.

The DTW algorithm [1] finds the optimal alignment between two time series.

This crate provides a DTW implementations as well as a FastDTW [2] implementation. FastDTW is
linear time and space approximation of DTW.

## References
[1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
Comparison.

[2] Salvador, Stan & Chan, Philip. (2004). Toward Accurate Dynamic Time Warping in Linear Time
and Space. Intelligent Data Analysis. 11. 70-80.
