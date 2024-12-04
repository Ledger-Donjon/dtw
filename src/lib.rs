//! Implementation of dynamic time warping (DTW) algorithm and approximations.
//!
//! The DTW algorithm [1] finds the optimal alignment between two time series.
//!
//! This crate provides a DTW implementations as well as a FastDTW [2] implementation. FastDTW is
//! linear time and space approximation of DTW.
//!
//! # References
//! [1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
//! to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
//! Comparison.
//!
//! [2] Salvador, Stan & Chan, Philip. (2004). Toward Accurate Dynamic Time Warping in Linear Time
//! and Space. Intelligent Data Analysis. 11. 70-80.

use std::{
    cmp::{self, Ordering},
    ops::{Add, Range},
};

use num_traits::bounds::UpperBounded;

/// Compute the warp path of two given time series using DTW [1].
///
/// NOTE: See [`dtw_with_cmp`] for types that do implement [`Ord`].
///
/// # Examples
/// ```rust
/// use dtw::{dtw, dist::euclidean_distance};
///
/// let x = vec![0i32, 2, 3, 4, 5, 4, 3, 2, 0, 0, 0, 0];
/// let y = vec![0, 0, 0, 0, 2, 3, 4, 5, 4, 3, 2, 0];
///
/// let warp_path = dtw(&x, &y, &euclidean_distance);
/// ```
///
/// # Algorithmic complexity
/// Time complexity: O(N*M) (where N and M are the lengths of x and y respectively)
///
/// Space complexity: O(N*M) (where N and M are the lengths of x and y respectively)
///
/// # References
/// [1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
/// to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
/// Comparison.
pub fn dtw<T, D>(x: &[T], y: &[T], dist: &D) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy,
    T::Container: Ord,
    D: Fn(T, T) -> T,
{
    dtw_with_cmp(x, y, dist, &T::Container::cmp)
}

/// A constraint for the DTW search space, where each row is constrained to a single contiguous
/// segment.
///
/// Not every constraint can be represented that way, but it allows for a more efficient cost
/// matrix representation. However, it is sufficient to represent FastDTW constraints as well as
/// Sakoe-Chiba band and Itakura parallelogram.
#[derive(Debug)]
pub struct Constraint {
    /// Rows contiguous segment constraint.
    row_constraints: Vec<RowConstraint>,
}

impl Constraint {
    /// Create a new [`Constraint`] from the given row constraints.
    pub fn new(row_constraints: Vec<RowConstraint>) -> Self {
        Self { row_constraints }
    }

    /// Create a new [`Constraint`] where each row is full.
    pub fn full(width: usize, height: usize) -> Self {
        Self {
            row_constraints: (0..height)
                .map(|_| RowConstraint::new(0, width - 1))
                .collect(),
        }
    }
}

/// A constraint constraining a row to a closed line segment.
#[derive(Debug, PartialEq)]
pub struct RowConstraint {
    /// Start column included.
    start_col: usize,
    /// End column included.
    end_col: usize,
}

impl RowConstraint {
    /// Create a new [`RowConstraint`].
    ///
    /// The given constraint must satisfy `start_col <= end_col`.
    pub fn new(start_col: usize, end_col: usize) -> Self {
        Self { start_col, end_col }
    }
}

/// Compute the warp path of two given time series using DTW [1] with a constrained search space.
///
/// NOTE: See [`constrained_dtw_with_cmp`] for types that do implement [`Ord`].
///
/// # References
/// [1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
/// to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
/// Comparison.
pub fn constrained_dtw<T, D>(
    x: &[T],
    y: &[T],
    constraint: Constraint,
    dist: &D,
) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy,
    T::Container: Ord,
    D: Fn(T, T) -> T,
{
    constrained_dtw_with_cmp(x, y, constraint, dist, &T::Container::cmp)
}

/// Compute the warp path of two given time series using FastDTW [1].
///
/// FastDTW is a linear time and space approximation of the DTW algorithm.
///
/// NOTE: See [`fast_dtw_with_cmp`] for types that do implement [`Ord`].
///
/// # Examples
/// ```rust
/// use dtw::{fast_dtw, dist::euclidean_distance};
///
/// let x = vec![0, 2, 3, 4, 5, 4, 3, 2, 0, 0, 0, 0];
/// let y = vec![0, 0, 0, 0, 2, 3, 4, 5, 4, 3, 2, 0];
///
/// let warp_path = fast_dtw(&x, &y, 1, &euclidean_distance);
/// ```
///
/// # Algorithmic complexity
/// Time complexity: O(N) if the radius is a small constant compared to N (where N is the length of
/// x and y)
///
/// Space complexity: O(N) if the radius is a small constant compared to N (where N is the length
/// of x and y)
///
/// # References
/// [1] Salvador, Stan & Chan, Philip. (2004). Toward Accurate Dynamic Time Warping in Linear Time
/// and Space. Intelligent Data Analysis. 11. 70-80.
pub fn fast_dtw<T, D>(x: &[T], y: &[T], radius: usize, dist: &D) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy + Average,
    T::Container: Ord,
    D: Fn(T, T) -> T,
{
    fast_dtw_with_cmp(x, y, radius, dist, &T::Container::cmp)
}

/// Compute the warp path of two given time series using DTW [1].
///
/// # Examples
/// ```rust
/// use dtw::{dtw_with_cmp, dist::euclidean_distance};
///
/// let x = vec![0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0., 0., 0., 0.];
/// let y = vec![0., 0., 0., 0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0.];
///
/// let warp_path = dtw_with_cmp(&x, &y, &euclidean_distance, &f64::total_cmp);
/// ```
///
/// # Algorithmic complexity
/// Time complexity: O(N*M) (where N and M are the lengths of x and y respectively)
///
/// Space complexity: O(N*M) (where N and M are the lengths of x and y respectively)
///
/// # References
/// [1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
/// to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
/// Comparison.
pub fn dtw_with_cmp<T, D, C>(x: &[T], y: &[T], dist: &D, cmp: &C) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy,
    D: Fn(T, T) -> T,
    C: Fn(&T::Container, &T::Container) -> Ordering,
{
    constrained_dtw_with_cmp(x, y, Constraint::full(x.len(), y.len()), dist, cmp)
}

/// Compute the warp path of two given time series using DTW [1] with a constrained search space.
///
/// # References
/// [1] Kruskal, JB & Liberman, Mark. (1983). The symmetric time-warping problem: From continuous
/// to discrete. Time Warps, String Edits, and Macromolecules: The Theory and Practice of Sequence
/// Comparison.
pub fn constrained_dtw_with_cmp<T, D, C>(
    x: &[T],
    y: &[T],
    constraint: Constraint,
    dist: &D,
    cmp: &C,
) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy,
    D: Fn(T, T) -> T,
    C: Fn(&T::Container, &T::Container) -> Ordering,
{
    let mut cost_matrix = PartialMatrix::from_constraint(
        constraint.row_constraints,
        x.len(),
        y.len(),
        T::Container::max_value(),
    );

    // Fill first matrix element
    *cost_matrix.get_mut(0, 0).unwrap() = dist(x[0], y[0]).into();

    // Fill first matrix column to eliminate further checks in loop
    for row in 1..y.len() {
        if let Some(&bottom) = cost_matrix.get(0, row - 1) {
            if let Some(elem) = cost_matrix.get_mut(0, row) {
                *elem = T::Container::from(dist(x[0], y[row])) + bottom;
            }
        }
    }

    // Fill first matrix row to eliminate further checks in loop
    for col in 1..x.len() {
        if let Some(&left) = cost_matrix.get(col - 1, 0) {
            if let Some(elem) = cost_matrix.get_mut(col, 0) {
                *elem = T::Container::from(dist(x[col], y[0])) + left;
            }
        }
    }

    // Fill matrix row by row
    for row in 1..y.len() {
        for col in cost_matrix.partial_row_range(row).skip(1) {
            let left = *cost_matrix
                .get(col - 1, row)
                .unwrap_or(&T::Container::max_value());
            let bottom = *cost_matrix
                .get(col, row - 1)
                .unwrap_or(&T::Container::max_value());
            let bottom_left = *cost_matrix
                .get(col - 1, row - 1)
                .unwrap_or(&T::Container::max_value());

            let elem = cost_matrix.get_mut(col, row).unwrap();
            *elem = T::Container::from(dist(x[col], y[row]))
                + std::cmp::min_by(left, std::cmp::min_by(bottom, bottom_left, cmp), cmp);
        }
    }

    // Backtrack to find the warp path
    let mut warp_path = Vec::new();

    let mut col = x.len() - 1;
    let mut row = y.len() - 1;
    while col >= 1 && row >= 1 {
        warp_path.push((col, row));

        let left = cost_matrix
            .get(col - 1, row)
            .copied()
            .unwrap_or(T::Container::max_value());
        let bottom = cost_matrix
            .get(col, row - 1)
            .copied()
            .unwrap_or(T::Container::max_value());
        let bottom_left = cost_matrix
            .get(col - 1, row - 1)
            .copied()
            .unwrap_or(T::Container::max_value());

        if cmp(&left, &bottom).is_le() {
            if cmp(&bottom_left, &left).is_le() {
                col -= 1;
                row -= 1;
            } else {
                col -= 1;
            }
        } else if cmp(&bottom_left, &bottom).is_le() {
            col -= 1;
            row -= 1;
        } else {
            row -= 1;
        }
    }

    // Last row
    if row == 0 {
        while col >= 1 {
            warp_path.push((col, row));
            col -= 1;
        }
    }

    // Last column
    if col == 0 {
        while row >= 1 {
            warp_path.push((col, row));
            row -= 1;
        }
    }

    warp_path.push((0, 0));
    warp_path.reverse();

    warp_path
}

/// A partial matrix representation containing one segment per row.
struct PartialMatrix<T> {
    buffer: Box<[T]>,
    partial_rows: Box<[PartialRow]>,
    width: usize,
    height: usize,
}

impl<T> PartialMatrix<T>
where
    T: Copy,
{
    /// Create a new [`PartialMatrix`] from the given row constraints filled with the given element.
    fn from_constraint(
        row_constraints: Vec<RowConstraint>,
        width: usize,
        height: usize,
        elem: T,
    ) -> Self {
        assert_eq!(row_constraints.len(), height);

        let mut size = 0;
        let mut partial_rows = Vec::with_capacity(row_constraints.len());
        for i in 0..row_constraints.len() {
            assert!(row_constraints[i].start_col <= row_constraints[i].end_col);
            let range_size = row_constraints[i].end_col - row_constraints[i].start_col + 1;

            partial_rows.push(PartialRow {
                start_col: row_constraints[i].start_col,
                start_buf_idx: size,
                len: range_size,
            });

            size += range_size;
        }

        Self {
            buffer: vec![elem; size].into_boxed_slice(),
            partial_rows: partial_rows.into_boxed_slice(),
            width,
            height,
        }
    }

    fn get(&self, column: usize, row: usize) -> Option<&T> {
        Some(&self.buffer[self.index_of(column, row)?])
    }

    fn get_mut(&mut self, column: usize, row: usize) -> Option<&mut T> {
        Some(&mut self.buffer[self.index_of(column, row)?])
    }

    /// # Panics
    /// Panic if `column` or `row` is out of bound.
    fn index_of(&self, column: usize, row: usize) -> Option<usize> {
        assert!(column < self.height);
        assert!(row < self.width);

        if column < self.partial_rows[row].start_col
            || column >= self.partial_rows[row].start_col + self.partial_rows[row].len
        {
            return None;
        }

        Some((column - self.partial_rows[row].start_col) + self.partial_rows[row].start_buf_idx)
    }

    fn partial_row_range(&self, row: usize) -> Range<usize> {
        assert!(row < self.width);

        self.partial_rows[row].start_col
            ..self.partial_rows[row].start_col + self.partial_rows[row].len
    }
}

struct PartialRow {
    start_col: usize,
    start_buf_idx: usize,
    len: usize,
}

/// Provide an associated type to store sum.
pub trait SumContainer: Sized {
    /// Type that can hold sums of [`Self`].
    type Container: From<Self> + Add<Output = Self::Container> + UpperBounded + Copy;
}

macro_rules! impl_sum_container {
    ($($t:ty),* => $c:ty) => {
        $(
            impl SumContainer for $t {
                type Container = $c;
            }
        )*
    };
}

impl_sum_container! { i8, i16, i32, i64 => i64 }
impl_sum_container! { u8, u16, u32, u64 => u64 }
impl_sum_container! { f32, f64 => f64 }

/// Compute the warp path of two given time series using FastDTW [1].
///
/// FastDTW is a linear time and space approximation of the DTW algorithm.
///
/// # Examples
/// ```rust
/// use dtw::{fast_dtw_with_cmp, dist::euclidean_distance};
///
/// let x = vec![0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0., 0., 0., 0.];
/// let y = vec![0., 0., 0., 0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0.];
///
/// let warp_path = fast_dtw_with_cmp(&x, &y, 1, &euclidean_distance, &f64::total_cmp);
/// ```
///
/// # Algorithmic complexity
/// Time complexity: O(N) if the radius is a small constant compared to N (where N is the length of
/// x and y)
///
/// Space complexity: O(N) if the radius is a small constant compared to N (where N is the length
/// of x and y)
///
/// # References
/// [1] Salvador, Stan & Chan, Philip. (2004). Toward Accurate Dynamic Time Warping in Linear Time
/// and Space. Intelligent Data Analysis. 11. 70-80.
pub fn fast_dtw_with_cmp<T, D, C>(
    x: &[T],
    y: &[T],
    radius: usize,
    dist: &D,
    cmp: &C,
) -> Vec<(usize, usize)>
where
    T: SumContainer + Copy + Average,
    D: Fn(T, T) -> T,
    C: Fn(&T::Container, &T::Container) -> Ordering,
{
    // WARN: The case where `x.len() != y.len()` has not been tested.

    let dtw_threshold = radius + 2;
    if x.len() < dtw_threshold || y.len() < dtw_threshold {
        return dtw_with_cmp(x, y, dist, cmp);
    }

    let shrunk_x = x
        .chunks_exact(2)
        .map(|chunk| T::average(chunk[0], chunk[1]))
        .collect::<Vec<_>>();
    let shrunk_y = y
        .chunks_exact(2)
        .map(|chunk| T::average(chunk[0], chunk[1]))
        .collect::<Vec<_>>();

    let low_res_path = fast_dtw_with_cmp(&shrunk_x, &shrunk_y, radius, dist, cmp);

    let row_constraints = expanded_res_window(low_res_path, x.len(), y.len(), radius);

    constrained_dtw_with_cmp(x, y, Constraint::new(row_constraints), dist, cmp)
}

/// Compute constraints resulting from the expanded warp path and with the given radius.
fn expanded_res_window(
    low_res_path: Vec<(usize, usize)>,
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<RowConstraint> {
    let mut row_constraints: Vec<RowConstraint> = Vec::with_capacity(height);
    for i in 0..low_res_path.len() {
        let (col, row) = low_res_path[i];

        for row_offset in 0..=radius {
            let col_left = (col * 2).saturating_sub(radius);
            let col_right = cmp::min(col * 2 + 1 + radius, width - 1);

            if let Some(row_below) = (row * 2).checked_sub(row_offset) {
                if i == 0 {
                    row_constraints.push(RowConstraint::new(col_left, col_right));
                } else {
                    row_constraints[row_below].start_col =
                        cmp::min(row_constraints[row_below].start_col, col_left);
                    row_constraints[row_below].end_col =
                        cmp::max(row_constraints[row_below].end_col, col_right);
                }
            }

            let row_above = row * 2 + 1 + row_offset;
            if row_above < height {
                if row_above < row_constraints.len() {
                    row_constraints[row_above].start_col =
                        cmp::min(row_constraints[row_above].start_col, col_left);
                    row_constraints[row_above].end_col =
                        cmp::max(row_constraints[row_above].end_col, col_right);
                } else {
                    assert!(row_above == row_constraints.len());
                    row_constraints.push(RowConstraint::new(col_left, col_right));
                }
            }
        }

        if i + 1 < low_res_path.len() {
            let (next_col, next_row) = low_res_path[i + 1];

            // Add corner radius for diagonal. In the diagonal case, the expanded path is smoothed:
            //   ##
            //  o##
            // ##o
            // ##
            //
            // 'o's are added to smooth the path.
            if next_col == col + 1 && next_row == row + 1 {
                // Lower right 'o'
                if let Some(new_row) = (row * 2 + 1).checked_sub(radius) {
                    let new_col = col * 2 + 2 + radius;
                    if new_col < width {
                        row_constraints[new_row].end_col =
                            cmp::max(row_constraints[new_row].end_col, new_col);
                    }
                }

                // Upper left 'o'
                if let Some(new_col) = (col * 2 + 1).checked_sub(radius) {
                    let new_row = row * 2 + 2 + radius;
                    if new_row < height {
                        row_constraints.push(RowConstraint::new(
                            new_col,
                            cmp::min(col * 2 + 1 + radius, width - 1),
                        ));
                    }
                }
            }
        }
    }

    row_constraints
}

/// Types that can be averaged.
pub trait Average {
    /// Average two values.
    fn average(a: Self, b: Self) -> Self;
}

macro_rules! impl_average_int {
    ($($t:ty),*) => {
        $(
            impl Average for $t {
                fn average(a: Self, b: Self) -> Self {
                    a / 2 + b / 2 + (a % 2 + b % 2) / 2
                }
            }
        )*
    };
}

impl_average_int! { i8, u8, i16, u16, i32, u32, i64, u64, isize, usize }

impl Average for f32 {
    fn average(a: Self, b: Self) -> Self {
        (a + b) / 2.
    }
}

impl Average for f64 {
    fn average(a: Self, b: Self) -> Self {
        (a + b) / 2.
    }
}

/// Usual distance functions.
pub mod dist {
    /// Types that have an euclidean distance.
    pub trait EuclideanDistance {
        /// Euclidean distance between two values.
        ///
        /// # References
        /// https://en.wikipedia.org/wiki/Euclidean_distance
        fn euclidean_distance(x: Self, y: Self) -> Self;
    }

    macro_rules! impl_euclidean_distance_signed {
        ($($t:ty),*) => {
            $(
                impl EuclideanDistance for $t {
                    fn euclidean_distance(x: Self, y: Self) -> Self {
                        (x - y).abs()
                    }
                }
            )*
        };
    }

    impl_euclidean_distance_signed! { i8, i16, i32, i64, isize, f32, f64 }

    macro_rules! impl_euclidean_distance_unsigned_int {
        ($($t:ty),*) => {
            $(
                impl EuclideanDistance for $t {
                    fn euclidean_distance(x: Self, y: Self) -> Self {
                        if x > y {
                            x - y
                        } else {
                            y - x
                        }
                    }
                }
            )*
        };
    }

    impl_euclidean_distance_unsigned_int! { u8, u16, u32, u64, usize }

    /// Euclidean distance between two values.
    ///
    /// See [`EuclideanDistance`].
    pub fn euclidean_distance<T: EuclideanDistance>(x: T, y: T) -> T {
        T::euclidean_distance(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::{dist::*, *};

    #[test]
    fn test_dtw() {
        let x = vec![0u8, 2, 3, 4, 5, 4, 3, 2, 0, 0, 0, 0];
        let y = vec![0u8, 0, 0, 0, 2, 3, 4, 5, 4, 3, 2, 0];

        let warp_path = dtw(&x, &y, &euclidean_distance);

        assert_eq!(
            warp_path,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (4, 7),
                (5, 8),
                (6, 9),
                (7, 10),
                (8, 11),
                (9, 11),
                (10, 11),
                (11, 11)
            ],
        );
    }

    #[test]
    fn test_dtw_with_cmp() {
        let x = vec![0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0., 0., 0., 0.];
        let y = vec![0., 0., 0., 0., 0.2, 0.3, 0.4, 0.45, 0.4, 0.3, 0.2, 0.];

        let warp_path = dtw_with_cmp(&x, &y, &euclidean_distance, &f64::total_cmp);

        assert_eq!(
            warp_path,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (2, 5),
                (3, 6),
                (4, 7),
                (5, 8),
                (6, 9),
                (7, 10),
                (8, 11),
                (9, 11),
                (10, 11),
                (11, 11)
            ],
        );
    }

    #[test]
    fn test_expand_path_along_side() {
        // Expand the following path:
        // .#
        // ##
        assert_eq!(
            expanded_res_window(vec![(0, 0), (1, 0), (1, 1)], 4, 4, 1),
            vec![
                RowConstraint::new(0, 3),
                RowConstraint::new(0, 3),
                RowConstraint::new(0, 3),
                RowConstraint::new(1, 3),
            ]
        );
    }

    #[test]
    fn test_expand_diagonal() {
        // Expand the following path:
        // .#
        // #.
        assert_eq!(
            expanded_res_window(vec![(0, 0), (1, 1)], 4, 4, 1),
            vec![
                RowConstraint::new(0, 3),
                RowConstraint::new(0, 3),
                RowConstraint::new(0, 3),
                RowConstraint::new(0, 3),
            ]
        );
    }

    #[test]
    fn test_fast_dtw() {
        let x = vec![
            149u8, 251, 228, 63, 206, 0, 65, 63, 238, 215, 89, 56, 86, 184, 98, 167, 246, 234, 139,
            169,
        ];
        let y = vec![
            45u8, 115, 173, 239, 112, 90, 19, 30, 250, 51, 41, 174, 136, 184, 177, 234, 142, 8, 5,
            29,
        ];

        let warp_path = fast_dtw(&x, &y, 1, &euclidean_distance);
        assert_eq!(
            warp_path,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 7),
                (8, 8),
                (9, 8),
                (10, 9),
                (11, 9),
                (12, 10),
                (13, 11),
                (14, 12),
                (15, 13),
                (15, 14),
                (16, 15),
                (17, 15),
                (18, 16),
                (18, 17),
                (18, 18),
                (19, 19)
            ]
        );
    }

    #[test]
    fn test_fast_dtw_with_cmp() {
        let x = vec![
            149., 251., 228., 63., 206., 0., 65., 63., 238., 215., 89., 56., 86., 184., 98., 167.,
            246., 234., 139., 169.,
        ];
        let y = vec![
            45., 115., 173., 239., 112., 90., 19., 30., 250., 51., 41., 174., 136., 184., 177.,
            234., 142., 8., 5., 29.,
        ];

        let warp_path = fast_dtw_with_cmp(&x, &y, 1, &euclidean_distance, &f64::total_cmp);
        assert_eq!(
            warp_path,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 7),
                (8, 8),
                (9, 8),
                (10, 9),
                (11, 9),
                (12, 10),
                (13, 11),
                (14, 12),
                (15, 13),
                (15, 14),
                (16, 15),
                (17, 15),
                (18, 16),
                (18, 17),
                (18, 18),
                (19, 19)
            ]
        );
    }
}
