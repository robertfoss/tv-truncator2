//! BK-tree index for 64-bit words under Hamming distance.
//!
//! Used to avoid O(n × m) scans when grouping rolling-hash sequences that are within a
//! Hamming-distance threshold of an existing representative.

use crate::hasher::hamming_distance;
use std::collections::HashMap;

/// BK-tree (Burkhard–Keller) for nearest-neighbor queries under Hamming distance.
pub(crate) struct HammingBkTree {
    root: Option<Box<Node>>,
}

struct Node {
    word: u64,
    children: HashMap<u32, Box<Node>>,
}

impl HammingBkTree {
    pub(crate) fn new() -> Self {
        Self { root: None }
    }

    /// Insert a representative hash. Duplicate inserts are ignored (same as identical word).
    pub(crate) fn insert(&mut self, word: u64) {
        if let Some(root) = &mut self.root {
            root.insert(word);
        } else {
            self.root = Some(Box::new(Node {
                word,
                children: HashMap::new(),
            }));
        }
    }

    /// All words in the tree within Hamming distance `<= max_distance` of `query`.
    pub(crate) fn search_within(&self, query: u64, max_distance: u32, out: &mut Vec<u64>) {
        if let Some(root) = &self.root {
            root.search_within(query, max_distance, out);
        }
    }
}

impl Node {
    fn insert(&mut self, word: u64) {
        let d = hamming_distance(self.word, word);
        if d == 0 {
            return;
        }
        if let Some(child) = self.children.get_mut(&d) {
            child.insert(word);
        } else {
            self.children.insert(
                d,
                Box::new(Node {
                    word,
                    children: HashMap::new(),
                }),
            );
        }
    }

    fn search_within(&self, query: u64, max_distance: u32, out: &mut Vec<u64>) {
        let d0 = hamming_distance(self.word, query);
        if d0 <= max_distance {
            out.push(self.word);
        }
        for (&edge_d, child) in &self.children {
            let dd = if d0 > edge_d {
                d0 - edge_d
            } else {
                edge_d - d0
            };
            if dd <= max_distance {
                child.search_within(query, max_distance, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_neighbor_within_radius() {
        let mut t = HammingBkTree::new();
        let a: u64 = 0xFFFF_FFFF_FFFF_FFF0;
        let b: u64 = 0xFFFF_FFFF_FFFF_FFF1;
        t.insert(a);
        let mut out = Vec::new();
        t.search_within(b, 2, &mut out);
        assert!(out.contains(&a));
    }

    #[test]
    fn ignores_far_points() {
        let mut t = HammingBkTree::new();
        t.insert(0x0000_0000_0000_0000u64);
        let mut out = Vec::new();
        t.search_within(0xFFFF_FFFF_FFFF_FFFF, 5, &mut out);
        assert!(out.is_empty());
    }
}
