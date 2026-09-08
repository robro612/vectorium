use crate::core::dataset::ScoredVector;
use crate::distances::Distance;

/// Generic trait for a vector index: searched with a query to return the
/// `k` nearest results.
///
/// This trait says nothing about how an index is built, building is
/// inherently type-directed (different indexes need different inputs and
/// configuration), so each index type exposes its own build entry point
/// (an inherent method or a free function) instead of a common trait method.
///
/// The query ([`Query`](Index::Query)) and the result distance
/// ([`Distance`](Index::Distance)) are the index's own associated types:
/// `search` receives the raw query and handles it however it wants.
pub trait Index {
    /// Raw query accepted by [`search`](Index::search). The query lifetime `'q`
    /// is independent of the index, so it imposes no `Self: 'q` bound.
    type Query<'q>;

    /// Distance/score type carried by each result.
    type Distance: Distance;

    /// Parameters for [`search`](Index::search).
    type SearchParams;

    /// Search the index with `query`, returning up to `k` scored results.
    fn search<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
    ) -> Vec<ScoredVector<Self::Distance>>;

    /// Search primed from caller-supplied entry points, given as IDs this index's own
    /// [`search`](Index::search) returns. Every seed must be a vector in the index.
    ///
    /// Seeds are a hint about *where to start looking*, so an index is always free to ignore
    /// them: the default does exactly that, and it is what indexes with no notion of a starting
    /// point (a flat scan, which already looks everywhere) should keep. Graph indexes override
    /// it, since for them the entry point is the whole cost model.
    ///
    /// Callers that iterate — refining an answer they already have, as k-means does when it
    /// reassigns points to moved centroids — can pass the previous round's answer here. An
    /// index that honours seeds should guarantee it never returns anything worse than the best
    /// seed it was given, which is what makes such a loop monotone rather than merely fast.
    ///
    /// An empty `seeds` must behave exactly as [`search`](Index::search).
    fn search_seeded<'q>(
        &self,
        query: Self::Query<'q>,
        k: usize,
        search_params: &Self::SearchParams,
        _seeds: &[usize],
    ) -> Vec<ScoredVector<Self::Distance>> {
        self.search(query, k, search_params)
    }
}

/// Optional metadata for indexes that store a fixed collection of vectors.
///
/// Kept separate from [`Index`] because we may have indexes where these
/// values are not known or not meaningful.
pub trait IndexStats {
    /// Number of indexed elements.
    ///
    /// For multivector datasets each element may contain several token vectors;
    /// this returns the element count (e.g. documents), not the token count.
    fn n_elements(&self) -> usize;

    /// Dimensionality of the indexed vectors.
    fn dim(&self) -> usize;
}
