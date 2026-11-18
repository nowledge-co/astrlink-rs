//! Worker turns shared by the scans of one model.
//!
//! Every window that needs inference takes a turn before it reaches the pool,
//! and a model has one turn per worker. Of the scans waiting, the scan with the
//! fewest windows left to start takes the next turn; ties go to the scan that
//! arrived first. Magpie orders requests for its local model the same way
//! (least left to read first), so a short scan goes before the rest of a long
//! one instead of waiting for all of it.
//!
//! A window holds one turn and runs on one worker, except a lone window: when
//! every turn is free and no other scan waits, a scan's last window takes them
//! all and splits over every worker. Magpie's guard likewise gives a window one
//! core plus those free, so a lone short text is read on every core and texts
//! read at once take a core each. A split window must hold every turn, because
//! a rayon worker that waits for split work runs other queued windows in the
//! meantime, and the split window would resume only after a whole one.
//!
//! Waiting scans are admitted scans, so `max_in_flight` bounds this queue. A
//! steady stream of shorter scans can delay a longer one, as in Magpie.

use std::collections::BTreeSet;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// How often a waiting scan checks its stop condition. Cancellation is an
/// atomic flag without a wake-up, so waiting re-checks it at this period.
const STOP_CHECK: Duration = Duration::from_millis(20);

pub(crate) struct Turns {
    count: usize,
    state: Mutex<State>,
    changed: Condvar,
}

struct State {
    free: usize,
    arrivals: u64,
    /// (windows left to start, arrival) of each waiting scan, first in order.
    waiting: BTreeSet<(usize, u64)>,
}

/// Held turns. Dropping them, including during unwinding, returns them.
pub(crate) struct Turn<'a> {
    turns: &'a Turns,
    held: usize,
}

impl Turns {
    pub(crate) fn new(count: usize) -> Self {
        Self {
            count,
            state: Mutex::new(State {
                free: count,
                arrivals: 0,
                waiting: BTreeSet::new(),
            }),
            changed: Condvar::new(),
        }
    }

    /// Returns a new scan's arrival number, which breaks ties between scans.
    pub(crate) fn arrive(&self) -> u64 {
        let mut state = self.lock();
        state.arrivals += 1;
        state.arrivals
    }

    /// Waits for a turn for a scan with `left` windows still to start.
    ///
    /// Returns `None`, without a turn, once `stop` returns true.
    pub(crate) fn take(
        &self,
        left: usize,
        arrival: u64,
        stop: impl Fn() -> bool,
    ) -> Option<Turn<'_>> {
        let key = (left, arrival);
        let mut state = self.lock();
        state.waiting.insert(key);
        loop {
            if stop() {
                state.waiting.remove(&key);
                drop(state);
                // This scan may have been first; the next one may now go.
                self.changed.notify_all();
                return None;
            }
            if state.free > 0 && state.waiting.first() == Some(&key) {
                state.waiting.remove(&key);
                let lone = left == 1 && state.free == self.count && state.waiting.is_empty();
                let held = if lone { self.count } else { 1 };
                state.free -= held;
                drop(state);
                // The next waiting scan may find another free turn.
                self.changed.notify_all();
                return Some(Turn { turns: self, held });
            }
            state = self
                .changed
                .wait_timeout(state, STOP_CHECK)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> (usize, usize) {
        let state = self.lock();
        (state.free, state.waiting.len())
    }

    // No code panics while holding the lock, so a poisoned state is intact.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Turn<'_> {
    /// Whether the window may split over the pool: only when it holds every
    /// worker, and there is more than one.
    pub(crate) fn split(&self) -> bool {
        self.held > 1
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        self.turns.lock().free += self.held;
        self.turns.changed.notify_all();
    }
}
