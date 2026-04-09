package story

// Store is the persistence interface used by Tracker.
//
// Implementations must be safe for concurrent use; the Tracker serialises
// writes internally but may call View concurrently with other View calls.
//
// Key ordering must be lexicographic over the raw byte slice — this is what
// bbolt, LevelDB, and most embedded KV stores provide out of the box.
// Range scan methods rely on this ordering for efficient prefix and time-index lookups.
type Store interface {
	// View executes fn inside a read-only transaction.
	View(fn func(tx Tx) error) error

	// Update executes fn inside a read-write transaction.
	// Concurrent Update calls are serialised by the implementation.
	Update(fn func(tx Tx) error) error

	// Close flushes any buffered writes and releases resources.
	Close() error
}

// Tx is a single key-value transaction provided to Store.View and Store.Update callbacks.
// Tx values must not be used outside the callback they are passed to.
type Tx interface {
	// Get returns the value for key, or nil if the key does not exist.
	// The returned slice is only valid for the lifetime of the transaction.
	// Implementations that reuse internal buffers must return a copy.
	Get(key []byte) ([]byte, error)

	// Put writes key → value. value must not be empty (use Delete to remove a key).
	Put(key, value []byte) error

	// Delete removes key. It is not an error if key does not exist.
	Delete(key []byte) error

	// DeletePrefix removes all keys that begin with prefix.
	DeletePrefix(prefix []byte) error

	// ScanRange calls fn for every key in [from, to) in ascending key order.
	// Iteration stops when fn returns a non-nil error; that error is returned
	// by ScanRange.
	ScanRange(from, to []byte, fn func(key, val []byte) error) error

	// ScanPrefix calls fn for every key that begins with prefix, in ascending
	// key order. Iteration stops when fn returns a non-nil error.
	ScanPrefix(prefix []byte, fn func(key, val []byte) error) error
}
