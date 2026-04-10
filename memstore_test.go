package story

import (
	"encoding/json"
	"sort"
	"sync"
)

// memStore is an in-memory Store for use in tests.
type memStore struct {
	mu   sync.RWMutex
	data map[string][]byte
}

func newMemStore() *memStore {
	return &memStore{data: make(map[string][]byte)}
}

func (s *memStore) View(fn func(Tx) error) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return fn(&memTx{store: s})
}

func (s *memStore) Update(fn func(Tx) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return fn(&memTx{store: s})
}

func (s *memStore) Close() error { return nil }

// Compile-time interface checks.
var _ Store = (*memStore)(nil)

type memTx struct{ store *memStore }

var _ Tx = (*memTx)(nil)

func (t *memTx) Get(key []byte) ([]byte, error) {
	v, ok := t.store.data[string(key)]
	if !ok {
		return nil, nil
	}
	cp := make([]byte, len(v))
	copy(cp, v)
	return cp, nil
}

func (t *memTx) Put(key, value []byte) error {
	cp := make([]byte, len(value))
	copy(cp, value)
	t.store.data[string(key)] = cp
	return nil
}

func (t *memTx) Delete(key []byte) error {
	delete(t.store.data, string(key))
	return nil
}

func (t *memTx) DeletePrefix(prefix []byte) error {
	p := string(prefix)
	for k := range t.store.data {
		if len(k) >= len(p) && k[:len(p)] == p {
			delete(t.store.data, k)
		}
	}
	return nil
}

func (t *memTx) ScanRange(from, to []byte, fn func(key, val []byte) error) error {
	fromS, toS := string(from), string(to)
	for _, k := range t.sortedKeys() {
		if k < fromS || k >= toS {
			continue
		}
		v := t.store.data[k]
		cp := make([]byte, len(v))
		copy(cp, v)
		if err := fn([]byte(k), cp); err != nil {
			return err
		}
	}
	return nil
}

func (t *memTx) ScanPrefix(prefix []byte, fn func(key, val []byte) error) error {
	p := string(prefix)
	for _, k := range t.sortedKeys() {
		if len(k) < len(p) || k[:len(p)] != p {
			continue
		}
		v := t.store.data[k]
		cp := make([]byte, len(v))
		copy(cp, v)
		if err := fn([]byte(k), cp); err != nil {
			return err
		}
	}
	return nil
}

func (t *memTx) sortedKeys() []string {
	keys := make([]string, 0, len(t.store.data))
	for k := range t.store.data {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// jsonCodec[T] is a test Codec that uses encoding/json.
type jsonCodec[T any] struct{}

func (jsonCodec[T]) Encode(sig Signal[T]) ([]byte, error) { return json.Marshal(sig) }
func (jsonCodec[T]) Decode(b []byte) (Signal[T], error) {
	var sig Signal[T]
	return sig, json.Unmarshal(b, &sig)
}
