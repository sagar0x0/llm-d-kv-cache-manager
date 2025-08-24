/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kvblock

import (
	"context"
	"fmt"
	"sync"

	"github.com/dgraph-io/ristretto/v2"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-kv-cache-manager/pkg/utils/logging"
)

const (
	// Ristretto defaults
	defaultMaxCost     = 1 << 30 // 1 GB (could be modified)
	defaultNumCounters = 1e7     // 10M counters, recommended by Ristretto for high-traffic caches.  (could be modified)
	// defaultBufferItems is 64, a Ristretto internal default.

	// Costing defaults for cache entries.
	// We assign a base cost for any key, plus an additional cost for each pod stored.
	baseCostPerKey = 1
	costPerPod     = 1
)

// InMemoryIndexConfig holds the configuration for the InMemoryIndex.
type InMemoryIndexConfig struct {
	// MaxCost is the maximum memory cost of the cache (e.g., in bytes).
	MaxCost int `json:"maxCost"`  // in_memory.go (its int)    ||   int64 can also be considered
	// NumCounters determines the number of counters for Ristretto's frequency tracking.
	NumCounters int `json:"numCounters"`   //     ||   int64 can also be considered
}

// DefaultInMemoryIndexConfig returns a default configuration for the InMemoryIndex.
func DefaultInMemoryIndexConfig() *InMemoryIndexConfig {
	return &InMemoryIndexConfig{
		MaxCost:     defaultMaxCost,
		NumCounters: defaultNumCounters,
	}
}

// PodSet is a thread-safe set of PodEntry objects.
type PodSet struct {
	mu   sync.RWMutex
	pods map[PodEntry]struct{}
}

// NewPodSet creates a new, empty PodSet.
func NewPodSet() *PodSet {
	return &PodSet{
		pods: make(map[PodEntry]struct{}),
	}
}

// Add adds multiple entries to the set.
func (ps *PodSet) Add(entries []PodEntry) {
	ps.mu.Lock()   // Acquires a full write lock
	defer ps.mu.Unlock() // Ensures the lock is released when the function exits.
	for _, entry := range entries {
		ps.pods[entry] = struct{}{}
	}
}

// Remove removes multiple entries from the set.
func (ps *PodSet) Remove(entries []PodEntry) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	for _, entry := range entries {
		delete(ps.pods, entry)
	}
}

// GetPods returns a slice of all pod identifiers in the set.
func (ps *PodSet) GetPods() []string {
	ps.mu.RLock()   // read-only lock
	defer ps.mu.RUnlock()
	podIdentifiers := make([]string, 0, len(ps.pods))
	for entry := range ps.pods {
		podIdentifiers = append(podIdentifiers, entry.PodIdentifier)
	}
	return podIdentifiers
}

// Len returns the number of entries in the set.
func (ps *PodSet) Len() int {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	return len(ps.pods)
}

// InMemoryIndex is an in-memory implementation of the Index interface using Ristretto.
type InMemoryIndex struct {
	// data holds the mapping of keys to sets of pod identifiers.
	data *ristretto.Cache
}

var _ Index = &InMemoryIndex{}   // InMemoryIndex implements Index interface

// NewInMemoryIndex creates a new InMemoryIndex instance.
func NewInMemoryIndex(cfg *InMemoryIndexConfig) (*InMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultInMemoryIndexConfig()
	}

	config := &ristretto.Config{
		NumCounters: cfg.NumCounters,
		MaxCost:     cfg.MaxCost,
		BufferItems: 64, // Recommended default by Ristretto.
	}

	cache, err := ristretto.NewCache(config)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory index (ristretto): %w", err)
	}

	return &InMemoryIndex{
		data: cache,
	}, nil
}

// Lookup receives a list of keys and retrieves the pods associated with those keys.
func (m *InMemoryIndex) Lookup(ctx context.Context, keys []Key,
	podIdentifierSet sets.Set[string],
) (map[Key][]string, error) {
	if len(keys) == 0 {
		return nil, fmt.Errorf("no keys provided for lookup")
	}

	traceLogger := klog.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Lookup")
	podsPerKey := make(map[Key][]string)

	for _, key := range keys {
		value, found := m.data.Get(key)
		if !found {
			// Early stop since the prefix-chain breaks here.
			traceLogger.Info("key not found in index, cutting search", "key", key)
			return podsPerKey, nil
		}

		podSet, ok := value.(*PodSet)   // type assertion to tackle data corruption 
		if !ok || podSet.Len() == 0 {
			traceLogger.Info("no pods found for key, cutting search", "key", key)
			return podsPerKey, nil
		}

		// Ristretto's Get does not provide an explicit way to know if it was a cache hit for LRU purposes
		// in the same way the old library did. We rely on its internal LFU policy.

		pods := podSet.GetPods()
		if podIdentifierSet.Len() == 0 {
			podsPerKey[key] = pods
		} else {
			// Filter pods based on the provided pod identifiers
			filteredPods := make([]string, 0, len(pods))
			for _, podIdentifier := range pods {
				if podIdentifierSet.Has(podIdentifier) {
					filteredPods = append(filteredPods, podIdentifier)
				}
			}
			if len(filteredPods) > 0 {
				podsPerKey[key] = filteredPods
			}
		}
	}

	traceLogger.Info("lookup completed", "pods-per-key", podsPerKeyPrintHelper(podsPerKey))
	return podsPerKey, nil
}

// Add adds a set of keys and their associated pod entries to the index.
func (m *InMemoryIndex) Add(ctx context.Context, keys []Key, entries []PodEntry) error {
	if len(keys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}

	traceLogger := klog.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Add")

	for _, key := range keys {
		value, found := m.data.Get(key)
		var podSet *PodSet
		if found {
			var ok bool
			podSet, ok = value.(*PodSet)
			if !ok {
				// This case should ideally not happen if the cache only stores *PodSet.
				// It indicates a type corruption. We overwrite it.
				traceLogger.Error(fmt.Errorf("cache entry for key is not a PodSet"), "overwriting entry", "key", key)
				podSet = NewPodSet()
			}
		} else {
			podSet = NewPodSet()
		}

		// Add entries to the set. This is thread-safe.
		podSet.Add(entries)

		// Calculate the cost and set the value in the cache. Set is an atomic operation.
		cost := int64(baseCostPerKey + (podSet.Len() * costPerPod))
		m.data.Set(key, podSet, cost)

		traceLogger.Info("added pods to key", "key", key, "pods", entries, "new-cost", cost)
	}

	return nil
}

// Evict removes pod entries associated with a key from the index.
func (m *InMemoryIndex) Evict(ctx context.Context, key Key, entries []PodEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	traceLogger := klog.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Evict")

	value, found := m.data.Get(key)
	if !found {
		traceLogger.Info("key not found in index, nothing to evict", "key", key)
		return nil
	}

	podSet, ok := value.(*PodSet)
	if !ok {
		traceLogger.Error(fmt.Errorf("cache entry for key is not a PodSet"), "cannot evict", "key", key)
		return nil
	}

	// Remove entries from the set. This is thread-safe.
	podSet.Remove(entries)
	traceLogger.Info("evicted pods from key", "key", key, "pods", entries)

	if podSet.Len() == 0 {
		m.data.Del(key)
		traceLogger.Info("evicted key from index as no pods remain", "key", key)
	} else {
		// Update the cost in the cache
		cost := int64(baseCostPerKey + (podSet.Len() * costPerPod))
		m.data.Set(key, podSet, cost)
	}

	return nil
}

// podsPerKeyPrintHelper formats a map of keys to pod names for printing.
func podsPerKeyPrintHelper(ks map[Key][]string) string {
	// This function remains the same as it's a utility for logging.
	flattened := ""
	for k, v := range ks {
		flattened += fmt.Sprintf("%s: %v\n", k.String(), v)
	}
	return flattened
}